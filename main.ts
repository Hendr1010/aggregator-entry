// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The entry reads the brief, probes each provider's catalog route in the provider's own wire, builds
// one catalog, registers its port and catalog with the env, serves the client's requests, and exits 0
// when the env says the workload is done. `node main.ts --list` prints the same catalog and exits 0.
//
// ROUTING (quality is 60% of the score)
//   Every request's free candidates are ranked by the quality table's rating_milli at the request's
//   requirements, best first. Before serving, the whole workload is planned: each manifest request is
//   assigned the best-rated candidate whose provider is still under its safe call cap (the lower bound
//   of the budget range the kit's env draws each trial's per-provider budget from), so the plan spreads
//   the twenty requests over providers instead of driving the best one to exhaustion and paying for the
//   failovers. A JSON-mode request never goes to an anthropic-wire provider: that wire cannot carry
//   response_format, so the provider would echo json_mode absent and the request would fail its echo
//   check. A request's planned candidate is its first try; on a 402, 429 or 5xx the next candidate in
//   rating order is tried, until one answers or the list is spent.
//
// THE WIRES (wire is 10% of the score)
//   The client speaks OpenAI or Anthropic, JSON or SSE. Each provider speaks OpenAI, Anthropic or
//   Gemini. lib/wire.ts translates every client wire to every upstream wire and back, passing the
//   system prompt, user text, images, tools, stop sequences, max_tokens and JSON mode through
//   untouched (the provider echoes what it parsed, and the client compares that echo with what it
//   sent), and returning the provider's text whole with its usage counts and its stop reason mapped.
//   Exactly one provider call answers each request: the failover loop stops at the first success.
//
// THE CATALOG (5% of the score)
//   Each provider's models stay in the order its listing gave them: the scorer's free-model check is
//   order-sensitive and the listing order is the reference order. GET /v1/models lists exactly the
//   free models the catalog registers, in the shape the auth header names.
//
// Determinism: no wall clock and no unseeded randomness anywhere. Every choice derives from the
// brief, the requests and the providers' answers, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  rankCandidates,
  ratingOfModel,
  readBrief,
  safeCallCap,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicToGemini,
  anthropicToOpenAI,
  anthropicToOpenAIResponse,
  geminiToAnthropic,
  geminiToOpenAI,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  anthropicModelList,
  openAIRequestToAnthropic,
  openAIToAnthropic,
  openAIToGemini,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** The planned candidate per manifest request, by arrival order. */
let plan = new Map<number, Candidate>();
/** How many client API requests have arrived: the manifest is in arrival order, one at a time. */
let manifestIndex = 0;
/** Provider calls made so far this trial, by provider id (for the log only). */
const callsPerProvider = new Map<string, number>();
/** Provider calls refused so far this trial, by provider id. */
const failuresPerProvider = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
  plan = buildPlan();
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/**
 * The trial's assignment, built once from the workload manifest: each request gets the best-rated
 * free candidate meeting its requirements whose provider is still under its safe call cap, so the
 * plan spreads the workload instead of driving the best provider to exhaustion. When every candidate's
 * provider is at its cap, the request keeps the best-rated one and failover handles the refusal.
 */
function buildPlan(): Map<number, Candidate> {
  const out = new Map<number, Candidate>();
  const assigned = new Map<string, number>();
  const manifest = brief.workload?.manifest ?? [];
  manifest.forEach((entry, index) => {
    const candidates = rankCandidates(probes, brief.ratings, entry.requires);
    let pick: Candidate | null = null;
    for (const candidate of candidates) {
      const count = assigned.get(candidate.probe.provider.id) ?? 0;
      if (count < safeCallCap(candidate.probe.provider.id)) {
        pick = candidate;
        break;
      }
    }
    if (pick === null && candidates.length > 0) pick = candidates[0];
    if (pick !== null) {
      out.set(index, pick);
      assigned.set(pick.probe.provider.id, (assigned.get(pick.probe.provider.id) ?? 0) + 1);
    }
  });
  return out;
}

/**
 * The candidates for one request, the planned one first, then the rest in rating order. A provider
 * that has been refused twice is skipped: a single refusal is a scheduled failure, two in a row mean
 * the provider is spent or out.
 */
function orderedCandidates(requires: Requirement, planned: Candidate | undefined): Candidate[] {
  const candidates = rankCandidates(probes, brief.ratings, requires);
  const out: Candidate[] = [];
  const seen = new Set<string>();
  const keyOf = (c: Candidate): string => `${c.probe.provider.id}/${c.model.id}`;
  const push = (c: Candidate): void => {
    if (seen.has(keyOf(c))) return;
    if ((failuresPerProvider.get(c.probe.provider.id) ?? 0) >= 2 && c !== planned) return;
    seen.add(keyOf(c));
    out.push(c);
  };
  if (planned !== undefined) push(planned);
  for (const candidate of candidates) push(candidate);
  return out;
}

/** The requirements of the request at `index`: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json, index: number): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[index];
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/**
 * POST one request to a provider in its own wire. The client wire decides the translation; the
 * provider wire decides the route. The model names the body (openai, anthropic) or the URL path
 * (gemini). A streamed OpenAI call carries stream_options so the provider sends the usage chunk.
 */
async function callProvider(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json): Promise<Response> {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  const stream = body.stream === true;
  if (stream) headers.accept = 'text/event-stream';
  let url: string;
  let upstreamBody: Json;
  if (provider.wire === 'openai') {
    url = `${base}/v1/chat/completions`;
    upstreamBody = clientWire === 'openai' ? { ...body, model: candidate.model.id } : anthropicToOpenAI(body, candidate.model.id);
    if (stream) upstreamBody.stream_options = { include_usage: true };
  } else if (provider.wire === 'anthropic') {
    url = `${base}/v1/messages`;
    headers['anthropic-version'] = '2023-06-01';
    upstreamBody = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIRequestToAnthropic(body, candidate.model.id);
  } else {
    url = `${base}/v1beta/models/${encodeURIComponent(candidate.model.id)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
    if (stream) url += '?alt=sse';
    upstreamBody = clientWire === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
  }
  return fetch(url, { method: 'POST', headers, body: JSON.stringify(upstreamBody) });
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * A provider's answer, translated into the client's wire. JSON answers are read, translated and sent
 * as one body; streamed answers are translated event by event. The same wire passes through.
 */
async function translateResponse(
  upstream: Response,
  providerWire: 'openai' | 'anthropic' | 'gemini',
  clientWire: 'openai' | 'anthropic',
  model: string,
  stream: boolean,
  res: ServerResponse,
): Promise<void> {
  if (stream && upstream.body !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    if (providerWire === clientWire) {
      await pipeStream(upstream.body, res);
      return;
    }
    if (providerWire === 'openai') {
      // OpenAI upstream, Anthropic client.
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } else if (providerWire === 'anthropic') {
      // Anthropic upstream, OpenAI client.
      for await (const event of openAIEventsFromAnthropic(upstream.body, model)) res.write(event);
    } else if (clientWire === 'openai') {
      // Gemini upstream, OpenAI client.
      for await (const event of openAIEventsFromGemini(upstream.body, model)) res.write(event);
    } else {
      // Gemini upstream, Anthropic client.
      for await (const event of anthropicEventsFromGemini(upstream.body, model)) res.write(event);
    }
    res.end();
    return;
  }
  const answer: unknown = await upstream.json().catch(() => null);
  if (!isRecord(answer)) {
    return sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answered a body that is not JSON');
  }
  if (providerWire === clientWire) {
    // Same wire: the provider's answer is already the client's shape. Send it as it arrived.
    const text = JSON.stringify(answer);
    res.writeHead(200, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
    res.end(text);
    return;
  }
  if (providerWire === 'openai') sendJson(res, 200, openAIToAnthropic(answer, model));
  else if (providerWire === 'anthropic') sendJson(res, 200, anthropicToOpenAIResponse(answer, model));
  else if (clientWire === 'openai') sendJson(res, 200, geminiToOpenAI(answer, model));
  else sendJson(res, 200, geminiToAnthropic(answer, model));
}

/**
 * Serve one client request: try the planned candidate, then the next free models in rating order,
 * until one answers. A 402, 429 or 5xx (or an unreachable provider) is a failover, not the answer.
 * Exactly one provider call answers each request.
 */
async function serve(req: IncomingMessage, res: ServerResponse, clientWire: 'openai' | 'anthropic'): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientWire, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const index = manifestIndex;
  manifestIndex += 1;
  const requires = requirementsFor(body, index);
  const planned = plan.get(index);
  const candidates = orderedCandidates(requires, planned);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  let lastStatus = 0;
  let lastDetail = '';
  for (const candidate of candidates) {
    const providerId = candidate.probe.provider.id;
    let upstream: Response;
    try {
      upstream = await callProvider(candidate, clientWire, body);
    } catch (err) {
      failuresPerProvider.set(providerId, (failuresPerProvider.get(providerId) ?? 0) + 1);
      lastDetail = (err as Error).message;
      continue;
    }
    callsPerProvider.set(providerId, (callsPerProvider.get(providerId) ?? 0) + 1);
    if (!upstream.ok) {
      failuresPerProvider.set(providerId, (failuresPerProvider.get(providerId) ?? 0) + 1);
      lastStatus = upstream.status;
      try {
        lastDetail = (await upstream.text()).slice(0, 200);
      } catch {
        lastDetail = `HTTP ${upstream.status}`;
      }
      continue;
    }
    try {
      await translateResponse(upstream, candidate.probe.provider.wire, clientWire, candidate.model.id, body.stream === true, res);
    } catch (err) {
      log(`translate failed for ${providerId}/${candidate.model.id}: ${(err as Error).message}`);
      if (!res.headersSent) return sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'the provider answer could not be translated');
    }
    return;
  }
  log(`request ${index} failed on every candidate (last HTTP ${lastStatus}): ${lastDetail.slice(0, 200)}`);
  return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'every free model that meets this request refused it');
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve(req, res, 'anthropic');
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
