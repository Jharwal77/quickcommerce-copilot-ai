export interface ToolCall {
  name: string;
  arguments: Record<string, string | number | boolean | null>;
  ok: boolean;
  error: string | null;
  latency_ms: number;
}

export interface Citation {
  marker: number;
  chunk_id: string;
  doc_id: string;
  label: string;
  score: number;
}

export interface EvidenceItem {
  marker: number;
  label: string;
  doc_id: string;
  kind: "passage" | "tool";
  text: string;
  score: number;
}

export type TraceKind = "retrieve" | "tool" | "reflect" | "guardrail" | "answer";

export interface TraceStep {
  kind: TraceKind;
  label: string;
  detail: string;
  ok: boolean;
  latency_ms: number;
  arguments: Record<string, string | number | boolean | null>;
  markers: number[];
}

export interface GuardrailReport {
  reflection_supported: boolean | null;
  unsupported_claims: string[];
  dropped_sentences: number;
  removed_markers: number;
  tool_calls_made: number;
  tool_call_cap: number;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
  estimated_cost_usd: number;
  models: string[];
}

export interface AskResponse {
  question: string;
  answer: string;
  citations: Citation[];
  contexts: string[];
  retrieved_doc_ids: string[];
  tool_calls: ToolCall[];
  evidence: EvidenceItem[];
  trace: TraceStep[];
  mode: "retrieval" | "agentic" | "system";
  refused: boolean;
  guardrails: GuardrailReport | null;
  usage: Usage;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  collection: string;
  vectors: number | null;
  llm_configured: boolean;
  detail: string | null;
}

export interface PolicySummary {
  doc_id: string;
  title: string;
  sections: string[];
}

export interface StoreSummary {
  store_id: string;
  name: string;
  city: string;
}

export interface KnowledgeSummary {
  policies: PolicySummary[];
  dark_stores: StoreSummary[];
  categories: string[];
  brands: string[];
  product_count: number;
  chunk_count: number | null;
}

export interface EvalHeadline {
  label: string | null;
  mode: string | null;
  rows: number | null;
  tool_rows: number | null;
  tool_call_success_rate: number | null;
  citation_coverage: number | null;
  hallucination_rate: number | null;
  faithfulness: number | null;
  answer_relevancy: number | null;
  context_precision: number | null;
  context_recall: number | null;
  p95_latency_ms: number | null;
  created_at: string | null;
}

export interface EvalPair {
  baseline?: EvalHeadline;
  after?: EvalHeadline;
}

export interface MetaResponse {
  service: string;
  version: string;
  description: string;
  links: Record<string, string>;
  mode: string;
  tools: string[];
  answer_chain: string[];
  reflect_chain: string[];
  tool_call_cap: number;
  knowledge: KnowledgeSummary;
  eval: Record<string, EvalPair>;
  synthetic_data: boolean;
}

export type ApiFailureKind = "budget" | "unavailable" | "invalid" | "network" | "server";

export class ApiError extends Error {
  readonly kind: ApiFailureKind;
  readonly status: number | null;

  constructor(kind: ApiFailureKind, message: string, status: number | null) {
    super(message);
    this.kind = kind;
    this.status = status;
  }
}

async function readDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail;
      return typeof detail === "string" ? detail : JSON.stringify(detail);
    }
  } catch {
    // Non-JSON error bodies fall through to the status text.
  }
  return response.statusText || `HTTP ${response.status}`;
}

function classify(status: number, detail: string): ApiFailureKind {
  if (status === 503 && /budget|exhausted|credit/i.test(detail)) return "budget";
  if (status === 503) return "unavailable";
  if (status === 422 || status === 400) return "invalid";
  return "server";
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url, init);
  } catch (error) {
    const message = error instanceof Error ? error.message : "network failure";
    throw new ApiError("network", message, null);
  }
  if (!response.ok) {
    const detail = await readDetail(response);
    throw new ApiError(classify(response.status, detail), detail, response.status);
  }
  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>(`${API_BASE_URL}/health`);
}

export function getMeta(): Promise<MetaResponse> {
  return request<MetaResponse>(`${API_BASE_URL}/meta`);
}

export function ask(question: string, signal?: AbortSignal): Promise<AskResponse> {
  return request<AskResponse>(`${API_BASE_URL}/ask`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
}



