import type { AskResponse, TraceStep } from "../lib/api";
import { argumentsText, ms, shortModel } from "../lib/format";

interface Props {
  response: AskResponse | null;
  onMarker: (marker: number) => void;
}

const KIND_LABEL: Record<TraceStep["kind"], string> = {
  retrieve: "RETRIEVE",
  tool: "TOOL",
  reflect: "REFLECT",
  guardrail: "GUARD",
  answer: "ANSWER",
};

function kindColor(step: TraceStep): string {
  if (!step.ok) return "text-warn-500";
  if (step.kind === "tool") return "text-tool-500";
  if (step.kind === "retrieve") return "text-signal-500";
  return "text-ok-500";
}

function Step({ step, index, onMarker }: { step: TraceStep; index: number; onMarker: (m: number) => void }) {
  const args = argumentsText(step.arguments);
  return (
    <li className="relative pl-6">
      <span
        aria-hidden="true"
        className={`absolute top-1.5 left-0 h-2 w-2 rounded-full ${step.ok ? "bg-current" : "bg-warn-500"} ${kindColor(step)}`}
      />
      <div className="flex flex-wrap items-baseline gap-x-2">
        <span className={`font-mono text-[0.65rem] font-semibold tracking-wider ${kindColor(step)}`}>
          {index + 1}. {KIND_LABEL[step.kind]}
        </span>
        <span className="text-sm">{step.label}</span>
        {step.latency_ms > 0 && <span className="muted font-mono text-xs">{ms(step.latency_ms)}</span>}
      </div>
      {step.detail && (
        <p className="muted mt-0.5 font-mono text-xs break-words" title={step.detail}>
          {step.kind === "retrieve" ? `query: “${step.detail}”` : step.detail}
        </p>
      )}
      {args && step.kind === "tool" && (
        <p className="muted mt-0.5 font-mono text-xs break-all">{args}</p>
      )}
      {step.markers.length > 0 && (
        <div className="mt-1 flex flex-wrap gap-1">
          {step.markers.map((m) => (
            <button
              key={m}
              type="button"
              className="cite"
              onClick={() => onMarker(m)}
              aria-label={`Show evidence ${m}`}
            >
              {m}
            </button>
          ))}
        </div>
      )}
    </li>
  );
}

export function TracePanel({ response, onMarker }: Props) {
  if (!response) {
    return (
      <p className="muted text-sm">
        Ask something and the agent’s steps appear here: what it retrieved, which tools it called with what arguments, how the reflection check judged the draft, and whether the citation guardrail removed anything.
      </p>
    );
  }
  const answerModels = response.usage.models.filter((m) => !m.startsWith("reflect="));
  const reflectModels = response.usage.models.filter((m) => m.startsWith("reflect="));
  return (
    <div className="space-y-4">
      <ol className="relative space-y-3 border-l pl-1 hairline ml-1">
        {response.trace.map((step, i) => (
          <Step key={`${i}-${step.kind}`} step={step} index={i} onMarker={onMarker} />
        ))}
      </ol>
      <dl className="hairline grid grid-cols-2 gap-x-4 gap-y-1 border-t pt-3 font-mono text-xs sm:grid-cols-3">
        <dt className="muted">mode</dt>
        <dd className="sm:col-span-2">{response.mode}</dd>
        <dt className="muted">answer model</dt>
        <dd className="sm:col-span-2">{answerModels.map(shortModel).join(" → ") || "n/a"}</dd>
        {reflectModels.length > 0 && (
          <>
            <dt className="muted">reflection model</dt>
            <dd className="sm:col-span-2">{reflectModels.map(shortModel).join(" → ")}</dd>
          </>
        )}
        <dt className="muted">tool calls</dt>
        <dd className="sm:col-span-2">
          {response.guardrails ? `${response.guardrails.tool_calls_made} of ${response.guardrails.tool_call_cap} allowed` : response.tool_calls.length}
        </dd>
        <dt className="muted">latency</dt>
        <dd className="sm:col-span-2">
          {ms(response.usage.total_ms)} total · {ms(response.usage.retrieval_ms)} retrieval · {ms(response.usage.generation_ms)} generation
        </dd>
        <dt className="muted">tokens</dt>
        <dd className="sm:col-span-2">
          {response.usage.total_tokens} (≈ ${response.usage.estimated_cost_usd.toFixed(5)} at list price)
        </dd>
      </dl>
    </div>
  );
}
