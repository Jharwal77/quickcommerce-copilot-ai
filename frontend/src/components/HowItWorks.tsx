import { useEffect, useRef } from "react";
import type { EvalHeadline, MetaResponse } from "../lib/api";
import { percent, score } from "../lib/format";

interface Props {
  open: boolean;
  onClose: () => void;
  meta: MetaResponse | null;
}

function Loop() {
  const boxes = ["decide", "retrieve or call a tool", "reflect", "cite"];
  return (
    <ol className="flex flex-wrap items-center gap-2 font-mono text-xs" aria-label="Agent loop">
      {boxes.map((label, i) => (
        <li key={label} className="flex items-center gap-2">
          <span className="hairline rounded border px-2 py-1">{label}</span>
          {i < boxes.length - 1 && <span aria-hidden="true" className="muted">→</span>}
        </li>
      ))}
      <li className="muted">↺ back to decide until the evidence is enough or the cap is hit</li>
    </ol>
  );
}

function Row({ name, baseline, after, fmt }: { name: string; baseline: number | null | undefined; after: number | null | undefined; fmt: (v: number | null | undefined) => string }) {
  return (
    <tr>
      <td className="py-1 pr-3">{name}</td>
      <td className="py-1 pr-3 text-right font-mono">{fmt(baseline)}</td>
      <td className="py-1 text-right font-mono">{fmt(after)}</td>
    </tr>
  );
}

function Results({ meta }: { meta: MetaResponse }) {
  // The full 60-row pair is the headline once both reports exist; the 30-row subset was the interim.
  const pair = meta.eval.full_60?.after ? meta.eval.full_60 : (meta.eval.subset_30 ?? meta.eval.full_60);
  if (!pair?.baseline) return null;
  const b: EvalHeadline = pair.baseline;
  const a: EvalHeadline | undefined = pair.after;
  const rows = b.rows ?? 0;
  return (
    <section>
      <h3 className="text-sm font-semibold">Measured on the committed golden set</h3>
      <p className="muted mt-1 text-xs">
        Retrieval-only baseline versus the agentic pipeline with guardrails, {rows} rows ({b.tool_rows ?? 0} need a live tool).
        {a ? "" : " The agentic run is still being judged; only the baseline is committed so far."}
      </p>
      <table className="mt-2 w-full text-xs">
        <thead>
          <tr className="muted font-mono text-[0.65rem] tracking-wider uppercase">
            <th className="py-1 pr-3 text-left font-normal">metric</th>
            <th className="py-1 pr-3 text-right font-normal">baseline</th>
            <th className="py-1 text-right font-normal">agentic</th>
          </tr>
        </thead>
        <tbody>
          <Row name="tool-call success" baseline={b.tool_call_success_rate} after={a?.tool_call_success_rate} fmt={percent} />
          <Row name="citation coverage" baseline={b.citation_coverage} after={a?.citation_coverage} fmt={percent} />
          <Row name="hallucination rate" baseline={b.hallucination_rate} after={a?.hallucination_rate} fmt={percent} />
          <Row name="faithfulness (RAGAS)" baseline={b.faithfulness} after={a?.faithfulness} fmt={score} />
          <Row name="answer relevancy" baseline={b.answer_relevancy} after={a?.answer_relevancy} fmt={score} />
        </tbody>
      </table>
    </section>
  );
}

export function HowItWorks({ open, onClose, meta }: Props) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const dialog = ref.current;
    if (!dialog) return;
    if (open && !dialog.open) dialog.showModal();
    if (!open && dialog.open) dialog.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      className="surface m-auto w-[min(92vw,44rem)] rounded-lg p-0 text-[var(--text)] backdrop:bg-black/60"
    >
      <div className="max-h-[85vh] overflow-auto p-5 sm:p-6">
        <div className="flex items-start justify-between gap-4">
          <h2 className="text-lg font-semibold">How it works</h2>
          <button type="button" onClick={onClose} className="hairline rounded border px-2 py-1 font-mono text-xs" aria-label="Close">
            esc
          </button>
        </div>
        <div className="mt-4 space-y-5 text-sm leading-relaxed">
          <p>
            Each question runs through an explicit LangGraph loop. The model first decides what evidence it needs. Policy, product, and store facts come from a Qdrant index of chunked documents; live stock comes from typed, read-only tools served over MCP against the inventory database. Every retrieved passage and tool result becomes numbered evidence.
          </p>
          <Loop />
          <p>
            Before anything is returned, a second model checks the draft against that evidence and revises or removes unsupported claims. A deterministic guardrail then drops any sentence that does not carry a valid citation marker; if nothing survives, the assistant refuses rather than guess. Live tool calls are capped
            {meta ? ` at ${meta.tool_call_cap} per question` : ""}, and the models sit in failover chains across free-tier providers, so the answer model can change mid-conversation and the trace records which one answered.
          </p>
          {meta && <Results meta={meta} />}
          <p className="muted text-xs">
            All catalog, store, inventory, and order data is synthetic and generated by a seeded script; the policies describe a fictional operator. Nothing here refers to a real company.
          </p>
        </div>
      </div>
    </dialog>
  );
}
