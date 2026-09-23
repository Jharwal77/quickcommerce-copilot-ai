import { useEffect, useRef } from "react";
import type { AskResponse, EvidenceItem } from "../lib/api";

interface Props {
  response: AskResponse | null;
  activeMarker: number | null;
  onSelect: (marker: number) => void;
}

function Card({ item, active, cited, onSelect }: { item: EvidenceItem; active: boolean; cited: boolean; onSelect: () => void }) {
  const ref = useRef<HTMLLIElement>(null);
  useEffect(() => {
    if (active) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [active]);
  return (
    <li
      ref={ref}
      className={`surface rounded-md p-3 ${active ? "ring-2 ring-signal-500" : ""}`}
      aria-current={active ? "true" : undefined}
    >
      <div className="flex items-start gap-2">
        <button type="button" className="cite" data-active={active} onClick={onSelect} aria-label={`Evidence ${item.marker}`}>
          {item.marker}
        </button>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-sm font-medium break-words">{item.label}</span>
            <span className={`font-mono text-[0.65rem] tracking-wider ${item.kind === "tool" ? "text-tool-500" : "text-signal-500"}`}>
              {item.kind === "tool" ? "LIVE TOOL" : "PASSAGE"}
            </span>
            {!cited && <span className="muted font-mono text-[0.65rem]">retrieved, not cited</span>}
          </div>
          <p className="muted mt-1 font-mono text-[0.68rem] break-all">{item.doc_id}{item.kind === "passage" ? ` · score ${item.score.toFixed(2)}` : ""}</p>
        </div>
      </div>
      <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap font-mono text-xs leading-relaxed">{item.text}</pre>
    </li>
  );
}

export function EvidencePanel({ response, activeMarker, onSelect }: Props) {
  if (!response) {
    return (
      <p className="muted text-sm">
        Every passage and tool result the agent gathered for the current answer is listed here with its marker. Click a [n] in the answer to jump to it.
      </p>
    );
  }
  if (response.mode === "system") {
    return <p className="muted text-sm">System response: nothing was retrieved or called, so there is no evidence to show. The “What I know” tab lists what the assistant covers.</p>;
  }
  if (response.evidence.length === 0) {
    return <p className="muted text-sm">No evidence was gathered for this answer, which is why it was refused.</p>;
  }
  const cited = new Set(response.citations.map((c) => c.marker));
  return (
    <ul className="space-y-2">
      {response.evidence.map((item) => (
        <Card
          key={item.marker}
          item={item}
          active={activeMarker === item.marker}
          cited={cited.has(item.marker)}
          onSelect={() => onSelect(item.marker)}
        />
      ))}
    </ul>
  );
}
