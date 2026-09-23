import { useEffect, useRef, useState } from "react";
import type { KeyboardEvent } from "react";
import type { AskResponse, ApiError } from "../lib/api";
import { EXAMPLES } from "../lib/examples";
import { ms } from "../lib/format";
import { AnswerMarkdown } from "./AnswerMarkdown";
import { ColdStartNotice, ErrorNotice, RefusalNotice, Skeleton } from "./Notices";

export type Turn =
  | { id: number; role: "user"; text: string }
  | { id: number; role: "agent"; response: AskResponse }
  | { id: number; role: "error"; error: ApiError };

interface Props {
  turns: Turn[];
  loading: boolean;
  coldStart: boolean;
  selectedTurn: number | null;
  activeMarker: number | null;
  onSend: (question: string) => void;
  onSelectTurn: (id: number) => void;
  onCite: (turnId: number, marker: number) => void;
  onShowKnowledge: () => void;
}

function Chips({ disabled, onPick }: { disabled: boolean; onPick: (q: string) => void }) {
  return (
    <div className="flex flex-wrap gap-2" aria-label="Example questions">
      {EXAMPLES.map((e) => (
        <button
          key={e.label}
          type="button"
          disabled={disabled}
          onClick={() => onPick(e.question)}
          title={e.question}
          className={`hairline rounded-full border px-3 py-1 text-xs hover:bg-[var(--surface-2)] disabled:opacity-50 ${
            e.kind === "out-of-scope" ? "border-dashed" : ""
          }`}
        >
          <span
            className={`mr-1.5 font-mono text-[0.6rem] tracking-wider uppercase ${
              e.kind === "live" ? "text-tool-500" : e.kind === "out-of-scope" ? "text-warn-500" : "text-signal-500"
            }`}
          >
            {e.kind === "policy" ? "kb" : e.kind}
          </span>
          {e.label}
        </button>
      ))}
    </div>
  );
}

function SystemTurn({ turn, selected, onSelect, onShowKnowledge }: { turn: Extract<Turn, { role: "agent" }>; selected: boolean; onSelect: () => void; onShowKnowledge: () => void }) {
  const r = turn.response;
  return (
    <article
      className={`surface rounded-lg border-l-2 border-tool-500 p-4 ${selected ? "ring-1 ring-signal-500/60" : ""}`}
      onClick={onSelect}
      onFocus={onSelect}
      tabIndex={0}
      aria-label="System response"
    >
      <div className="muted mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[0.68rem]">
        <span className="text-tool-500">system</span>
        <span>capability summary · not a grounded answer · nothing retrieved or called</span>
      </div>
      <AnswerMarkdown text={r.answer} validMarkers={new Set<number>()} activeMarker={null} onCite={() => undefined} />
      <div className="mt-3">
        <button
          type="button"
          onClick={(event) => {
            event.stopPropagation();
            onShowKnowledge();
          }}
          className="hairline rounded-full border px-3 py-1 text-xs hover:bg-[var(--surface-2)]"
        >
          Open “What I know”
        </button>
      </div>
    </article>
  );
}

function AgentTurn({ turn, selected, activeMarker, onSelect, onCite }: { turn: Extract<Turn, { role: "agent" }>; selected: boolean; activeMarker: number | null; onSelect: () => void; onCite: (m: number) => void }) {
  const r = turn.response;
  const toolCount = r.tool_calls.length;
  const passages = r.evidence.filter((e) => e.kind === "passage").length;
  return (
    <article
      className={`surface rounded-lg p-4 ${selected ? "ring-1 ring-signal-500/60" : ""}`}
      onClick={onSelect}
      onFocus={onSelect}
      tabIndex={0}
      aria-label="Assistant answer"
    >
      <div className="muted mb-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[0.68rem]">
        <span className="text-ok-500">assistant</span>
        <span>{passages} passage{passages === 1 ? "" : "s"}</span>
        <span className={toolCount ? "text-tool-500" : ""}>{toolCount} tool call{toolCount === 1 ? "" : "s"}</span>
        <span>{r.citations.length} citation{r.citations.length === 1 ? "" : "s"}</span>
        <span>{ms(r.usage.total_ms)}</span>
        {selected && <span className="text-signal-500">shown in trace &amp; evidence</span>}
      </div>
      <AnswerMarkdown
        text={r.answer}
        validMarkers={new Set(r.evidence.map((e) => e.marker))}
        activeMarker={selected ? activeMarker : null}
        onCite={onCite}
      />
      {r.refused && <RefusalNotice />}
    </article>
  );
}

export function Chat({ turns, loading, coldStart, selectedTurn, activeMarker, onSend, onSelectTurn, onCite, onShowKnowledge }: Props) {
  const [draft, setDraft] = useState("");
  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [turns.length, loading]);

  const submit = () => {
    const q = draft.trim();
    if (q.length < 3 || loading) return;
    onSend(q);
    setDraft("");
    inputRef.current?.focus();
  };

  const onKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <section className="flex min-h-0 flex-1 flex-col" aria-label="Conversation">
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="surface rounded-lg p-4">
            <p className="text-sm">
              Ask about a policy, a product, a dark store, or live stock. Answers cite their evidence; questions the evidence cannot support are refused on purpose. Not sure where to start? Ask “what can you do”.
            </p>
            <div className="mt-3">
              <Chips disabled={loading} onPick={onSend} />
            </div>
          </div>
        )}
        {turns.map((turn) => {
          if (turn.role === "user") {
            return (
              <div key={turn.id} className="flex justify-end">
                <p className="surface-2 max-w-[85%] rounded-lg px-3.5 py-2 text-sm">{turn.text}</p>
              </div>
            );
          }
          if (turn.role === "error") {
            return <ErrorNotice key={turn.id} error={turn.error} />;
          }
          if (turn.response.mode === "system") {
            return (
              <SystemTurn
                key={turn.id}
                turn={turn}
                selected={selectedTurn === turn.id}
                onSelect={() => onSelectTurn(turn.id)}
                onShowKnowledge={onShowKnowledge}
              />
            );
          }
          return (
            <AgentTurn
              key={turn.id}
              turn={turn}
              selected={selectedTurn === turn.id}
              activeMarker={activeMarker}
              onSelect={() => onSelectTurn(turn.id)}
              onCite={(m) => onCite(turn.id, m)}
            />
          );
        })}
        {loading && (
          <div className="surface rounded-lg p-4" aria-live="polite" aria-busy="true">
            <p className="muted mb-2 font-mono text-[0.68rem]">assistant · deciding, retrieving, checking…</p>
            <Skeleton />
            {coldStart && <div className="mt-3"><ColdStartNotice /></div>}
          </div>
        )}
        <div ref={endRef} />
      </div>
      <form
        className="mt-3 space-y-2"
        onSubmit={(e) => {
          e.preventDefault();
          submit();
        }}
      >
        {turns.length > 0 && <Chips disabled={loading} onPick={onSend} />}
        <div className="surface flex items-end gap-2 rounded-lg p-2">
          <label htmlFor="question" className="sr-only">Question</label>
          <textarea
            id="question"
            ref={inputRef}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKey}
            rows={2}
            maxLength={1000}
            placeholder="Ask about returns, delivery, substitutions, a product, or current stock at a store…"
            className="min-h-[2.5rem] flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-[var(--muted)]"
          />
          <button
            type="submit"
            disabled={loading || draft.trim().length < 3}
            className="rounded bg-signal-500 px-3 py-2 text-sm font-semibold text-ink-950 hover:bg-signal-600 disabled:opacity-50"
          >
            Ask
          </button>
        </div>
        <p className="muted font-mono text-[0.65rem]">Enter to send · Shift+Enter for a new line · history is kept in memory for this tab only</p>
      </form>
    </section>
  );
}
