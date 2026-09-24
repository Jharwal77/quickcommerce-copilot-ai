import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError, ask, getHealth, getMeta } from "./lib/api";
import type { AskResponse, HealthResponse, MetaResponse } from "./lib/api";
import { useTheme } from "./lib/theme";
import { Chat } from "./components/Chat";
import type { Turn } from "./components/Chat";
import { EvidencePanel } from "./components/EvidencePanel";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { HowItWorks } from "./components/HowItWorks";
import { KnowledgePanel } from "./components/KnowledgePanel";
import { TracePanel } from "./components/TracePanel";

type Tab = "trace" | "evidence" | "knowledge";

const TABS: { id: Tab; label: string }[] = [
  { id: "trace", label: "Agent trace" },
  { id: "evidence", label: "Evidence" },
  { id: "knowledge", label: "What I know" },
];

const README = "https://github.com/Jharwal77/quickcommerce-copilot-ai#readme";

export default function App() {
  const [theme, toggleTheme] = useTheme();
  const [meta, setMeta] = useState<MetaResponse | null>(null);
  const [health, setHealth] = useState<HealthResponse | null | "loading">("loading");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [loading, setLoading] = useState(false);
  const [coldStart, setColdStart] = useState(false);
  const [answered, setAnswered] = useState(false);
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null);
  const [activeMarker, setActiveMarker] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>("trace");
  const [howOpen, setHowOpen] = useState(false);
  const nextId = useRef(1);
  // The trace tab is forced open once, on the first answer that used a tool, so visitors
  // see the tool calls without hunting; after that the panel stays where they put it.
  const shownToolTrace = useRef(false);

  useEffect(() => {
    // Hitting /health first also warms a sleeping free-tier instance before the user types.
    getHealth().then(setHealth).catch(() => setHealth(null));
    getMeta().then(setMeta).catch(() => setMeta(null));
  }, []);

  const asked = useRef(false);
  useEffect(() => {
    // A ?q= link asks its question on load so a specific answer can be shared.
    const q = new URLSearchParams(window.location.search).get("q")?.trim();
    if (q && q.length >= 3 && !asked.current) {
      asked.current = true;
      void send(q);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const send = useCallback(
    async (question: string) => {
      const userId = nextId.current++;
      setTurns((t) => [...t, { id: userId, role: "user", text: question }]);
      setLoading(true);
      // The cold-start notice only appears when the first answer is slow to arrive.
      const timer = null;
      try {
        const response: AskResponse = await ask(question);
        const id = nextId.current++;
        setTurns((t) => [...t, { id, role: "agent", response }]);
        setSelectedTurn(id);
        setActiveMarker(null);
        if (response.mode === "system") {
          setTab("knowledge");
        } else if (response.tool_calls.length > 0 && !shownToolTrace.current) {
          shownToolTrace.current = true;
          setTab("trace");
        }
        setAnswered(true);
      } catch (error) {
        const apiError = error instanceof ApiError ? error : new ApiError("network", String(error), null);
        setTurns((t) => [...t, { id: nextId.current++, role: "error", error: apiError }]);
      } finally {
        if (timer !== null) window.clearTimeout(timer);
        setColdStart(false);
        setLoading(false);
      }
    },
    [answered],
  );

  const selected = useMemo(() => {
    const turn = turns.find((t) => t.id === selectedTurn);
    return turn && turn.role === "agent" ? turn.response : null;
  }, [turns, selectedTurn]);

  const cite = (turnId: number, marker: number) => {
    setSelectedTurn(turnId);
    setActiveMarker(marker);
    setTab("evidence");
  };

  return (
    <div className="flex min-h-full flex-col">
      <Header meta={meta} health={health} theme={theme} onToggleTheme={toggleTheme} onHowItWorks={() => setHowOpen(true)} />
      <main className="mx-auto grid w-full max-w-[1400px] flex-1 grid-cols-1 gap-4 px-4 py-4 sm:px-6 lg:grid-cols-12 lg:py-6">
        <div className="flex min-h-[45vh] flex-col lg:col-span-7 lg:h-[calc(100vh-11rem)]">
          <Chat
            turns={turns}
            loading={loading}
            coldStart={coldStart}
            selectedTurn={selectedTurn}
            activeMarker={activeMarker}
            onSend={send}
            onSelectTurn={(id) => setSelectedTurn(id)}
            onCite={cite}
            onShowKnowledge={() => setTab("knowledge")}
          />
        </div>
        <aside className="surface flex flex-col rounded-lg lg:col-span-5 lg:h-[calc(100vh-11rem)]" aria-label="Details">
          <div role="tablist" aria-label="Detail panels" className="hairline flex border-b">
            {TABS.map((t) => (
              <button
                key={t.id}
                role="tab"
                type="button"
                aria-selected={tab === t.id}
                onClick={() => setTab(t.id)}
                className={`flex-1 px-3 py-2.5 text-xs font-medium sm:text-sm ${
                  tab === t.id ? "border-b-2 border-signal-500 text-[var(--text)]" : "muted hover:text-[var(--text)]"
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>
          <div role="tabpanel" className="min-h-0 flex-1 overflow-y-auto p-4">
            {tab === "trace" && <TracePanel response={selected} onMarker={(m) => { setActiveMarker(m); setTab("evidence"); }} />}
            {tab === "evidence" && <EvidencePanel response={selected} activeMarker={activeMarker} onSelect={setActiveMarker} />}
            {tab === "knowledge" && <KnowledgePanel meta={meta} />}
          </div>
        </aside>
      </main>
      <Footer readme={meta?.links.readme ?? README} />
      <HowItWorks open={howOpen} onClose={() => setHowOpen(false)} meta={meta} />
    </div>
  );
}

