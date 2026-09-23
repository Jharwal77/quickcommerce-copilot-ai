import type { HealthResponse, MetaResponse } from "../lib/api";
import type { Theme } from "../lib/theme";

interface Props {
  meta: MetaResponse | null;
  health: HealthResponse | null | "loading";
  theme: Theme;
  onToggleTheme: () => void;
  onHowItWorks: () => void;
}

function StatusPill({ health }: { health: Props["health"] }) {
  if (health === "loading") {
    return <span className="muted font-mono text-xs">connecting…</span>;
  }
  if (!health) {
    return <span className="font-mono text-xs text-warn-500">api unreachable</span>;
  }
  const ok = health.status === "ok";
  return (
    <span className="inline-flex items-center gap-1.5 font-mono text-xs">
      <span
        aria-hidden="true"
        className={`inline-block h-1.5 w-1.5 rounded-full ${ok ? "bg-ok-500" : "bg-signal-500"}`}
      />
      <span className={ok ? "text-ok-500" : "text-signal-500"}>{ok ? "live" : "degraded"}</span>
      {health.vectors !== null && (
        <span className="muted hidden sm:inline">· {health.vectors} chunks indexed</span>
      )}
    </span>
  );
}

export function Header({ meta, health, theme, onToggleTheme, onHowItWorks }: Props) {
  const source = meta?.links.source ?? "https://github.com/Jharwal77/quickcommerce-copilot";
  return (
    <header className="hairline sticky top-0 z-20 border-b backdrop-blur" style={{ background: "color-mix(in srgb, var(--bg) 88%, transparent)" }}>
      <div className="mx-auto flex max-w-[1400px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3 sm:px-6">
        <div className="min-w-0 flex-1 basis-56">
          <div className="flex items-baseline gap-3">
            <h1 className="font-mono text-base font-semibold tracking-tight sm:text-lg">
              quickcommerce-copilot
            </h1>
            <StatusPill health={health} />
          </div>
          <p className="muted mt-0.5 hidden truncate text-sm sm:block">
            Grounded answers for dark-store staff: policies from the knowledge base, stock from live tools, refusals when neither has it.
          </p>
        </div>
        <nav className="flex flex-wrap items-center justify-end gap-1 sm:gap-2" aria-label="Site">
          <button
            type="button"
            onClick={onHowItWorks}
            className="hairline rounded border px-2.5 py-1.5 text-xs font-medium hover:bg-[var(--surface-2)] sm:text-sm"
          >
            How it works
          </button>
          <a
            href="/docs"
            className="hairline rounded border px-2.5 py-1.5 text-xs font-medium hover:bg-[var(--surface-2)] sm:text-sm"
          >
            API
          </a>
          <a
            href={source}
            target="_blank"
            rel="noreferrer"
            className="hairline rounded border px-2.5 py-1.5 text-xs font-medium hover:bg-[var(--surface-2)] sm:text-sm"
          >
            Source
          </a>
          <button
            type="button"
            onClick={onToggleTheme}
            aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
            className="hairline rounded border px-2.5 py-1.5 font-mono text-xs hover:bg-[var(--surface-2)]"
          >
            {theme === "dark" ? "light" : "dark"}
          </button>
        </nav>
      </div>
    </header>
  );
}
