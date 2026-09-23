import type { ApiError } from "../lib/api";

export function ColdStartNotice() {
  return (
    <div className="surface rounded-md px-3 py-2 text-sm" role="status">
      <span className="font-medium">Waking the instance.</span>{" "}
      <span className="muted">
        This runs on a free tier that sleeps when idle; the first request after a pause can take about a minute. Later requests take a few seconds.
      </span>
    </div>
  );
}

export function RefusalNotice() {
  return (
    <div className="hairline mt-2 rounded-md border border-dashed px-3 py-2 text-sm" role="note">
      <span className="font-medium text-signal-500">Refused by design.</span>{" "}
      <span className="muted">
        Nothing retrieved or returned by a tool supported an answer, so the assistant said so instead of guessing. The trace shows what it checked before refusing.
      </span>
    </div>
  );
}

export function ErrorNotice({ error }: { error: ApiError }) {
  if (error.kind === "budget") {
    return (
      <div className="surface rounded-md px-3 py-2 text-sm" role="alert">
        <span className="font-medium">Provider budgets are used up for today.</span>{" "}
        <span className="muted">
          Generation runs on free tiers of several model providers with daily caps. The chain failed over through every one of them and none has budget left right now. Retrieval and the tool server are fine; try again later in the day.
        </span>
      </div>
    );
  }
  if (error.kind === "network") {
    return (
      <div className="surface rounded-md px-3 py-2 text-sm" role="alert">
        <span className="font-medium">Could not reach the API.</span>{" "}
        <span className="muted">Check your connection, or wait a moment if the instance is still waking.</span>
      </div>
    );
  }
  if (error.kind === "invalid") {
    return (
      <div className="surface rounded-md px-3 py-2 text-sm" role="alert">
        <span className="font-medium">That question was rejected.</span>{" "}
        <span className="muted">Questions need between 3 and 1000 characters.</span>
      </div>
    );
  }
  return (
    <div className="surface rounded-md px-3 py-2 text-sm" role="alert">
      <span className="font-medium">The service returned an error.</span>{" "}
      <span className="muted font-mono text-xs">{error.message}</span>
    </div>
  );
}

export function Skeleton() {
  return (
    <div className="space-y-2" aria-hidden="true">
      <div className="surface-2 h-3 w-5/6 rounded motion-safe:animate-pulse" />
      <div className="surface-2 h-3 w-2/3 rounded motion-safe:animate-pulse" />
      <div className="surface-2 h-3 w-1/2 rounded motion-safe:animate-pulse" />
    </div>
  );
}
