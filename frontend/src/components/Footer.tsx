export function Footer({ readme }: { readme: string }) {
  return (
    <footer className="hairline border-t">
      <div className="muted mx-auto flex max-w-[1400px] flex-wrap items-center justify-between gap-2 px-4 py-4 text-xs sm:px-6">
        <p>
          Synthetic data only: fictional operator, generated catalog and inventory. Not a real company.
        </p>
        <p className="font-mono">
          <a href={readme} target="_blank" rel="noreferrer" className="underline">README</a>
          {" · "}
          <a href="/docs" className="underline">OpenAPI</a>
          {" · "}
          <a href="/health" className="underline">health</a>
        </p>
      </div>
    </footer>
  );
}
