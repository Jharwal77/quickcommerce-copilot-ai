import type { ReactNode } from "react";
import type { MetaResponse } from "../lib/api";

interface Props {
  meta: MetaResponse | null;
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <h3 className="muted font-mono text-[0.65rem] font-semibold tracking-wider uppercase">{title}</h3>
      <div className="mt-1.5">{children}</div>
    </section>
  );
}

export function KnowledgePanel({ meta }: Props) {
  if (!meta) return <p className="muted text-sm">Loading what the assistant knows…</p>;
  const k = meta.knowledge;
  return (
    <div className="space-y-4 text-sm">
      <p className="muted">
        Derived from the data on disk, not typed in: {k.policies.length} policy documents, {k.dark_stores.length} dark stores, {k.product_count} products
        {k.chunk_count !== null ? ` indexed as ${k.chunk_count} chunks` : ""}. Stock and reorder levels come from the live tools ({meta.tools.join(", ") || "none loaded"}), never from the index.
      </p>
      <Group title="Policies">
        <ul className="space-y-2">
          {k.policies.map((p) => (
            <li key={p.doc_id}>
              <span className="font-medium">{p.title}</span>
              <p className="muted mt-0.5 text-xs">{p.sections.join(" · ")}</p>
            </li>
          ))}
        </ul>
      </Group>
      <Group title="Dark stores">
        <ul className="grid grid-cols-1 gap-1 font-mono text-xs sm:grid-cols-2">
          {k.dark_stores.map((s) => (
            <li key={s.store_id}>
              <span className="text-signal-500">{s.store_id}</span> {s.name}, {s.city}
            </li>
          ))}
        </ul>
      </Group>
      <Group title="Categories">
        <p className="text-xs">{k.categories.join(" · ")}</p>
      </Group>
      <Group title="Brands">
        <p className="text-xs">{k.brands.join(" · ")}</p>
      </Group>
      <p className="muted text-xs">
        Every brand, store, product, and order is synthetic, generated for this project; the policies describe a fictional operator. Anything outside this list is out of scope and the assistant is expected to refuse it.
      </p>
    </div>
  );
}
