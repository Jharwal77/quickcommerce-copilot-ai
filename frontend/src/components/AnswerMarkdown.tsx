import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

interface Props {
  text: string;
  validMarkers: Set<number>;
  activeMarker: number | null;
  onCite: (marker: number) => void;
}

// Citation markers are rewritten to fragment links before parsing so the markdown
// renderer hands them to us as anchors, which we turn into focusable buttons.
function linkMarkers(text: string): string {
  return text.replace(/\[(\d{1,2})\]/g, (_, n: string) => `[${n}](#cite-${n})`);
}

export function AnswerMarkdown({ text, validMarkers, activeMarker, onCite }: Props) {
  return (
    <div className="prose-answer text-[0.95rem] leading-relaxed">
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const match = href?.match(/^#cite-(\d{1,2})$/);
            if (match) {
              const marker = Number(match[1]);
              if (!validMarkers.has(marker)) return <span>[{marker}]</span>;
              return (
                <button
                  type="button"
                  className="cite"
                  data-active={activeMarker === marker}
                  aria-label={`Show evidence ${marker}`}
                  onClick={() => onCite(marker)}
                >
                  {marker}
                </button>
              );
            }
            return (
              <a href={href} target="_blank" rel="noreferrer" className="underline">
                {children}
              </a>
            );
          },
        }}
      >
        {linkMarkers(text)}
      </Markdown>
    </div>
  );
}
