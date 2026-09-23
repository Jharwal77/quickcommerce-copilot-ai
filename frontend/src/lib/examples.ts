export interface Example {
  label: string;
  question: string;
  kind: "policy" | "live" | "catalog" | "out-of-scope";
}

// Two knowledge-base questions, two live-tool questions, one catalog question, and one
// deliberately out of scope so the refusal path is visible from the first click.
export const EXAMPLES: Example[] = [
  {
    label: "Returns policy",
    kind: "policy",
    question: "How long does a customer have to report a problem with a dairy item?",
  },
  {
    label: "Delivery fees",
    kind: "policy",
    question: "When is the delivery fee waived, and what does an order below that pay?",
  },
  {
    label: "Substitution rules",
    kind: "policy",
    question: "Can a vegetarian item be substituted with a non-vegetarian one?",
  },
  {
    label: "Live stock",
    kind: "live",
    question:
      "How many units of Nandhini Fresh Paneer 200 g are available at the Indiranagar Dark Store right now?",
  },
  {
    label: "Reorder check",
    kind: "live",
    question:
      "Is Tinytots Baby Diapers Medium 32 pieces at or below its reorder point at the Koramangala Dark Store?",
  },
  {
    label: "Out of scope",
    kind: "out-of-scope",
    question: "Is Nandhini Fresh Paneer 200 g in stock at the Whitefield Dark Store?",
  },
];
