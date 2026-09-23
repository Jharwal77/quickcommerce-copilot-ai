export function ms(value: number): string {
  return value >= 1000 ? `${(value / 1000).toFixed(1)} s` : `${Math.round(value)} ms`;
}

export function percent(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : `${Math.round(value * 100)}%`;
}

export function score(value: number | null | undefined): string {
  return value === null || value === undefined ? "n/a" : value.toFixed(2);
}

export function shortModel(ref: string): string {
  const [provider, model] = ref.replace(/^reflect=/, "").split(":");
  const name = (model ?? "").split("/").pop() ?? model ?? "";
  return provider ? `${provider} / ${name}` : name;
}

export function argumentsText(args: Record<string, string | number | boolean | null>): string {
  return Object.entries(args)
    .map(([key, value]) => `${key}=${String(value)}`)
    .join(", ");
}
