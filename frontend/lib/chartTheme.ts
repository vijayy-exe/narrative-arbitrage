// Shared Recharts theme -- every chart in the app imports from here so grid
// lines, tooltips, and font sizing stay consistent without restyling per
// component. Colors match app/globals.css's semantic palette exactly (kept
// as a plain object, not read from CSS variables, since Recharts' SVG props
// need literal color strings, not var() references, for fill/stroke).

export const semanticColor = {
  gain: "#22c55e",
  loss: "#f43f5e",
  attentionSignal: "#38bdf8",
  attentionSignal2: "#818cf8",
  attentionSignal3: "#2dd4bf",
  agentAction: "#f59e0b",
  priceReward: "#71717a",
  neutral: "#52525b",
} as const;

export const chartGrid = {
  stroke: "#26262f",
  strokeDasharray: "3 3",
  opacity: 0.6,
};

export const chartAxisTick = {
  fontSize: 11,
  fill: "#8b8b9a",
  fontFamily: "var(--font-jetbrains-mono), ui-monospace, monospace",
};

export const chartTooltipStyle = {
  contentStyle: {
    fontSize: 12,
    fontFamily: "var(--font-jetbrains-mono), ui-monospace, monospace",
    background: "#121218",
    border: "1px solid #26262f",
    borderRadius: 8,
    color: "#e8e8ed",
  },
  labelStyle: { fontSize: 12, color: "#8b8b9a" },
  itemStyle: { fontSize: 12 },
};

export const chartLegendStyle = {
  fontSize: 11,
  color: "#8b8b9a",
};

/** Returns a color for gain/loss framing (positive -> gain color, negative
 * -> loss color, exactly zero -> neutral) -- the one place this convention
 * is defined, reused everywhere a number's SIGN needs a color. */
export function gainLossColor(value: number): string {
  if (value > 0) return semanticColor.gain;
  if (value < 0) return semanticColor.loss;
  return semanticColor.neutral;
}
