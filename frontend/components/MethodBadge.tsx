import type { Method } from "@/lib/api";

// Never blur "ppo" and "momentum_baseline" -- see src/realtime/infer.py's
// module docstring / this project's overall honesty policy. This badge is
// the one place in the UI a viewer sees which is which, so it always
// renders literally which method produced the decision, not a paraphrase.
export default function MethodBadge({ method }: { method: Method }) {
  const isPpo = method === "ppo";
  return (
    <span
      title={
        isPpo
          ? "Produced by the trained PPO policy checkpoint (models/ppo_DOGE/model.zip)."
          : "No trained RL checkpoint exists for this asset. Produced by the momentum baseline (src/baselines/momentum.py) instead -- this is NOT an RL decision."
      }
      className={
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium whitespace-nowrap " +
        (isPpo
          ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
          : "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300")
      }
    >
      {isPpo ? "PPO (trained RL)" : "Momentum baseline (not RL)"}
    </span>
  );
}
