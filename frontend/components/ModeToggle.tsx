"use client";

import type { Mode } from "@/lib/api";

export default function ModeToggle({
  mode,
  onChange,
}: {
  mode: Mode;
  onChange: (mode: Mode) => void;
}) {
  const options: Mode[] = ["live", "replay"];
  return (
    <div className="inline-flex rounded-lg border border-zinc-300 dark:border-zinc-700 overflow-hidden text-sm">
      {options.map((m) => (
        <button
          key={m}
          onClick={() => onChange(m)}
          className={
            "px-3 py-1.5 transition-colors " +
            (mode === m
              ? "bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900"
              : "bg-transparent text-zinc-600 dark:text-zinc-400 hover:bg-zinc-100 dark:hover:bg-zinc-800")
          }
        >
          {m === "live" ? "Live" : "Replay"}
        </button>
      ))}
    </div>
  );
}
