import { useDeferredValue, useMemo } from "react";
import { motion } from "framer-motion";
import { SAMPLE_LIBRARY } from "../features/analyzer/examples";
import type { PipelinePhase } from "../types/contracts";

interface TextWorkbenchProps {
  value: string;
  phase: PipelinePhase;
  onChange: (value: string) => void;
  onAnalyze: () => void;
  onReset: () => void;
  disabled: boolean;
}

const PHASE_LABELS: Record<PipelinePhase, string> = {
  idle: "Ready",
  routing: "Preparing",
  running: "Running",
  revealing: "Finishing",
  done: "Complete",
  error: "Error",
};

function pickRandomSample(currentValue: string) {
  const candidates = SAMPLE_LIBRARY.filter((sample) => sample.value !== currentValue);
  const pool = candidates.length > 0 ? candidates : SAMPLE_LIBRARY;
  return pool[Math.floor(Math.random() * pool.length)];
}

export function TextWorkbench({
  value,
  phase,
  onChange,
  onAnalyze,
  onReset,
  disabled,
}: TextWorkbenchProps) {
  const deferredValue = useDeferredValue(value);
  const stats = useMemo(() => {
    const trimmed = deferredValue.trim();
    return {
      chars: trimmed.length,
      words: trimmed ? trimmed.split(/\s+/).length : 0,
    };
  }, [deferredValue]);

  const handleGenerateSample = () => {
    const nextSample = pickRandomSample(value);
    if (nextSample) {
      onChange(nextSample.value);
    }
  };

  return (
    <section className="panel workbench">
      <div className="workbench__header">
        <div className="workbench__title">
          <div className="panel__eyebrow">Input</div>
          <p>Paste text or generate a sample.</p>
        </div>

        <motion.div
          className={`phase-pill phase-pill--${phase}`}
          animate={{
            opacity: phase === "idle" ? 0.8 : 1,
            scale: phase === "running" ? [1, 1.03, 1] : 1,
          }}
          transition={{ duration: 1.2, repeat: phase === "running" ? Infinity : 0 }}
        >
          {PHASE_LABELS[phase]}
        </motion.div>
      </div>

      <div className="textarea-shell">
        <textarea
          className="input-box"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Paste a review, social post, academic paragraph, or moderation edge case."
        />
      </div>

      <div className="workbench__footer">
        <div className="action-row">
          <button
            className="primary-button"
            type="button"
            disabled={disabled || !value.trim()}
            onClick={onAnalyze}
          >
            Analyze
          </button>
          <button className="secondary-button" type="button" onClick={onReset}>
            Clear
          </button>
        </div>

        <div className="meta-row">
          <span>{stats.chars} chars</span>
          <span>{stats.words} words</span>
          <button className="generator-button" type="button" onClick={handleGenerateSample}>
            Generate sample for me
          </button>
        </div>
      </div>
    </section>
  );
}
