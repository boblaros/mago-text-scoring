import type { CSSProperties } from "react";
import { motion } from "framer-motion";
import { getDomainTheme } from "../features/analyzer/domainTheme";
import { getCompactDomainLabel } from "../features/analyzer/domainLabels";
import type { DomainCatalogEntry, LaneState } from "../types/contracts";

interface ResultCardProps {
  domain: DomainCatalogEntry;
  lane: LaneState;
}

const STATUS_LABELS = {
  idle: "Pending",
  queued: "Queued",
  running: "Running",
  resolved: "Resolved",
  error: "Failed",
} as const;

const STATUS_COPY = {
  idle: "Waiting to start",
  queued: "Queued for analysis",
  running: "Evaluating text",
  resolved: "Result ready",
  error: "Run failed",
} as const;

export function ResultCard({ domain, lane }: ResultCardProps) {
  const theme = getDomainTheme(domain);
  const result = lane.result;
  const confidence = result ? Math.round(result.confidence * 100) : 0;
  const width =
    lane.status === "resolved"
      ? "100%"
      : lane.status === "running"
        ? "44%"
        : lane.status === "queued"
          ? "26%"
          : lane.status === "error"
            ? "100%"
            : "10%";

  return (
    <motion.article
      className={`result-card result-card--${lane.status}`}
      style={
        {
          "--result-accent": theme.accent,
          "--result-glow": theme.glow,
          "--result-panel": theme.panel,
        } as CSSProperties
      }
      animate={{
        boxShadow:
          lane.status === "running"
            ? `0 0 36px ${theme.glow}`
            : lane.status === "resolved"
              ? `0 0 22px ${theme.glow}`
              : "0 18px 40px rgba(0, 0, 0, 0.22)",
      }}
      transition={{ duration: 0.3 }}
    >
      <div className="result-card__header">
        <span className="result-card__domain">{getCompactDomainLabel(domain)}</span>
        <span className={`result-card__status result-card__status--${lane.status}`}>
          {STATUS_LABELS[lane.status]}
        </span>
      </div>

      <div className="result-card__value">
        {result?.predicted_label ?? STATUS_COPY[lane.status]}
      </div>

      <div className="result-card__model-row">
        <span className="result-card__model">
          {result?.model_name ?? domain.active_model_name}
        </span>
      </div>

      <div className="result-card__bar">
        <motion.div
          className="result-card__bar-fill"
          animate={{
            width,
            x: lane.status === "running" ? ["-16%", "118%"] : "0%",
          }}
          transition={{
            duration: lane.status === "running" ? 1.1 : 0.3,
            repeat: lane.status === "running" ? Infinity : 0,
            ease: "linear",
          }}
        />
      </div>

      <div className="result-card__meta">
        <span className="result-card__latency">
          {result ? `Latency: ${result.latency_ms.toFixed(1)} ms` : "Latency: —"}
        </span>
        <span className="result-card__confidence">
          {result ? `Confidence: ${confidence}%` : "Confidence: —"}
        </span>
      </div>
    </motion.article>
  );
}
