import type { LogEntry } from "../types/contracts";

interface LogsPanelProps {
  logs: LogEntry[];
}

function formatTimestamp(timestamp: number) {
  const date = new Date(timestamp);
  return date.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
}

export function LogsPanel({ logs }: LogsPanelProps) {
  return (
    <section className="panel logs-panel">
      <div className="logs-panel__header">
        <div>
          <div className="panel__eyebrow">Logs</div>
          <p>Chronological runtime trace for requests and domain execution.</p>
        </div>
        <span className="logs-panel__count">{logs.length}</span>
      </div>

      <div className="logs-list">
        {logs.map((entry) => (
          <div key={entry.id} className={`log-row log-row--${entry.tone}`}>
            <span className="log-row__time">{formatTimestamp(entry.timestamp)}</span>
            <span className="log-row__stage">{entry.stage}</span>
            <span className="log-row__message">
              {entry.domain ? `${entry.domain}: ` : ""}
              {entry.message}
            </span>
            <span className="log-row__meta">
              {entry.latency_ms ? `${entry.latency_ms.toFixed(1)} ms` : ""}
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
