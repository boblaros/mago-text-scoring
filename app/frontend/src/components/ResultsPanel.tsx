import type { DomainCatalogEntry, LaneState } from "../types/contracts";
import { ResultCard } from "./ResultCard";

interface ResultsPanelProps {
  domains: DomainCatalogEntry[];
  lanes: Record<string, LaneState>;
  onNavigateToModels: () => void;
}

const PANEL_COPY = "Live domain results with confidence and status.";

export function ResultsPanel({
  domains,
  lanes,
  onNavigateToModels,
}: ResultsPanelProps) {
  return (
    <section className="panel results-panel">
      <div className="results-panel__header">
        <div>
          <div className="panel__eyebrow">Output</div>
          <p>{PANEL_COPY}</p>
        </div>
      </div>

      {domains.length ? (
        <div className="results-grid">
          {domains.map((domain) => (
            <ResultCard
              key={domain.domain}
              domain={domain}
              lane={lanes[domain.domain] ?? { domain: domain.domain, status: "idle" }}
            />
          ))}
        </div>
      ) : (
        <div className="results-empty-state">
          <strong>No active models in Home.</strong>
          <p>Enable at least one model in Models to restore the live scoring lanes.</p>
        </div>
      )}

      <button
        className="domain-slot-placeholder"
        type="button"
        onClick={onNavigateToModels}
      >
        Manage models
      </button>
    </section>
  );
}
