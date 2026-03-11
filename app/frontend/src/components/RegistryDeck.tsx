import type { AnalysisResponse, DomainCatalogEntry, PipelinePhase } from "../types/contracts";
import { getCompactDomainLabel } from "../features/analyzer/domainLabels";

interface RegistryDeckProps {
  domains: DomainCatalogEntry[];
  phase: PipelinePhase;
  analysis: AnalysisResponse | null;
  error?: string | null;
}

const PHASE_LABELS: Record<PipelinePhase, string> = {
  idle: "Ready",
  routing: "Preparing",
  running: "Running",
  revealing: "Finishing",
  done: "Complete",
  error: "Error",
};

function averageLatency(analysis: AnalysisResponse | null) {
  if (!analysis || analysis.results.length === 0) {
    return null;
  }

  const total = analysis.results.reduce((sum, result) => sum + result.latency_ms, 0);
  return total / analysis.results.length;
}

export function RegistryDeck({
  domains,
  phase,
  analysis,
  error,
}: RegistryDeckProps) {
  const meanLatency = averageLatency(analysis);
  const activeCount = domains.length;
  const systemStats = [
    { label: "Domains", value: String(activeCount) },
    { label: "Active", value: `${activeCount}/${activeCount}` },
    { label: "Status", value: PHASE_LABELS[phase] },
    { label: "Runtime", value: import.meta.env.DEV ? "Local" : "App" },
    {
      label: "Mean confidence",
      value: analysis?.aggregate.mean_confidence
        ? `${Math.round(analysis.aggregate.mean_confidence * 100)}%`
        : "—",
    },
    { label: "Avg latency", value: meanLatency ? `${meanLatency.toFixed(1)} ms` : "—" },
  ];

  return (
    <aside className="panel sidebar-panel">
      <div className="panel__eyebrow">Setup</div>

      <div className="sidebar-block">
        <div className="sidebar-section__title">System</div>
        <div className="stat-grid stat-grid--system">
          {systemStats.map((item) => (
            <div key={item.label} className="stat-card">
              <span>{item.label}</span>
              <strong>{item.value}</strong>
            </div>
          ))}
        </div>
      </div>

      {error ? <div className="inline-alert">{error}</div> : null}

      <div className="sidebar-block">
        <div className="sidebar-section__title">Models</div>
        {domains.length ? (
          <div className="model-capsules">
            {domains.map((domain) => (
              <div key={domain.domain} className="model-capsule">
                <div className="model-capsule__content">
                  <span>{getCompactDomainLabel(domain)}</span>
                  <strong>{domain.active_model_name ?? "No active model"}</strong>
                </div>
                <span className="model-capsule__state">Active</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="inline-alert">No active models. Activate a model in Models.</div>
        )}
      </div>
    </aside>
  );
}
