import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { PlotlyFigure } from "../components/models/PlotlyFigure";
import { ModelUploadWizard } from "../components/models/ModelUploadWizard";
import { getCompactDomainLabel } from "../features/analyzer/domainLabels";
import { getDomainTheme } from "../features/analyzer/domainTheme";
import { useCatalog } from "../hooks/useCatalog";
import {
  deleteModel as deleteModelRequest,
  fetchModelDashboard,
  reorderModels,
  updateModel,
} from "../services/api";
import type {
  CatalogSnapshotResponse,
  DashboardFigure,
  DashboardSectionSummary,
  DomainCatalogModel,
  ModelDashboardResponse,
} from "../types/contracts";

function asRecord(value: unknown) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asArray(value: unknown) {
  return Array.isArray(value) ? value : [];
}

function formatNumber(value: unknown) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  if (value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString();
  }
  return value.toFixed(3);
}

function formatTimestamp(value?: string | null) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return date.toLocaleString([], {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function humanizeStatus(status: string) {
  return status.replace(/_/g, " ");
}

function familyLabel(model: DomainCatalogModel) {
  if (model.framework_type === "transformers") {
    return "Transformer";
  }
  if (model.framework_type === "pytorch") {
    return model.architecture ? `PyTorch · ${model.architecture}` : "PyTorch";
  }
  if (model.framework_type === "sklearn") {
    return "Classical ML";
  }
  return model.framework_type;
}

function stringifyStructuredValue(value: unknown) {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

function renderKeyValueValue(value: unknown) {
  if (typeof value === "number") {
    return <strong className="dashboard-kv__value">{formatNumber(value)}</strong>;
  }

  if (typeof value === "boolean") {
    return <strong className="dashboard-kv__value">{value ? "true" : "false"}</strong>;
  }

  if (Array.isArray(value) || (value !== null && typeof value === "object")) {
    return <pre className="dashboard-kv__code">{stringifyStructuredValue(value)}</pre>;
  }

  return <strong className="dashboard-kv__value">{String(value)}</strong>;
}

function findDocument(dashboard: ModelDashboardResponse | null, path: string) {
  if (!dashboard) {
    return null;
  }
  return dashboard.documents[path] ?? null;
}

function figuresForSection(
  dashboard: ModelDashboardResponse | null,
  sectionId: string,
) {
  return dashboard?.figures.filter((figure) => figure.section_id === sectionId) ?? [];
}

function imagesForSection(
  dashboard: ModelDashboardResponse | null,
  sectionId: string,
) {
  return dashboard?.images.filter((image) => image.section_id === sectionId) ?? [];
}

function KeyValueBlock({
  title,
  data,
}: {
  title: string;
  data: Record<string, unknown>;
}) {
  const entries = Object.entries(data).filter(([, value]) => value !== null && value !== undefined);
  if (!entries.length) {
    return null;
  }

  return (
    <article className="dashboard-block">
      <div className="dashboard-block__header">
        <div className="dashboard-block__eyebrow">Metadata</div>
        <h4>{title}</h4>
      </div>
      <div className="dashboard-kv-grid">
        {entries.map(([key, value]) => (
          <div key={key} className="dashboard-kv">
            <span>{key.replace(/_/g, " ")}</span>
            {renderKeyValueValue(value)}
          </div>
        ))}
      </div>
    </article>
  );
}

function TableBlock({
  title,
  eyebrow,
  rows,
}: {
  title: string;
  eyebrow: string;
  rows: Record<string, unknown>[];
}) {
  if (!rows.length) {
    return null;
  }

  const columns = Array.from(
    rows.reduce((keys, row) => {
      Object.keys(row).forEach((key) => {
        if (key !== "source_file" && key !== "artifact_paths") {
          keys.add(key);
        }
      });
      return keys;
    }, new Set<string>()),
  ).slice(0, 7);

  return (
    <article className="dashboard-block">
      <div className="dashboard-block__header">
        <div className="dashboard-block__eyebrow">{eyebrow}</div>
        <h4>{title}</h4>
      </div>
      <div className="dashboard-table-shell">
        <table className="dashboard-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column.replace(/_/g, " ")}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={`${title}-${rowIndex}`}>
                {columns.map((column) => {
                  const value = row[column];
                  return (
                    <td key={column}>
                      {typeof value === "number" ? formatNumber(value) : value ? String(value) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </article>
  );
}

function renderDashboardSection(
  section: DashboardSectionSummary,
  dashboard: ModelDashboardResponse | null,
) {
  const overview = asRecord(dashboard?.overview);
  const sectionFigures = figuresForSection(dashboard, section.id);
  const sectionImages = imagesForSection(dashboard, section.id);
  const primaryEvaluation = asRecord(findDocument(dashboard, "metrics/primary-evaluation.json"));
  const benchmarkRows = asArray(findDocument(dashboard, "metrics/benchmark-test.json")).filter(
    (item): item is Record<string, unknown> => Boolean(asRecord(item)),
  );
  const crossDatasetRows = asArray(findDocument(dashboard, "metrics/cross-dataset.json")).filter(
    (item): item is Record<string, unknown> => Boolean(asRecord(item)),
  );
  const trainingHistory = asRecord(findDocument(dashboard, "curves/training-history.json"));
  const learningCurves = asArray(findDocument(dashboard, "curves/learning-curve.json")).filter(
    (item): item is Record<string, unknown> => Boolean(asRecord(item)),
  );
  const classDistribution = asRecord(findDocument(dashboard, "distributions/class-distribution.json"));
  const sourceDistribution = asRecord(
    findDocument(dashboard, "distributions/source-dataset-distribution.json"),
  );
  const predictionSamples = asArray(findDocument(dashboard, "samples/prediction-samples.json")).filter(
    (item): item is Record<string, unknown> => Boolean(asRecord(item)),
  );
  const metadataModel = asRecord(findDocument(dashboard, "metadata/model.json"));
  const experimentConfig = asRecord(findDocument(dashboard, "metadata/experiment-config.json"));

  if (section.status === "missing" || section.status === "not_applicable") {
    return (
      <article key={section.id} className="dashboard-section">
        <div className="dashboard-section__header">
          <div>
            <div className="dashboard-section__eyebrow">{section.title}</div>
            <h3>{section.title}</h3>
            <p>{section.description}</p>
          </div>
          <span className={`status-chip status-chip--${section.status}`}>{humanizeStatus(section.status)}</span>
        </div>
        <div className="dashboard-empty">
          <strong>{section.reason ?? "No data was attached for this section."}</strong>
        </div>
      </article>
    );
  }

  return (
    <article key={section.id} className="dashboard-section">
      <div className="dashboard-section__header">
        <div>
          <div className="dashboard-section__eyebrow">{section.title}</div>
          <h3>{section.title}</h3>
          <p>{section.description}</p>
        </div>
        <span className={`status-chip status-chip--${section.status}`}>{humanizeStatus(section.status)}</span>
      </div>

      {section.id === "summary" && overview ? (
        <>
          <div className="dashboard-stat-grid">
            {Object.entries(asRecord(overview.evaluation_highlights) ?? {}).flatMap(([split, metrics]) => {
              const metricRecord = asRecord(metrics);
              if (!metricRecord) {
                return [];
              }
              return Object.entries(metricRecord).map(([metric, value]) => (
                <div key={`${split}-${metric}`} className="dashboard-stat-card">
                  <span>{`${split} ${metric}`}</span>
                  <strong>{formatNumber(value)}</strong>
                </div>
              ));
            })}
          </div>

          {overview.notes && Array.isArray(overview.notes) ? (
            <div className="dashboard-note-list">
              {overview.notes.map((note, index) => (
                <div key={`${section.id}-note-${index}`} className="dashboard-note">
                  {String(note)}
                </div>
              ))}
            </div>
          ) : null}

          {asRecord(overview.artifact_status) ? (
            <div className="dashboard-chip-row">
              {Object.entries(asRecord(overview.artifact_status) ?? {}).map(([key, value]) => (
                <span key={key} className="dashboard-mini-chip">
                  {key.replace(/_/g, " ")}: {String(value)}
                </span>
              ))}
            </div>
          ) : null}
        </>
      ) : null}

      {section.id === "metadata" ? (
        <div className="dashboard-stack">
          {metadataModel ? <KeyValueBlock title="Model metadata" data={metadataModel} /> : null}
          {experimentConfig ? <KeyValueBlock title="Experiment config" data={experimentConfig} /> : null}
        </div>
      ) : null}

      {section.id === "evaluation" && primaryEvaluation ? (
        <div className="dashboard-stack">
          {asRecord(primaryEvaluation.splits) ? (
            <div className="dashboard-stat-grid">
              {Object.entries(asRecord(primaryEvaluation.splits) ?? {}).flatMap(([split, metrics]) => {
                const metricRecord = asRecord(metrics);
                if (!metricRecord) {
                  return [];
                }
                return Object.entries(metricRecord).map(([metric, value]) => (
                  <div key={`${split}-${metric}`} className="dashboard-stat-card">
                    <span>{`${split} ${metric}`}</span>
                    <strong>{formatNumber(value)}</strong>
                  </div>
                ));
              })}
            </div>
          ) : null}
          {asRecord(primaryEvaluation.artifact_paths) ? (
            <KeyValueBlock
              title="Artifact paths"
              data={asRecord(primaryEvaluation.artifact_paths) ?? {}}
            />
          ) : null}
        </div>
      ) : null}

      {section.id === "benchmark" ? (
        <div className="dashboard-stack">
          <TableBlock title="Benchmark rows" eyebrow="Leaderboard" rows={benchmarkRows.slice(0, 10)} />
          {sectionFigures.length ? (
            <div className="dashboard-plot-grid">
              {sectionFigures.map((figure) => (
                <PlotlyFigure key={figure.id} figure={figure} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {section.id === "training_curves" ? (
        <div className="dashboard-stack">
          {trainingHistory ? (
            <div className="dashboard-stat-grid">
              <div className="dashboard-stat-card">
                <span>Train events</span>
                <strong>{asArray(trainingHistory.train_events).length.toLocaleString()}</strong>
              </div>
              <div className="dashboard-stat-card">
                <span>Eval events</span>
                <strong>{asArray(trainingHistory.eval_events).length.toLocaleString()}</strong>
              </div>
              <div className="dashboard-stat-card">
                <span>X axis</span>
                <strong>{String(trainingHistory.x_axis ?? "step")}</strong>
              </div>
            </div>
          ) : null}
          {sectionFigures.length ? (
            <div className="dashboard-plot-grid">
              {sectionFigures.map((figure) => (
                <PlotlyFigure key={figure.id} figure={figure} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {section.id === "learning_curves" ? (
        <div className="dashboard-stack">
          {learningCurves.length ? (
            <TableBlock title="Learning curve points" eyebrow="Curves" rows={learningCurves.slice(0, 8)} />
          ) : null}
          {sectionFigures.length ? (
            <div className="dashboard-plot-grid">
              {sectionFigures.map((figure) => (
                <PlotlyFigure key={figure.id} figure={figure} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {section.id === "cross_dataset" ? (
        <div className="dashboard-stack">
          <TableBlock title="Cross-dataset results" eyebrow="External eval" rows={crossDatasetRows} />
          {sectionFigures.length ? (
            <div className="dashboard-plot-grid">
              {sectionFigures.map((figure) => (
                <PlotlyFigure key={figure.id} figure={figure} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {section.id === "class_distribution" ? (
        <div className="dashboard-stack">
          {classDistribution ? (
            <div className="dashboard-stack">
              <TableBlock
                title="Class distribution overview"
                eyebrow="Distribution"
                rows={asArray(classDistribution.overall).filter(
                  (item): item is Record<string, unknown> => Boolean(asRecord(item)),
                )}
              />
              <TableBlock
                title="Class distribution by split"
                eyebrow="Distribution"
                rows={asArray(classDistribution.splits).filter(
                  (item): item is Record<string, unknown> => Boolean(asRecord(item)),
                )}
              />
            </div>
          ) : null}
          {sourceDistribution ? (
            <div className="dashboard-stack">
              <TableBlock
                title="Source distribution overview"
                eyebrow="Sources"
                rows={asArray(sourceDistribution.overall).filter(
                  (item): item is Record<string, unknown> => Boolean(asRecord(item)),
                )}
              />
              <TableBlock
                title="Source distribution by split"
                eyebrow="Sources"
                rows={asArray(sourceDistribution.splits).filter(
                  (item): item is Record<string, unknown> => Boolean(asRecord(item)),
                )}
              />
            </div>
          ) : null}
          {sectionFigures.length ? (
            <div className="dashboard-plot-grid">
              {sectionFigures.map((figure) => (
                <PlotlyFigure key={figure.id} figure={figure} />
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {section.id === "samples" ? (
        <div className="dashboard-samples">
          {predictionSamples.map((sample, index) => {
            const productionPrediction = asRecord(sample.production_prediction);
            const referencePrediction = asRecord(sample.reference_prediction);
            return (
              <article key={`${section.id}-sample-${index}`} className="sample-card">
                <div className="sample-card__meta">
                  <span>Example {String(sample.example_id ?? index + 1)}</span>
                </div>
                <p>{String(sample.text ?? "")}</p>
                <div className="sample-card__predictions">
                  {productionPrediction ? (
                    <div>
                      <strong>{String(productionPrediction.model ?? "Production")}</strong>
                      <span>
                        {String(productionPrediction.label ?? "—")} · {formatNumber(productionPrediction.confidence)}
                      </span>
                    </div>
                  ) : null}
                  {referencePrediction ? (
                    <div>
                      <strong>{String(referencePrediction.model ?? "Reference")}</strong>
                      <span>
                        {String(referencePrediction.label ?? "—")} · {formatNumber(referencePrediction.confidence)}
                      </span>
                    </div>
                  ) : null}
                </div>
              </article>
            );
          })}
        </div>
      ) : null}

      {section.id === "confusion_matrix" ? (
        <div className="dashboard-image-grid">
          {sectionImages.map((image) => (
            <figure key={image.path} className="dashboard-image-card">
              <img src={image.url} alt={image.title} />
              <figcaption>{image.title}</figcaption>
            </figure>
          ))}
        </div>
      ) : null}

      {![
        "summary",
        "metadata",
        "evaluation",
        "benchmark",
        "training_curves",
        "learning_curves",
        "cross_dataset",
        "class_distribution",
        "samples",
        "confusion_matrix",
      ].includes(section.id) && sectionFigures.length ? (
        <div className="dashboard-plot-grid">
          {sectionFigures.map((figure: DashboardFigure) => (
            <PlotlyFigure key={figure.id} figure={figure} />
          ))}
        </div>
      ) : null}

      {section.reason ? (
        <div className="dashboard-footnote">{section.reason}</div>
      ) : null}
    </article>
  );
}

export function ModelsPage() {
  const {
    managementDomains,
    applySnapshot,
    isLoading,
    error,
    managementReady,
    managementWarning,
  } = useCatalog();
  const [isUploadOpen, setIsUploadOpen] = useState(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [editingModelId, setEditingModelId] = useState<string | null>(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [dashboardCache, setDashboardCache] = useState<Record<string, ModelDashboardResponse>>({});
  const [dashboardError, setDashboardError] = useState<string | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);
  const [mutationError, setMutationError] = useState<string | null>(null);
  const [busyModelId, setBusyModelId] = useState<string | null>(null);
  const [artifactIssueModelId, setArtifactIssueModelId] = useState<string | null>(null);
  const [modelInfoId, setModelInfoId] = useState<string | null>(null);

  const models = useMemo(
    () => managementDomains.flatMap((domain) => domain.models),
    [managementDomains],
  );
  const domainDisplayNames = useMemo(
    () => Object.fromEntries(managementDomains.map((domain) => [domain.domain, domain.display_name])),
    [managementDomains],
  );
  const domainColorTokens = useMemo(
    () => Object.fromEntries(managementDomains.map((domain) => [domain.domain, domain.color_token])),
    [managementDomains],
  );

  const selectedModel = useMemo(
    () => models.find((model) => model.model_id === selectedModelId) ?? null,
    [models, selectedModelId],
  );
  const artifactIssueModel = useMemo(
    () => models.find((model) => model.model_id === artifactIssueModelId) ?? null,
    [artifactIssueModelId, models],
  );
  const modelInfo = useMemo(
    () => models.find((model) => model.model_id === modelInfoId) ?? null,
    [modelInfoId, models],
  );

  const selectedDashboard = selectedModelId ? dashboardCache[selectedModelId] ?? null : null;

  useEffect(() => {
    if (!models.length) {
      setSelectedModelId(null);
      return;
    }
    if (!selectedModelId || !models.some((model) => model.model_id === selectedModelId)) {
      setSelectedModelId(models[0].model_id);
    }
  }, [models, selectedModelId]);

  useEffect(() => {
    const activeModalModel = artifactIssueModel ?? modelInfo;
    if (!activeModalModel) {
      return;
    }
    if ((artifactIssueModelId && !artifactIssueModel) || (modelInfoId && !modelInfo)) {
      setArtifactIssueModelId(null);
      setModelInfoId(null);
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setArtifactIssueModelId(null);
        setModelInfoId(null);
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [artifactIssueModel, artifactIssueModelId, modelInfo, modelInfoId]);

  useEffect(() => {
    if (!managementReady) {
      setDashboardLoading(false);
      setDashboardError(null);
      return;
    }
    setDashboardError(null);
    if (!selectedModelId || dashboardCache[selectedModelId]) {
      return;
    }

    let cancelled = false;
    setDashboardLoading(true);
    setDashboardError(null);

    fetchModelDashboard(selectedModelId)
      .then((dashboard) => {
        if (!cancelled) {
          setDashboardCache((current) => ({
            ...current,
            [selectedModelId]: dashboard,
          }));
        }
      })
      .catch((dashboardLoadError) => {
        if (!cancelled) {
          setDashboardError(
            dashboardLoadError instanceof Error
              ? dashboardLoadError.message
              : "Failed to load dashboard.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) {
          setDashboardLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [dashboardCache, managementReady, selectedModelId]);

  const stats = useMemo(
    () => ({
      totalModels: models.length,
      activeModels: models.filter((model) => model.is_active).length,
      readyModels: models.filter((model) => model.status === "ready").length,
      dashboardReady: models.filter((model) => model.dashboard_status !== "missing").length,
    }),
    [models],
  );

  const applyManagementSnapshot = (snapshot: CatalogSnapshotResponse) => {
    applySnapshot(snapshot);
    setMutationError(null);
  };

  const withModelAction = async (
    modelId: string,
    action: () => Promise<CatalogSnapshotResponse>,
  ) => {
    setBusyModelId(modelId);
    setMutationError(null);
    try {
      const snapshot = await action();
      applyManagementSnapshot(snapshot);
      setDashboardCache((current) => {
        const next = { ...current };
        delete next[modelId];
        return next;
      });
    } catch (actionError) {
      setMutationError(actionError instanceof Error ? actionError.message : "Action failed.");
    } finally {
      setBusyModelId(null);
    }
  };

  const moveModel = async (modelId: string, direction: -1 | 1) => {
    const index = models.findIndex((model) => model.model_id === modelId);
    const targetIndex = index + direction;
    if (index < 0 || targetIndex < 0 || targetIndex >= models.length) {
      return;
    }
    const orderedIds = models.map((model) => model.model_id);
    const [moved] = orderedIds.splice(index, 1);
    orderedIds.splice(targetIndex, 0, moved);
    await withModelAction(modelId, () => reorderModels({ ordered_model_ids: orderedIds }));
  };

  return (
    <section className="models-workspace">
      <div className="panel models-hero">
        <div className="models-hero__copy">
          <div className="panel__eyebrow">Models</div>
          <h1>Manage local models and inspect their dashboards from one workspace.</h1>
          <p>
            This page now drives the live registry. Enable, reorder, rename, delete, or upload
            models here and Home updates from the same catalog immediately.
          </p>
        </div>

        <div className="models-hero__stats">
          <div className="models-hero__stat">
            <span>Total models</span>
            <strong>{stats.totalModels}</strong>
          </div>
          <div className="models-hero__stat">
            <span>Active in Home</span>
            <strong>{stats.activeModels}</strong>
          </div>
          <div className="models-hero__stat">
            <span>Runtime-ready</span>
            <strong>{stats.readyModels}</strong>
          </div>
          <div className="models-hero__stat">
            <span>With dashboards</span>
            <strong>{stats.dashboardReady}</strong>
          </div>
        </div>
      </div>

      <div className="models-layout">
        <section className="panel models-management">
          <div className="models-management__header">
            <div>
              <div className="panel__eyebrow">Model Management</div>
              <h2>Registry control</h2>
              <p>Local manifests, runtime status, artifact health, and Home sync live here.</p>
            </div>

            <button
              className="primary-button"
              type="button"
              disabled={!managementReady}
              onClick={() => setIsUploadOpen(true)}
            >
              Upload model
            </button>
          </div>

          {error ? <div className="inline-alert">{error}</div> : null}
          {managementWarning ? (
            <div className="inline-alert">
              {managementWarning} Model Management is in read-only fallback mode until the updated
              backend is running.
            </div>
          ) : null}
          {mutationError ? <div className="inline-alert">{mutationError}</div> : null}

          {isUploadOpen && managementReady ? (
            <ModelUploadWizard
              isOpen={isUploadOpen}
              onClose={() => setIsUploadOpen(false)}
              onSuccess={(snapshot, modelId) => {
                applyManagementSnapshot(snapshot);
                setSelectedModelId(modelId);
              }}
            />
          ) : null}

          <div className="management-subheader">
            <div className="management-subheader__copy">
              <span>Ordered registry</span>
              <strong>Home reads this order from top to bottom.</strong>
            </div>
            <div className="dashboard-chip-row">
              <span className="dashboard-mini-chip">Active: {stats.activeModels}</span>
              <span className="dashboard-mini-chip">Ready: {stats.readyModels}</span>
            </div>
          </div>

          {isLoading && !models.length ? (
            <div className="dashboard-empty">
              <strong>Loading model registry…</strong>
            </div>
          ) : null}

          {!isLoading && !models.length ? (
            <div className="dashboard-empty">
              <strong>No models registered locally yet.</strong>
              <p>Upload the first model to create a local manifest and optionally attach a dashboard.</p>
            </div>
          ) : null}

          <div className="model-list">
            {models.map((model, index) => {
              const theme = getDomainTheme({
                domain: model.domain,
                color_token: domainColorTokens[model.domain] ?? model.domain,
              });
              const isSelected = selectedModelId === model.model_id;
              const isEditing = editingModelId === model.model_id;
              return (
                <article
                  key={model.model_id}
                  className={`model-row${isSelected ? " model-row--selected" : ""}`}
                  style={
                    {
                      "--model-accent": theme.accent,
                      "--model-glow": theme.glow,
                    } as CSSProperties
                  }
                  onClick={() => setSelectedModelId(model.model_id)}
                >
                  <div className="model-row__topline">
                    <span className="model-row__domain">
                      {getCompactDomainLabel({
                        domain: model.domain,
                        display_name: domainDisplayNames[model.domain] ?? model.domain,
                      })}
                    </span>
                    <div className="model-row__chips">
                      <span className={`status-chip status-chip--${model.is_active ? "available" : "missing"}`}>
                        {model.is_active ? "live in Home" : "disabled"}
                      </span>
                      <span className={`status-chip status-chip--${model.status}`}>
                        {humanizeStatus(model.status)}
                      </span>
                      <span className={`status-chip status-chip--${model.dashboard_status}`}>
                        dashboard {model.dashboard_sections_available}/{model.dashboard_sections_total || 0}
                      </span>
                    </div>
                  </div>

                  <div className="model-row__body">
                    <div className="model-row__identity">
                      {isEditing ? (
                        <div className="inline-rename">
                          <input
                            value={renameDraft}
                            onChange={(event) => setRenameDraft(event.target.value)}
                            onClick={(event) => event.stopPropagation()}
                          />
                          <button
                            type="button"
                            className="mini-button"
                            onClick={(event) => {
                              event.stopPropagation();
                              void withModelAction(model.model_id, () =>
                                updateModel(model.model_id, { display_name: renameDraft.trim() }),
                              );
                              setEditingModelId(null);
                            }}
                            disabled={!renameDraft.trim()}
                          >
                            Save
                          </button>
                          <button
                            type="button"
                            className="mini-button"
                            onClick={(event) => {
                              event.stopPropagation();
                              setEditingModelId(null);
                            }}
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <>
                          <h3>{model.display_name}</h3>
                        </>
                      )}
                    </div>

                    {model.status === "missing_artifacts" && model.missing_artifacts.length ? (
                      <button
                        type="button"
                        className="model-row__warning-button"
                        onClick={(event) => {
                          event.stopPropagation();
                          setArtifactIssueModelId(model.model_id);
                        }}
                      >
                        <span>Artifacts missing</span>
                        <strong>
                          {model.missing_artifacts.length} unresolved runtime path
                          {model.missing_artifacts.length === 1 ? "" : "s"}
                        </strong>
                      </button>
                    ) : model.status_reason ? (
                      <div className="model-row__warning">{model.status_reason}</div>
                    ) : null}
                  </div>

                  <div className="model-row__actions">
                    <button
                      className="mini-button"
                      type="button"
                      disabled={!managementReady || index === 0 || busyModelId === model.model_id}
                      onClick={(event) => {
                        event.stopPropagation();
                        void moveModel(model.model_id, -1);
                      }}
                    >
                      Move up
                    </button>
                    <button
                      className="mini-button"
                      type="button"
                      disabled={
                        !managementReady ||
                        index === models.length - 1 ||
                        busyModelId === model.model_id
                      }
                      onClick={(event) => {
                        event.stopPropagation();
                        void moveModel(model.model_id, 1);
                      }}
                    >
                      Move down
                    </button>
                    <button
                      className="mini-button"
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation();
                        setModelInfoId(model.model_id);
                      }}
                    >
                      Model info
                    </button>
                    <button
                      className="mini-button"
                      type="button"
                      disabled={!managementReady}
                      onClick={(event) => {
                        event.stopPropagation();
                        setEditingModelId(model.model_id);
                        setRenameDraft(model.display_name);
                      }}
                    >
                      Rename
                    </button>
                    <button
                      className="mini-button"
                      type="button"
                      disabled={
                        !managementReady ||
                        (!model.can_activate && !model.is_active) ||
                        busyModelId === model.model_id
                      }
                      onClick={(event) => {
                        event.stopPropagation();
                        void withModelAction(model.model_id, () =>
                          updateModel(model.model_id, { is_active: !model.is_active }),
                        );
                      }}
                    >
                      {model.is_active ? "Disable" : "Enable"}
                    </button>
                    <button
                      className="mini-button mini-button--danger"
                      type="button"
                      disabled={!managementReady || busyModelId === model.model_id}
                      onClick={(event) => {
                        event.stopPropagation();
                        const confirmed = window.confirm(
                          `Delete '${model.display_name}' and remove its local files from app/app-models?`,
                        );
                        if (!confirmed) {
                          return;
                        }
                        void withModelAction(model.model_id, () => deleteModelRequest(model.model_id));
                      }}
                    >
                      Delete
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        </section>

        <section className="panel models-dashboard">
          <div className="models-dashboard__header">
            <div>
              <div className="panel__eyebrow">Model Dashboard</div>
              <h2>{selectedModel?.display_name ?? "Select a model"}</h2>
              <p>
                Optional dashboards are loaded per model. Missing data does not break the workspace;
                each section falls back gracefully.
              </p>
            </div>

            {selectedModel ? (
              <div className="models-dashboard__meta">
                <span className={`status-chip status-chip--${selectedModel.status}`}>
                  {humanizeStatus(selectedModel.status)}
                </span>
                <span className={`status-chip status-chip--${selectedModel.dashboard_status}`}>
                  {selectedModel.dashboard_status}
                </span>
              </div>
            ) : null}
          </div>

          {dashboardError ? <div className="inline-alert">{dashboardError}</div> : null}

          {!selectedModel ? (
            <div className="dashboard-empty">
              <strong>Select a model from the management list.</strong>
            </div>
          ) : !managementReady ? (
            <div className="dashboard-empty">
              <strong>Dashboard API is not available yet.</strong>
              <p>Restart the backend on the updated patch to enable dashboard loading and write actions.</p>
            </div>
          ) : dashboardLoading && !selectedDashboard ? (
            <div className="dashboard-empty">
              <strong>Loading dashboard…</strong>
            </div>
          ) : selectedDashboard && selectedDashboard.available ? (
            <div className="dashboard-stack">
              <div className="dashboard-overview-bar">
                <div className="dashboard-overview-bar__item">
                  <span>Domain</span>
                  <strong>{selectedModel.domain}</strong>
                </div>
                <div className="dashboard-overview-bar__item">
                  <span>Generated</span>
                  <strong>{formatTimestamp(selectedDashboard.manifest?.generated_at)}</strong>
                </div>
                <div className="dashboard-overview-bar__item">
                  <span>Sections</span>
                  <strong>
                    {selectedModel.dashboard_sections_available}/{selectedModel.dashboard_sections_total}
                  </strong>
                </div>
                <div className="dashboard-overview-bar__item">
                  <span>Runtime</span>
                  <strong>{familyLabel(selectedModel)}</strong>
                </div>
              </div>

              {selectedDashboard.source_audit ? (
                <div className="dashboard-chip-row">
                  {Object.entries(
                    asRecord(asRecord(selectedDashboard.source_audit)?.artifact_counts) ?? {},
                  ).map(([key, value]) => (
                    <span key={key} className="dashboard-mini-chip">
                      {key.replace(/_/g, " ")}: {String(value)}
                    </span>
                  ))}
                </div>
              ) : null}

              {selectedDashboard.manifest?.notes.length ? (
                <div className="dashboard-note-list">
                  {selectedDashboard.manifest.notes.map((note) => (
                    <div key={note} className="dashboard-note">
                      {note}
                    </div>
                  ))}
                </div>
              ) : null}

              {selectedDashboard.manifest?.sections.map((section) =>
                renderDashboardSection(section, selectedDashboard),
              )}

              {selectedDashboard.manifest?.selected_sources.length ? (
                <article className="dashboard-section">
                  <div className="dashboard-section__header">
                    <div>
                      <div className="dashboard-section__eyebrow">Sources</div>
                      <h3>Selected source files</h3>
                      <p>Dashboard provenance captured from the manifest.</p>
                    </div>
                  </div>

                  <div className="source-list">
                    {selectedDashboard.manifest.selected_sources.map((source) => (
                      <div key={`${source.category}-${source.path}`} className="source-list__item">
                        <span>{source.category}</span>
                        <strong>{source.path}</strong>
                        <p>{source.reason ?? "No reason attached."}</p>
                      </div>
                    ))}
                  </div>
                </article>
              ) : null}
            </div>
          ) : (
            <div className="dashboard-empty">
              <strong>No dashboard attached to this model.</strong>
              <p>
                The model can still be enabled, reordered, renamed, or used in Home. Attach a
                standardized dashboard folder during upload if you want evaluation views here.
              </p>
            </div>
          )}
        </section>
      </div>

      {artifactIssueModel ? (
        <div
          className="artifact-modal"
          role="presentation"
          onClick={() => setArtifactIssueModelId(null)}
        >
          <div
            className="artifact-modal__card"
            role="dialog"
            aria-modal="true"
            aria-labelledby="artifact-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="artifact-modal__header">
              <div>
                <div className="panel__eyebrow">Artifact Check</div>
                <h3 id="artifact-modal-title">{artifactIssueModel.display_name}</h3>
                <p>
                  These required runtime artifacts are still unresolved. The model stays disabled
                  until they are available locally.
                </p>
              </div>
              <button
                type="button"
                className="mini-button"
                onClick={() => setArtifactIssueModelId(null)}
              >
                Close
              </button>
            </div>

            <div className="artifact-modal__summary">
              <span className="status-chip status-chip--incompatible">Activation blocked</span>
              <span className="dashboard-mini-chip">
                Missing: {artifactIssueModel.missing_artifacts.length}
              </span>
            </div>

            <div className="artifact-modal__list">
              {artifactIssueModel.missing_artifacts.map((artifactPath) => (
                <div key={artifactPath} className="artifact-modal__item">
                  {artifactPath}
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {modelInfo ? (
        <div
          className="artifact-modal"
          role="presentation"
          onClick={() => setModelInfoId(null)}
        >
          <div
            className="artifact-modal__card model-info-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="model-info-modal-title"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="artifact-modal__header">
              <div>
                <div className="panel__eyebrow">Model Info</div>
                <h3 id="model-info-modal-title">{modelInfo.display_name}</h3>
                <p>
                  Full manifest-level information for this model, separated from the compact card
                  layout.
                </p>
              </div>
              <button
                type="button"
                className="mini-button"
                onClick={() => setModelInfoId(null)}
              >
                Close
              </button>
            </div>

            <div className="artifact-modal__summary">
              <span className={`status-chip status-chip--${modelInfo.status}`}>
                {humanizeStatus(modelInfo.status)}
              </span>
              <span className="dashboard-mini-chip">{familyLabel(modelInfo)}</span>
              <span className="dashboard-mini-chip">
                priority: {modelInfo.priority}
              </span>
            </div>

            <div className="model-info-grid">
              <div className="model-info-item">
                <span>Domain</span>
                <strong>{modelInfo.domain}</strong>
              </div>
              <div className="model-info-item">
                <span>Model id</span>
                <strong>{modelInfo.model_id}</strong>
              </div>
              <div className="model-info-item">
                <span>Version</span>
                <strong>{modelInfo.version ?? "—"}</strong>
              </div>
              <div className="model-info-item">
                <span>Library</span>
                <strong>{modelInfo.framework_library ?? "—"}</strong>
              </div>
              <div className="model-info-item">
                <span>Framework</span>
                <strong>{modelInfo.framework_type}</strong>
              </div>
              <div className="model-info-item">
                <span>Architecture</span>
                <strong>{modelInfo.architecture ?? modelInfo.backbone ?? "—"}</strong>
              </div>
              <div className="model-info-item">
                <span>Runtime device</span>
                <strong>{modelInfo.runtime_device ?? "—"}</strong>
              </div>
              <div className="model-info-item">
                <span>Max sequence length</span>
                <strong>{modelInfo.runtime_max_sequence_length ?? "—"}</strong>
              </div>
              <div className="model-info-item model-info-item--wide">
                <span>Description</span>
                <strong>{modelInfo.description ?? "No description attached to this local manifest."}</strong>
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
