import { useEffect, useMemo, useState, type CSSProperties } from "react";
import { ModelDashboardPanel } from "../components/models/ModelDashboardPanel";
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
  DomainCatalogModel,
  ModelDashboardResponse,
} from "../types/contracts";

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
            </div>
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
            <ModelDashboardPanel model={selectedModel} dashboard={selectedDashboard} />
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
