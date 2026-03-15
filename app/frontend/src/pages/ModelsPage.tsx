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

function asRecord(value: unknown) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function toRecordArray(value: unknown) {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => item !== null && typeof item === "object")
    : [];
}

function formatOutputType(value?: string | null, compact = false) {
  if (!value) {
    return "—";
  }

  const normalized = value.toLowerCase();
  if (normalized === "single-label-classification") {
    return compact ? "single-label:" : "single-label";
  }

  const humanized = value.replace(/_/g, " ").replace(/-/g, " ");
  return compact ? `${humanized}:` : humanized;
}

function modelInfoChipTone(index: number) {
  return ["aurora", "cyan", "gold", "rose"][index % 4];
}

function moveModelIdOneStep(modelIds: string[], modelId: string, direction: -1 | 1) {
  const index = modelIds.indexOf(modelId);
  const targetIndex = index + direction;
  if (index < 0 || targetIndex < 0 || targetIndex >= modelIds.length) {
    return null;
  }

  const nextIds = [...modelIds];
  const [movedId] = nextIds.splice(index, 1);
  nextIds.splice(targetIndex, 0, movedId);
  return nextIds;
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
  const [statusReasonModelId, setStatusReasonModelId] = useState<string | null>(null);

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
  const activeModels = useMemo(
    () => models.filter((model) => model.is_active),
    [models],
  );
  const disabledModels = useMemo(
    () => models.filter((model) => !model.is_active),
    [models],
  );

  const selectedDashboard = selectedModelId ? dashboardCache[selectedModelId] ?? null : null;
  const modelInfoDashboard = modelInfoId ? dashboardCache[modelInfoId] ?? null : null;
  const modelInfoMetadata = asRecord(modelInfoDashboard?.documents["metadata/model.json"]);
  const modelInfoLabels = asRecord(modelInfoMetadata?.labels);
  const modelInfoLabelClasses = toRecordArray(modelInfoLabels?.classes);
  const modelInfoOutputType = String(modelInfoLabels?.type ?? modelInfo?.output_type ?? "");

  useEffect(() => {
    if (!models.length) {
      setSelectedModelId(null);
      setStatusReasonModelId(null);
      return;
    }
    if (!selectedModelId || !models.some((model) => model.model_id === selectedModelId)) {
      setSelectedModelId(models[0].model_id);
    }
    if (statusReasonModelId && !models.some((model) => model.model_id === statusReasonModelId)) {
      setStatusReasonModelId(null);
    }
  }, [models, selectedModelId, statusReasonModelId]);

  useEffect(() => {
    if (!statusReasonModelId) {
      return;
    }

    const handlePointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (target instanceof Element && target.closest(".model-row__status-popover-anchor")) {
        return;
      }
      setStatusReasonModelId(null);
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setStatusReasonModelId(null);
      }
    };

    window.addEventListener("pointerdown", handlePointerDown);
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("pointerdown", handlePointerDown);
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [statusReasonModelId]);

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

  useEffect(() => {
    if (!managementReady || !modelInfoId || dashboardCache[modelInfoId]) {
      return;
    }

    let cancelled = false;

    fetchModelDashboard(modelInfoId)
      .then((dashboard) => {
        if (!cancelled) {
          setDashboardCache((current) => ({
            ...current,
            [modelInfoId]: dashboard,
          }));
        }
      })
      .catch(() => {
        // Modal can fall back to manifest-level fields when dashboard metadata is unavailable.
      });

    return () => {
      cancelled = true;
    };
  }, [dashboardCache, managementReady, modelInfoId]);

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

  const moveModelWithinSection = (model: DomainCatalogModel, direction: -1 | 1) => {
    const sectionModels = model.is_active ? activeModels : disabledModels;
    const sectionIds = sectionModels.map((sectionModel) => sectionModel.model_id);
    const nextSectionIds = moveModelIdOneStep(sectionIds, model.model_id, direction);
    if (!nextSectionIds) {
      return;
    }

    const orderedIds = model.is_active
      ? [...nextSectionIds, ...disabledModels.map((sectionModel) => sectionModel.model_id)]
      : [...activeModels.map((sectionModel) => sectionModel.model_id), ...nextSectionIds];

    void withModelAction(model.model_id, () => reorderModels({ ordered_model_ids: orderedIds }));
  };

  const renderModelCard = (
    model: DomainCatalogModel,
    sectionIndex: number,
    sectionLength: number,
  ) => {
    const theme = getDomainTheme({
      domain: model.domain,
      color_token: domainColorTokens[model.domain] ?? model.domain,
    });
    const isSelected = selectedModelId === model.model_id;
    const isEditing = editingModelId === model.model_id;
    const reorderDisabled =
      !managementReady || busyModelId !== null || editingModelId !== null || isUploadOpen;
    const showStatusReason =
      statusReasonModelId === model.model_id && model.status === "incompatible" && Boolean(model.status_reason);

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

          <div className="model-row__topline-meta">
            <div className="model-row__chips">
              <span className={`status-chip status-chip--${model.is_active ? "available" : "missing"}`}>
                {model.is_active ? "enabled" : "disabled"}
              </span>
              {model.status === "incompatible" && model.status_reason ? (
                <div className="model-row__status-popover-anchor">
                  <button
                    type="button"
                    className={`status-chip status-chip-button status-chip--${model.status}`}
                    aria-expanded={showStatusReason}
                    aria-label={`Show incompatibility details for ${model.display_name}`}
                    onClick={(event) => {
                      event.stopPropagation();
                      setStatusReasonModelId((current) => (current === model.model_id ? null : model.model_id));
                    }}
                  >
                    {humanizeStatus(model.status)}
                  </button>
                  {showStatusReason ? (
                    <div
                      className="model-row__status-popover"
                      role="dialog"
                      aria-label={`${model.display_name} incompatibility details`}
                      onClick={(event) => event.stopPropagation()}
                    >
                      {model.status_reason}
                    </div>
                  ) : null}
                </div>
              ) : (
                <span className={`status-chip status-chip--${model.status}`}>
                  {humanizeStatus(model.status)}
                </span>
              )}
            </div>

            <div className="model-row__reorder-controls">
              <button
                type="button"
                className="model-row__reorder-button"
                aria-label={`Move ${model.display_name} left`}
                title={`Move ${model.display_name} left`}
                disabled={reorderDisabled || sectionIndex === 0}
                onClick={(event) => {
                  event.stopPropagation();
                  moveModelWithinSection(model, -1);
                }}
              >
                &larr;
              </button>
              <button
                type="button"
                className="model-row__reorder-button"
                aria-label={`Move ${model.display_name} right`}
                title={`Move ${model.display_name} right`}
                disabled={reorderDisabled || sectionIndex === sectionLength - 1}
                onClick={(event) => {
                  event.stopPropagation();
                  moveModelWithinSection(model, 1);
                }}
              >
                &rarr;
              </button>
            </div>
          </div>
        </div>

        <div className="model-row__body">
          <div className="model-row__identity">
            <div className="model-row__headline">
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
                <h3>{model.display_name}</h3>
              )}
            </div>
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
          ) : null}
        </div>

        <div className="model-row__actions">
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
            disabled={!managementReady || isUploadOpen}
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
              busyModelId === model.model_id ||
              isUploadOpen
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
            disabled={!managementReady || busyModelId === model.model_id || isUploadOpen}
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
  };

  return (
    <section className="models-workspace">
      <div className="panel models-hero">
        <div className="models-hero__copy">
          <div className="panel__eyebrow">Models</div>
          <h1>
            Manage local <span className="about-hero__title-accent">models</span> and inspect
            their <span className="about-hero__title-accent">dashboards</span> from one workspace.
          </h1>
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
              <div className="models-management__title-row">
                <div className="panel__eyebrow">Model Management</div>
                <details className="models-management__info">
                  <summary aria-label="About Model Management">
                    <span>i</span>
                  </summary>
                  <div className="models-management__popover">
                    Local manifests, runtime status, artifact health, and Home sync live here.
                  </div>
                </details>
              </div>
            </div>

            <button
              className="primary-button models-management__upload-button"
              type="button"
              disabled={!managementReady || isUploadOpen}
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
            {activeModels.length || !disabledModels.length ? (
              <div className="model-list__section">
                {disabledModels.length ? (
                  <div className="model-list__section-header">
                    <span>Active models</span>
                    <strong>{activeModels.length}</strong>
                  </div>
                ) : null}
                <div className="model-list__grid">
                  {activeModels.map((model, index) => renderModelCard(model, index, activeModels.length))}
                </div>
              </div>
            ) : null}

            {disabledModels.length ? (
              <div className="model-list__section">
                <div className="model-list__section-header">
                  <span>Disabled models</span>
                  <strong>{disabledModels.length}</strong>
                </div>
                <div className="model-list__grid model-list__grid--disabled">
                  {disabledModels.map((model, index) =>
                    renderModelCard(model, index, disabledModels.length),
                  )}
                </div>
              </div>
            ) : null}
          </div>
        </section>

        <section className="panel models-dashboard">
          <div className="models-dashboard__header">
            <div>
              <div className="panel__eyebrow">Model Dashboard</div>
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
            <ModelDashboardPanel
              model={selectedModel}
              dashboard={selectedDashboard}
              onOpenModelInfo={() => setModelInfoId(selectedModel.model_id)}
            />
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

      {isUploadOpen && managementReady ? (
        <ModelUploadWizard
          isOpen={isUploadOpen}
          domains={managementDomains.map((domain) => ({
            domain: domain.domain,
            display_name: domain.display_name,
            color_token: domain.color_token,
            group: domain.group,
          }))}
          onClose={() => setIsUploadOpen(false)}
          onSuccess={(snapshot, modelId) => {
            applyManagementSnapshot(snapshot);
            setSelectedModelId(modelId);
          }}
        />
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
                <span>Task</span>
                <strong>{modelInfo.framework_task ?? "—"}</strong>
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
                <span>Max sequence length</span>
                <strong>{modelInfo.runtime_max_sequence_length ?? "—"}</strong>
              </div>
              <div
                className={`model-info-item model-info-item--output${
                  modelInfoLabelClasses.length ? " model-info-item--wide" : ""
                }`}
              >
                <span>Output</span>
                {modelInfoLabelClasses.length ? (
                  <div className="model-info-output">
                    <strong className="model-info-output__label">
                      {formatOutputType(modelInfoOutputType, true)}
                    </strong>
                    <div className="model-info-output__chips">
                      {modelInfoLabelClasses.map((label, index) => (
                        <span
                          key={String(label.id ?? label.name ?? index)}
                          className={`dashboard-mini-chip model-info-output__chip model-info-output__chip--${modelInfoChipTone(index)}`}
                        >
                          {String(label.display_name ?? label.name ?? label.id)}
                        </span>
                      ))}
                    </div>
                  </div>
                ) : (
                  <strong>{formatOutputType(modelInfoOutputType)}</strong>
                )}
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
