import { useEffect, useRef, useState } from "react";
import { uploadModel as uploadModelRequest } from "../../services/api";
import type {
  CatalogSnapshotResponse,
  UploadLabelClass,
  UploadModelMetadata,
} from "../../types/contracts";

type FrameworkType = UploadModelMetadata["framework_type"];

interface ArtifactRequirement {
  slot: string;
  title: string;
  required: boolean;
  accepts: string;
  hint: string;
}

const ARTIFACT_REQUIREMENTS: Record<FrameworkType, ArtifactRequirement[]> = {
  transformers: [
    {
      slot: "weights",
      title: "Weights",
      required: true,
      accepts: ".safetensors,.bin,.pt,.pth",
      hint: "Checkpoint weights for the transformer model.",
    },
    {
      slot: "tokenizer",
      title: "Tokenizer",
      required: true,
      accepts: ".json,.txt,.model,.vocab",
      hint: "Tokenizer files such as tokenizer.json and tokenizer_config.json.",
    },
    {
      slot: "config",
      title: "Config",
      required: true,
      accepts: ".json,.yaml,.yml,.bin",
      hint: "Model/config artifacts required by AutoModel and AutoTokenizer.",
    },
    {
      slot: "label_map_file",
      title: "Label Map",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional encoder or label map artifact.",
    },
    {
      slot: "label_classes_file",
      title: "Label Classes",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional exported label classes file.",
    },
  ],
  pytorch: [
    {
      slot: "weights",
      title: "Weights",
      required: true,
      accepts: ".pt,.pth,.bin",
      hint: "PyTorch checkpoint or state dict.",
    },
    {
      slot: "vocabulary",
      title: "Vocabulary",
      required: true,
      accepts: ".pkl,.json,.txt",
      hint: "Vocabulary or preprocessing lookup assets.",
    },
    {
      slot: "config",
      title: "Config",
      required: true,
      accepts: ".json,.yaml,.yml,.bin",
      hint: "Runtime and architecture configuration files.",
    },
    {
      slot: "label_classes_file",
      title: "Label Classes",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional label classes artifact.",
    },
    {
      slot: "label_encoder_file",
      title: "Label Encoder",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional encoder artifact.",
    },
  ],
  sklearn: [
    {
      slot: "weights",
      title: "Serialized Model",
      required: true,
      accepts: ".pkl,.joblib,.bin",
      hint: "Serialized estimator artifact.",
    },
    {
      slot: "config",
      title: "Feature Config",
      required: true,
      accepts: ".json,.yaml,.yml,.txt",
      hint: "Feature extraction or runtime config.",
    },
    {
      slot: "label_classes_file",
      title: "Label Classes",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional label classes artifact.",
    },
    {
      slot: "label_encoder_file",
      title: "Label Encoder",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional encoder artifact.",
    },
  ],
};

const DEFAULT_LABELS: UploadLabelClass[] = [
  { id: 0, name: "class_0", display_name: "Class 0" },
];

const DEFAULT_FORM: UploadModelMetadata & { model_config_text: string; runtime_padding_mode: string } = {
  model_id: "",
  domain: "",
  display_name: "",
  description: "",
  version: "",
  enable_on_upload: false,
  framework_type: "transformers",
  framework_task: "sequence-classification",
  framework_library: "huggingface",
  backbone: "",
  architecture: "",
  base_model: "",
  embeddings: "",
  output_type: "single-label-classification",
  runtime_device: "auto",
  runtime_max_sequence_length: 256,
  runtime_batch_size: 1,
  runtime_truncation: true,
  runtime_padding: true,
  runtime_padding_mode: "true",
  ui_display_name: "",
  color_token: "",
  group: "",
  labels: DEFAULT_LABELS,
  model_config: {},
  model_config_text: "{}",
};

function createDefaultForm() {
  return {
    ...DEFAULT_FORM,
    labels: DEFAULT_FORM.labels.map((label) => ({ ...label })),
  };
}

interface ModelUploadWizardProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: (snapshot: CatalogSnapshotResponse, modelId: string) => void;
}

function parseAcceptedExtensions(accepts: string) {
  return accepts
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

function normalizeDashboardRelativePath(file: File) {
  const raw = file.webkitRelativePath || file.name;
  return raw.replace(/\\/g, "/");
}

function summarizeDashboardReference(fileRef: string) {
  const normalized = fileRef.replace(/\\/g, "/");
  const dashboardIndex = normalized.indexOf("/dashboard/");
  if (dashboardIndex >= 0) {
    return normalized.slice(dashboardIndex + "/dashboard/".length);
  }
  const appModelsIndex = normalized.indexOf("app/app-models/");
  if (appModelsIndex >= 0) {
    return normalized.split("/").slice(-2).join("/");
  }
  return normalized;
}

export function ModelUploadWizard({
  isOpen,
  onClose,
  onSuccess,
}: ModelUploadWizardProps) {
  const [step, setStep] = useState<"metadata" | "artifacts">("metadata");
  const [form, setForm] = useState(createDefaultForm);
  const [artifactFiles, setArtifactFiles] = useState<Record<string, File[]>>({});
  const [dashboardFiles, setDashboardFiles] = useState<File[]>([]);
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({});
  const [artifactErrors, setArtifactErrors] = useState<Record<string, string>>({});
  const [dashboardError, setDashboardError] = useState<string | null>(null);
  const [serverError, setServerError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const dashboardInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    if (!dashboardInputRef.current) {
      return;
    }
    dashboardInputRef.current.setAttribute("webkitdirectory", "");
    dashboardInputRef.current.setAttribute("directory", "");
  }, []);

  if (!isOpen) {
    return null;
  }

  const requirements = ARTIFACT_REQUIREMENTS[form.framework_type];

  const setFormField = <Key extends keyof typeof form>(key: Key, value: (typeof form)[Key]) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const handleLabelChange = (
    index: number,
    key: keyof UploadLabelClass,
    value: string | number,
  ) => {
    setForm((current) => ({
      ...current,
      labels: current.labels.map((label, labelIndex) =>
        labelIndex === index
          ? {
              ...label,
              [key]: value,
            }
          : label,
      ),
    }));
  };

  const resetWizard = () => {
    setStep("metadata");
    setForm(createDefaultForm());
    setArtifactFiles({});
    setDashboardFiles([]);
    setFieldErrors({});
    setArtifactErrors({});
    setDashboardError(null);
    setServerError(null);
    setIsSubmitting(false);
  };

  const closeWizard = () => {
    resetWizard();
    onClose();
  };

  const validateMetadataStep = () => {
    const nextErrors: Record<string, string> = {};

    if (!form.model_id.trim()) {
      nextErrors.model_id = "Model id is required.";
    }
    if (!form.domain.trim()) {
      nextErrors.domain = "Domain is required.";
    }
    if (!form.display_name.trim()) {
      nextErrors.display_name = "Display name is required.";
    }
    if (!form.ui_display_name?.trim()) {
      nextErrors.ui_display_name = "Domain display name is required.";
    }
    if (form.runtime_max_sequence_length < 1) {
      nextErrors.runtime_max_sequence_length = "Sequence length must be at least 1.";
    }
    if (form.runtime_batch_size < 1) {
      nextErrors.runtime_batch_size = "Batch size must be at least 1.";
    }
    if (form.labels.length === 0) {
      nextErrors.labels = "At least one label is required.";
    }

    const seenIds = new Set<number>();
    const seenNames = new Set<string>();
    form.labels.forEach((label, index) => {
      if (label.name.trim().length === 0) {
        nextErrors[`label-name-${index}`] = "Label name is required.";
      }
      if (seenIds.has(label.id)) {
        nextErrors[`label-id-${index}`] = "Label ids must be unique.";
      }
      if (seenNames.has(label.name.trim().toLowerCase())) {
        nextErrors[`label-name-${index}`] = "Label names must be unique.";
      }
      seenIds.add(label.id);
      seenNames.add(label.name.trim().toLowerCase());
    });

    if (form.model_config_text.trim()) {
      try {
        const parsed = JSON.parse(form.model_config_text);
        if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
          nextErrors.model_config = "Model config JSON must be an object.";
        }
      } catch {
        nextErrors.model_config = "Model config JSON is invalid.";
      }
    }

    setFieldErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const validateArtifactStep = async () => {
    const nextErrors: Record<string, string> = {};
    let nextDashboardError: string | null = null;
    for (const requirement of requirements) {
      const files = artifactFiles[requirement.slot] ?? [];
      if (requirement.required && files.length === 0) {
        nextErrors[requirement.slot] = `${requirement.title} is required.`;
        continue;
      }

      const allowedExtensions = parseAcceptedExtensions(requirement.accepts);
      const invalidFile = files.find((file) => {
        const extension = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
        return !allowedExtensions.includes(extension);
      });

      if (invalidFile) {
        nextErrors[requirement.slot] = `${invalidFile.name} has an unsupported file type.`;
      }
    }

    if (dashboardFiles.length > 0) {
      const manifestFile = dashboardFiles.find((file) =>
        normalizeDashboardRelativePath(file).endsWith("dashboard-manifest.json"),
      );

      if (!manifestFile) {
        nextDashboardError = "Dashboard folder must include dashboard-manifest.json.";
      } else {
        try {
          const manifest = JSON.parse(await manifestFile.text()) as {
            entrypoints?: Record<string, string>;
            sections?: Array<{ id?: string; files?: string[] }>;
          };
          const relativePaths = new Set(dashboardFiles.map(normalizeDashboardRelativePath));
          const missingRefs: string[] = [];
          const references = [
            ...Object.values(manifest.entrypoints ?? {}),
            ...(manifest.sections ?? []).flatMap((section) => section.files ?? []),
          ];
          for (const reference of references) {
            const normalized = summarizeDashboardReference(reference);
            const referenceSegments = reference.split("/");
            const byName = referenceSegments[referenceSegments.length - 1] ?? reference;
            const matches = Array.from(relativePaths).some(
              (path) => path.endsWith(normalized) || path.endsWith(byName),
            );
            if (!matches) {
              missingRefs.push(reference);
            }
          }

          if (missingRefs.length > 0) {
            nextDashboardError =
              `Dashboard manifest references missing files: ${missingRefs.slice(0, 3).join(", ")}${missingRefs.length > 3 ? "..." : ""}`;
          }
        } catch {
          nextDashboardError = "Dashboard manifest JSON is invalid.";
        }
      }
    }

    setArtifactErrors(nextErrors);
    setDashboardError(nextDashboardError);
    return Object.keys(nextErrors).length === 0 && !nextDashboardError;
  };

  const handleSubmit = async () => {
    setServerError(null);
    const metadataIsValid = validateMetadataStep();
    const artifactsAreValid = await validateArtifactStep();
    if (!metadataIsValid || !artifactsAreValid) {
      setStep(metadataIsValid ? "artifacts" : "metadata");
      return;
    }

    let parsedModelConfig: Record<string, unknown> = {};
    if (form.model_config_text.trim()) {
      parsedModelConfig = JSON.parse(form.model_config_text);
    }

    const formData = new FormData();
    const payload: UploadModelMetadata = {
      model_id: form.model_id,
      domain: form.domain,
      display_name: form.display_name,
      description: form.description,
      version: form.version,
      enable_on_upload: form.enable_on_upload,
      framework_type: form.framework_type,
      framework_task: form.framework_task,
      framework_library: form.framework_library,
      backbone: form.backbone,
      architecture: form.architecture,
      base_model: form.base_model,
      embeddings: form.embeddings,
      output_type: form.output_type,
      runtime_device: form.runtime_device,
      runtime_max_sequence_length: form.runtime_max_sequence_length,
      runtime_batch_size: form.runtime_batch_size,
      runtime_truncation: form.runtime_truncation,
      runtime_padding:
        form.runtime_padding_mode === "max_length"
          ? "max_length"
          : form.runtime_padding_mode === "false"
            ? false
            : true,
      ui_display_name: form.ui_display_name,
      color_token: form.color_token,
      group: form.group,
      labels: form.labels,
      model_config: parsedModelConfig,
    };

    formData.append("metadata", JSON.stringify(payload));
    for (const requirement of requirements) {
      for (const file of artifactFiles[requirement.slot] ?? []) {
        formData.append("artifact_files", file, `${requirement.slot}/${file.name}`);
      }
    }

    for (const file of dashboardFiles) {
      formData.append("dashboard_files", file, normalizeDashboardRelativePath(file));
    }

    setIsSubmitting(true);
    try {
      const snapshot = await uploadModelRequest(formData);
      onSuccess(snapshot, payload.model_id);
      closeWizard();
    } catch (error) {
      setServerError(error instanceof Error ? error.message : "Upload failed.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <section className="model-upload">
      <div className="model-upload__header">
        <div>
          <div className="panel__eyebrow">Local Upload</div>
          <h3>Register a new model</h3>
          <p>Capture metadata first, then attach the exact artifacts required by the model type.</p>
        </div>

        <div className="model-upload__steps">
          <span className={`model-upload__step${step === "metadata" ? " model-upload__step--active" : ""}`}>
            01 Metadata
          </span>
          <span className={`model-upload__step${step === "artifacts" ? " model-upload__step--active" : ""}`}>
            02 Artifacts
          </span>
        </div>
      </div>

      {serverError ? <div className="inline-alert">{serverError}</div> : null}

      {step === "metadata" ? (
        <div className="model-upload__grid">
          <label className={`field-shell${fieldErrors.model_id ? " field-shell--error" : ""}`}>
            <span>Model id</span>
            <input
              value={form.model_id}
              onChange={(event) => setFormField("model_id", event.target.value)}
              placeholder="age-roberta-500k"
            />
            <small>{fieldErrors.model_id ?? "Unique identifier used by the registry and dashboard routes."}</small>
          </label>

          <label className={`field-shell${fieldErrors.domain ? " field-shell--error" : ""}`}>
            <span>Domain</span>
            <input
              value={form.domain}
              onChange={(event) => setFormField("domain", event.target.value)}
              placeholder="age"
            />
            <small>{fieldErrors.domain ?? "Canonical domain slug. Home sync is based on this domain."}</small>
          </label>

          <label className={`field-shell${fieldErrors.display_name ? " field-shell--error" : ""}`}>
            <span>Display name</span>
            <input
              value={form.display_name}
              onChange={(event) => setFormField("display_name", event.target.value)}
              placeholder="RoBERTa 500k"
            />
            <small>{fieldErrors.display_name ?? "Primary model label shown in Models and Home."}</small>
          </label>

          <label className={`field-shell${fieldErrors.ui_display_name ? " field-shell--error" : ""}`}>
            <span>Domain display name</span>
            <input
              value={form.ui_display_name ?? ""}
              onChange={(event) => setFormField("ui_display_name", event.target.value)}
              placeholder="Age"
            />
            <small>{fieldErrors.ui_display_name ?? "Friendly domain title shown in cards and navigation copy."}</small>
          </label>

          <label className="field-shell field-shell--wide">
            <span>Description</span>
            <textarea
              value={form.description ?? ""}
              onChange={(event) => setFormField("description", event.target.value)}
              placeholder="Short model description"
            />
            <small>Optional technical description for the management surface.</small>
          </label>

          <label className="field-shell">
            <span>Version</span>
            <input
              value={form.version ?? ""}
              onChange={(event) => setFormField("version", event.target.value)}
              placeholder="v1"
            />
            <small>Optional runtime/version tag.</small>
          </label>

          <label className="field-shell">
            <span>Framework type</span>
            <select
              value={form.framework_type}
              onChange={(event) =>
                setForm((current) => ({
                  ...current,
                  framework_type: event.target.value as FrameworkType,
                  framework_library:
                    event.target.value === "transformers"
                      ? "huggingface"
                      : event.target.value === "sklearn"
                        ? "sklearn"
                        : current.framework_library,
                }))
              }
            >
              <option value="transformers">Transformer</option>
              <option value="pytorch">Deep learning / PyTorch</option>
              <option value="sklearn">Classical ML / sklearn</option>
            </select>
            <small>Artifact validation changes immediately with this selection.</small>
          </label>

          <label className="field-shell">
            <span>Framework library</span>
            <input
              value={form.framework_library ?? ""}
              onChange={(event) => setFormField("framework_library", event.target.value)}
              placeholder={form.framework_type === "transformers" ? "huggingface" : "torch"}
            />
            <small>Optional but useful for status and future runtime extensions.</small>
          </label>

          <label className="field-shell">
            <span>Backbone</span>
            <input
              value={form.backbone ?? ""}
              onChange={(event) => setFormField("backbone", event.target.value)}
              placeholder="distilbert-base-uncased"
            />
            <small>Recommended for transformer exports.</small>
          </label>

          <label className="field-shell">
            <span>Architecture</span>
            <input
              value={form.architecture ?? ""}
              onChange={(event) => setFormField("architecture", event.target.value)}
              placeholder={form.framework_type === "pytorch" ? "bilstm-attention" : "DistilBertForSequenceClassification"}
            />
            <small>Used to decide runtime compatibility.</small>
          </label>

          <label className="field-shell">
            <span>Max sequence length</span>
            <input
              type="number"
              min={1}
              value={form.runtime_max_sequence_length}
              onChange={(event) =>
                setFormField("runtime_max_sequence_length", Number(event.target.value))
              }
            />
            <small>{fieldErrors.runtime_max_sequence_length ?? "Runtime truncation length."}</small>
          </label>

          <label className="field-shell">
            <span>Batch size</span>
            <input
              type="number"
              min={1}
              value={form.runtime_batch_size}
              onChange={(event) => setFormField("runtime_batch_size", Number(event.target.value))}
            />
            <small>{fieldErrors.runtime_batch_size ?? "Batch size stored in the local manifest."}</small>
          </label>

          <label className="field-shell">
            <span>Runtime device</span>
            <select
              value={form.runtime_device}
              onChange={(event) => setFormField("runtime_device", event.target.value)}
            >
              <option value="auto">Auto</option>
              <option value="cpu">CPU</option>
              <option value="cuda">CUDA</option>
              <option value="mps">MPS</option>
            </select>
            <small>Matches the current manifest structure used by local production models.</small>
          </label>

          <label className="field-shell">
            <span>Padding</span>
            <select
              value={form.runtime_padding_mode}
              onChange={(event) => setFormField("runtime_padding_mode", event.target.value)}
            >
              <option value="true">True</option>
              <option value="false">False</option>
              <option value="max_length">max_length</option>
            </select>
            <small>Stored exactly in `model-config.yaml` runtime settings.</small>
          </label>

          <label className={`field-shell field-shell--wide${fieldErrors.model_config ? " field-shell--error" : ""}`}>
            <span>Advanced model config (JSON)</span>
            <textarea
              value={form.model_config_text}
              onChange={(event) => setFormField("model_config_text", event.target.value)}
              placeholder='{"hidden_dim": 128, "num_layers": 2}'
            />
            <small>{fieldErrors.model_config ?? "Optional architecture-specific config stored under the `model` block."}</small>
          </label>

          <div className={`field-shell field-shell--wide${fieldErrors.labels ? " field-shell--error" : ""}`}>
            <span>Labels</span>
            <div className="label-editor">
              {form.labels.map((label, index) => (
                <div key={`${label.id}-${index}`} className="label-editor__row">
                  <input
                    type="number"
                    value={label.id}
                    onChange={(event) => handleLabelChange(index, "id", Number(event.target.value))}
                  />
                  <input
                    value={label.name}
                    onChange={(event) => handleLabelChange(index, "name", event.target.value)}
                    placeholder="label_name"
                  />
                  <input
                    value={label.display_name ?? ""}
                    onChange={(event) =>
                      handleLabelChange(index, "display_name", event.target.value)
                    }
                    placeholder="Display label"
                  />
                  <button
                    type="button"
                    className="mini-button"
                    onClick={() =>
                      setForm((current) => ({
                        ...current,
                        labels: current.labels.filter((_, labelIndex) => labelIndex !== index),
                      }))
                    }
                    disabled={form.labels.length === 1}
                  >
                    Remove
                  </button>
                </div>
              ))}
            </div>
            <div className="label-editor__actions">
              <button
                type="button"
                className="mini-button"
                onClick={() =>
                  setForm((current) => ({
                    ...current,
                    labels: [
                      ...current.labels,
                      {
                        id: current.labels.length,
                        name: `class_${current.labels.length}`,
                        display_name: `Class ${current.labels.length}`,
                      },
                    ],
                  }))
                }
              >
                Add label
              </button>
              <span>{fieldErrors.labels ?? "Keep ids unique and aligned with the model export."}</span>
            </div>
          </div>

          <label className="toggle-field field-shell--wide">
            <input
              type="checkbox"
              checked={form.enable_on_upload}
              onChange={(event) => setFormField("enable_on_upload", event.target.checked)}
            />
            <span>Enable immediately after upload</span>
            <small>
              If the model is compatible and artifact-complete, it will become the active model for
              its domain and Home will update immediately.
            </small>
          </label>
        </div>
      ) : (
        <div className="model-upload__artifact-stage">
          <div className="model-upload__artifact-grid">
            {requirements.map((requirement) => {
              const files = artifactFiles[requirement.slot] ?? [];
              return (
                <label
                  key={requirement.slot}
                  className={`artifact-slot${artifactErrors[requirement.slot] ? " artifact-slot--error" : ""}`}
                >
                  <div className="artifact-slot__header">
                    <strong>{requirement.title}</strong>
                    <span>{requirement.required ? "Required" : "Optional"}</span>
                  </div>
                  <p>{requirement.hint}</p>
                  <input
                    type="file"
                    multiple
                    accept={requirement.accepts}
                    onChange={(event) =>
                      setArtifactFiles((current) => ({
                        ...current,
                        [requirement.slot]: Array.from(event.target.files ?? []),
                      }))
                    }
                  />
                  <div className="artifact-slot__list">
                    {files.length ? files.map((file) => <span key={file.name}>{file.name}</span>) : <span>No files selected.</span>}
                  </div>
                  <small>{artifactErrors[requirement.slot] ?? `Accepts: ${requirement.accepts}`}</small>
                </label>
              );
            })}
          </div>

          <label className={`artifact-slot artifact-slot--wide${dashboardError ? " artifact-slot--error" : ""}`}>
            <div className="artifact-slot__header">
              <strong>Optional dashboard folder</strong>
              <span>Optional</span>
            </div>
            <p>
              Select the prepared dashboard directory if you want this model to expose evaluation
              views immediately. If skipped, the model still uploads normally.
            </p>
            <input
              ref={dashboardInputRef}
              type="file"
              multiple
              onChange={(event) => setDashboardFiles(Array.from(event.target.files ?? []))}
            />
            <div className="artifact-slot__list">
              {dashboardFiles.length ? (
                dashboardFiles.slice(0, 6).map((file) => (
                  <span key={normalizeDashboardRelativePath(file)}>{normalizeDashboardRelativePath(file)}</span>
                ))
              ) : (
                <span>No dashboard folder selected.</span>
              )}
            </div>
            <small>{dashboardError ?? "Manifest and referenced files are validated before upload."}</small>
          </label>

          <div className="upload-review">
            <div className="upload-review__item">
              <span>Model</span>
              <strong>{form.display_name || "Untitled model"}</strong>
            </div>
            <div className="upload-review__item">
              <span>Domain</span>
              <strong>{form.domain || "—"}</strong>
            </div>
            <div className="upload-review__item">
              <span>Runtime</span>
              <strong>{form.framework_type}</strong>
            </div>
            <div className="upload-review__item">
              <span>Artifacts</span>
              <strong>
                {requirements.reduce((sum, requirement) => sum + (artifactFiles[requirement.slot]?.length ?? 0), 0)}
              </strong>
            </div>
          </div>
        </div>
      )}

      <div className="model-upload__actions">
        <button className="secondary-button" type="button" onClick={closeWizard}>
          Cancel
        </button>

        {step === "metadata" ? (
          <button
            className="primary-button"
            type="button"
            onClick={() => {
              if (validateMetadataStep()) {
                setStep("artifacts");
              }
            }}
          >
            Continue to artifacts
          </button>
        ) : (
          <>
            <button className="secondary-button" type="button" onClick={() => setStep("metadata")}>
              Back to metadata
            </button>
            <button
              className="primary-button"
              type="button"
              disabled={isSubmitting}
              onClick={() => void handleSubmit()}
            >
              {isSubmitting ? "Uploading..." : "Upload model"}
            </button>
          </>
        )}
      </div>
    </section>
  );
}
