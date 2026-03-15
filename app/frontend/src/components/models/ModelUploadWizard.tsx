import { useEffect, useId, useMemo, useReducer, useRef } from "react";
import huggingFaceLogo from "../../assets/huggingface-logo.svg";
import {
  ApiRequestError,
  importHuggingFaceModel,
  importLocalModel,
  preflightHuggingFaceImport,
  preflightLocalUpload,
} from "../../services/api";
import type {
  CatalogSnapshotResponse,
  HuggingFacePreflightRequest,
  HuggingFacePreflightResponse,
  LocalUploadPreflightResponse,
  ModelRegistrationResponse,
  UploadLabelClass,
} from "../../types/contracts";
import {
  ARTIFACT_REQUIREMENTS,
  branchHeading,
  buildFileDescriptor,
  buildLocalPreflightPayload,
  buildSuggestedModelId,
  canonicalizeSlug,
  createDefaultMetadataDraft,
  createNextLabel,
  exampleConfigTemplate,
  formatBytes,
  frameworkLabel,
  frameworkLibraryDefault,
  fromUploadMetadata,
  normalizeDashboardRelativePath,
  parseAcceptedExtensions,
  resolveBranchMode,
  summarizeDashboardReference,
  toUploadMetadata,
  type BranchMode,
  type DomainChoice,
  type FrameworkType,
  type UploadSource,
  type WizardMetadataDraft,
  type WizardStep,
} from "./modelUploadWizardSchema";

type PreflightResult = LocalUploadPreflightResponse | HuggingFacePreflightResponse;

interface WizardState {
  step: WizardStep;
  source: UploadSource | null;
  metadata: WizardMetadataDraft;
  domainMode: "existing" | "new";
  existingDomain: string;
  newDomainDisplayName: string;
  newDomainColorToken: string;
  newDomainGroup: string;
  artifactFiles: Record<string, File[]>;
  dashboardFiles: File[];
  registrationConfigFiles: File[];
  hfRepoInput: string;
  fieldErrors: Record<string, string>;
  message: string | null;
  preflight: PreflightResult | null;
  submissionStatus: "idle" | "running" | "error" | "success";
  result: ModelRegistrationResponse | null;
}

type WizardAction =
  | { type: "reset"; existingDomain: string }
  | { type: "set-step"; step: WizardStep }
  | { type: "set-source"; source: UploadSource }
  | { type: "set-domain-mode"; mode: "existing" | "new" }
  | { type: "set-existing-domain"; domain: string }
  | { type: "set-new-domain"; key: "display_name" | "color_token" | "group"; value: string }
  | { type: "set-metadata"; key: keyof WizardMetadataDraft; value: WizardMetadataDraft[keyof WizardMetadataDraft] }
  | { type: "patch-metadata"; patch: Partial<WizardMetadataDraft> }
  | { type: "set-artifact-files"; slot: string; files: File[] }
  | { type: "set-dashboard-files"; files: File[] }
  | { type: "set-registration-config-files"; files: File[] }
  | { type: "set-hf-repo-input"; value: string }
  | { type: "update-label"; index: number; patch: Partial<UploadLabelClass> }
  | { type: "add-label" }
  | { type: "remove-label"; index: number }
  | { type: "set-errors"; errors: Record<string, string>; message?: string | null }
  | { type: "clear-errors" }
  | { type: "set-preflight"; preflight: PreflightResult }
  | { type: "apply-normalized-metadata"; metadata: WizardMetadataDraft }
  | { type: "start-submit" }
  | { type: "submit-error"; message: string; errors?: Record<string, string> }
  | { type: "submit-success"; result: ModelRegistrationResponse };

const STEP_DEFS: Array<{ id: WizardStep; label: string }> = [
  { id: "source", label: "Source" },
  { id: "details", label: "Questions" },
  { id: "validate", label: "Validate" },
  { id: "review", label: "Review" },
  { id: "progress", label: "Progress" },
  { id: "result", label: "Result" },
];

function LocalSourceIcon() {
  return (
    <svg
      viewBox="0 0 48 48"
      className="model-upload-sheet__choice-icon"
      aria-hidden="true"
    >
      <rect x="9" y="8" width="30" height="20" rx="5" />
      <path d="M17 36h14" />
      <path d="M22 28v8" />
      <path d="M18 16h12" />
      <path d="M12 20h24" />
      <path d="M12 33.5h10.5l3-5.5h10.5" />
    </svg>
  );
}

function createInitialState(existingDomain: string): WizardState {
  return {
    step: "source",
    source: null,
    metadata: createDefaultMetadataDraft(),
    domainMode: "existing",
    existingDomain,
    newDomainDisplayName: "",
    newDomainColorToken: "",
    newDomainGroup: "",
    artifactFiles: {},
    dashboardFiles: [],
    registrationConfigFiles: [],
    hfRepoInput: "",
    fieldErrors: {},
    message: null,
    preflight: null,
    submissionStatus: "idle",
    result: null,
  };
}

function clearExecutionState(state: WizardState): WizardState {
  return {
    ...state,
    fieldErrors: {},
    message: null,
    preflight: null,
    submissionStatus: "idle",
    result: null,
    step:
      state.step === "review" || state.step === "progress" || state.step === "result"
        ? "details"
        : state.step,
  };
}

function wizardReducer(state: WizardState, action: WizardAction): WizardState {
  switch (action.type) {
    case "reset":
      return createInitialState(action.existingDomain);
    case "set-step":
      return { ...state, step: action.step, fieldErrors: {}, message: null };
    case "set-source":
      return {
        ...clearExecutionState(state),
        source: action.source,
        step: "source",
      };
    case "set-domain-mode":
      return {
        ...clearExecutionState(state),
        domainMode: action.mode,
      };
    case "set-existing-domain":
      return {
        ...clearExecutionState(state),
        existingDomain: action.domain,
      };
    case "set-new-domain":
      return {
        ...clearExecutionState(state),
        [action.key === "display_name"
          ? "newDomainDisplayName"
          : action.key === "color_token"
            ? "newDomainColorToken"
            : "newDomainGroup"]: action.value,
      } as WizardState;
    case "set-metadata": {
      const nextState = clearExecutionState(state);
      const nextMetadata = {
        ...nextState.metadata,
        [action.key]: action.value,
      } as WizardMetadataDraft;
      if (action.key === "framework_type") {
        nextMetadata.framework_library = frameworkLibraryDefault(action.value as FrameworkType);
      }
      return {
        ...nextState,
        metadata: nextMetadata,
      };
    }
    case "patch-metadata":
      return {
        ...clearExecutionState(state),
        metadata: {
          ...state.metadata,
          ...action.patch,
        },
      };
    case "set-artifact-files":
      return {
        ...clearExecutionState(state),
        artifactFiles: {
          ...state.artifactFiles,
          [action.slot]: action.files,
        },
      };
    case "set-dashboard-files":
      return {
        ...clearExecutionState(state),
        dashboardFiles: action.files,
      };
    case "set-registration-config-files":
      return {
        ...clearExecutionState(state),
        registrationConfigFiles: action.files,
      };
    case "set-hf-repo-input":
      return {
        ...clearExecutionState(state),
        hfRepoInput: action.value,
      };
    case "update-label":
      return {
        ...clearExecutionState(state),
        metadata: {
          ...state.metadata,
          labels: state.metadata.labels.map((label, index) =>
            index === action.index ? { ...label, ...action.patch } : label,
          ),
        },
      };
    case "add-label":
      return {
        ...clearExecutionState(state),
        metadata: {
          ...state.metadata,
          labels: [...state.metadata.labels, createNextLabel(state.metadata.labels)],
        },
      };
    case "remove-label":
      return {
        ...clearExecutionState(state),
        metadata: {
          ...state.metadata,
          labels: state.metadata.labels.filter((_, index) => index !== action.index),
        },
      };
    case "set-errors":
      return {
        ...state,
        fieldErrors: action.errors,
        message: action.message ?? null,
      };
    case "clear-errors":
      return {
        ...state,
        fieldErrors: {},
        message: null,
      };
    case "set-preflight":
      return {
        ...state,
        preflight: action.preflight,
        fieldErrors: {},
        message: null,
      };
    case "apply-normalized-metadata":
      return {
        ...state,
        metadata: action.metadata,
        newDomainDisplayName:
          state.domainMode === "new"
            ? action.metadata.ui_display_name ?? state.newDomainDisplayName
            : state.newDomainDisplayName,
        newDomainColorToken:
          state.domainMode === "new"
            ? action.metadata.color_token ?? state.newDomainColorToken
            : state.newDomainColorToken,
        newDomainGroup:
          state.domainMode === "new"
            ? action.metadata.group ?? state.newDomainGroup
            : state.newDomainGroup,
      };
    case "start-submit":
      return {
        ...state,
        step: "progress",
        submissionStatus: "running",
        fieldErrors: {},
        message: null,
      };
    case "submit-error":
      return {
        ...state,
        step: "progress",
        submissionStatus: "error",
        fieldErrors: action.errors ?? {},
        message: action.message,
      };
    case "submit-success":
      return {
        ...state,
        step: "result",
        submissionStatus: "success",
        fieldErrors: {},
        message: null,
        result: action.result,
      };
    default:
      return state;
  }
}

interface ModelUploadWizardProps {
  isOpen: boolean;
  domains: DomainChoice[];
  onClose: () => void;
  onSuccess: (snapshot: CatalogSnapshotResponse, modelId: string) => void;
}

function resolveDomainChoice(state: WizardState, domains: DomainChoice[]): DomainChoice | null {
  if (state.domainMode === "existing") {
    return domains.find((domain) => domain.domain === state.existingDomain) ?? domains[0] ?? null;
  }

  const displayName = state.newDomainDisplayName.trim();
  const slug = canonicalizeSlug(displayName);
  if (!displayName || !slug) {
    return null;
  }
  const colorToken = canonicalizeSlug(state.newDomainColorToken || displayName) || slug;
  const group = canonicalizeSlug(state.newDomainGroup || `${slug}-custom`) || `${slug}-custom`;
  return {
    domain: slug,
    display_name: displayName,
    color_token: colorToken,
    group,
  };
}

function fieldErrorFor(state: WizardState, ...keys: string[]) {
  return keys.map((key) => state.fieldErrors[key]).find(Boolean) ?? null;
}

function mapApiFieldErrors(error: unknown): Record<string, string> {
  if (!(error instanceof ApiRequestError) || !error.detail || typeof error.detail !== "object") {
    return {};
  }
  const raw = (error.detail as { field_errors?: Record<string, string> }).field_errors;
  if (!raw) {
    return {};
  }

  const mapped: Record<string, string> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (key === "metadata.model_id") {
      mapped.model_id = value;
    } else if (key === "metadata.framework_type") {
      mapped.framework_type = value;
    } else if (key === "huggingface.repo") {
      mapped.hf_repo = value;
    } else {
      mapped[key] = value;
    }
  }
  return mapped;
}

function validateDetailsStep(
  state: WizardState,
  domains: DomainChoice[],
): Record<string, string> {
  const errors: Record<string, string> = {};
  const domainChoice = resolveDomainChoice(state, domains);

  if (!state.metadata.display_name.trim()) {
    errors.display_name = "Display name is required.";
  }
  if (!state.metadata.model_id.trim()) {
    errors.model_id = "Model id is required.";
  } else if (state.metadata.model_id.trim().length < 2) {
    errors.model_id = "Model id must be at least 2 characters.";
  }
  if (state.domainMode === "existing") {
    if (!state.existingDomain) {
      errors.domain = "Choose an existing domain.";
    }
  } else {
    if (!state.newDomainDisplayName.trim()) {
      errors.domain = "Enter a display name for the new domain.";
    } else if (canonicalizeSlug(state.newDomainDisplayName).length < 2) {
      errors.domain = "The new domain name needs at least 2 slug-safe characters.";
    }
  }
  if (!domainChoice) {
    errors.domain = errors.domain ?? "Choose or create a valid domain.";
  }
  if (!state.metadata.framework_task.trim()) {
    errors.framework_task = "Framework task is required.";
  }
  if (state.metadata.runtime_max_sequence_length < 1) {
    errors.runtime_max_sequence_length = "Sequence length must be at least 1.";
  }
  if (state.metadata.runtime_batch_size < 1) {
    errors.runtime_batch_size = "Batch size must be at least 1.";
  }
  if (state.metadata.labels.length === 0) {
    errors.labels = "At least one label is required.";
  }

  const seenIds = new Set<number>();
  const seenNames = new Set<string>();
  state.metadata.labels.forEach((label, index) => {
    if (!label.name.trim()) {
      errors[`label-name-${index}`] = "Label name is required.";
    }
    if (seenIds.has(label.id)) {
      errors[`label-id-${index}`] = "Label ids must be unique.";
    }
    if (seenNames.has(label.name.trim().toLowerCase())) {
      errors[`label-name-${index}`] = "Label names must be unique.";
    }
    seenIds.add(label.id);
    seenNames.add(label.name.trim().toLowerCase());
  });

  if (state.metadata.model_config_text.trim()) {
    try {
      const parsed = JSON.parse(state.metadata.model_config_text);
      if (parsed === null || Array.isArray(parsed) || typeof parsed !== "object") {
        errors.model_config_text = "Model config must be a JSON object.";
      }
    } catch {
      errors.model_config_text = "Model config JSON is invalid.";
    }
  }

  return errors;
}

async function validateLocalFilesStep(
  state: WizardState,
  frameworkType: FrameworkType,
): Promise<Record<string, string>> {
  const errors: Record<string, string> = {};
  const requirements = ARTIFACT_REQUIREMENTS[frameworkType];
  for (const requirement of requirements) {
    const files = state.artifactFiles[requirement.slot] ?? [];
    if (requirement.required && files.length === 0) {
      errors[`artifacts.${requirement.slot}`] = `${requirement.title} is required.`;
      continue;
    }

    const allowedExtensions = parseAcceptedExtensions(requirement.accepts);
    const invalidFile = files.find((file) => {
      const extension = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
      return !allowedExtensions.includes(extension);
    });
    if (invalidFile) {
      errors[`artifacts.${requirement.slot}`] = `${invalidFile.name} has an unsupported file type.`;
    }
  }

  if (state.dashboardFiles.length > 0) {
    const manifestFile = state.dashboardFiles.find((file) =>
      normalizeDashboardRelativePath(file).endsWith("dashboard-manifest.json"),
    );
    if (!manifestFile) {
      errors.dashboard = "Dashboard folder must include dashboard-manifest.json.";
    } else {
      try {
        const manifest = JSON.parse(await manifestFile.text()) as {
          entrypoints?: Record<string, string>;
          sections?: Array<{ files?: string[] }>;
        };
        const relativePaths = new Set(state.dashboardFiles.map(normalizeDashboardRelativePath));
        const references = [
          ...Object.values(manifest.entrypoints ?? {}),
          ...(manifest.sections ?? []).flatMap((section) => section.files ?? []),
        ];
        const missingRefs = references.filter((reference) => {
          const normalized = summarizeDashboardReference(reference);
          const referenceSegments = reference.split("/");
          const byName = referenceSegments[referenceSegments.length - 1] ?? reference;
          return !Array.from(relativePaths).some(
            (path) => path.endsWith(normalized) || path.endsWith(byName),
          );
        });
        if (missingRefs.length > 0) {
          errors.dashboard =
            `Dashboard manifest references missing files: ${missingRefs.slice(0, 3).join(", ")}${missingRefs.length > 3 ? "..." : ""}`;
        }
      } catch {
        errors.dashboard = "Dashboard manifest JSON is invalid.";
      }
    }
  }

  return errors;
}

function getCurrentMetadata(state: WizardState, domains: DomainChoice[]) {
  const domainChoice = resolveDomainChoice(state, domains);
  if (!domainChoice) {
    return null;
  }
  return toUploadMetadata(state.metadata, domainChoice);
}

function isLocalPreflight(
  value: PreflightResult | null,
): value is LocalUploadPreflightResponse {
  return Boolean(value && "artifact_checks" in value);
}

function isHuggingFacePreflight(
  value: PreflightResult | null,
): value is HuggingFacePreflightResponse {
  return Boolean(value && "ready_to_import" in value);
}

function progressStages(branch: BranchMode | null): string[] {
  if (branch === "huggingface") {
    return [
      "Validate the remote repo and confirm compatibility.",
      "Download the selected Hugging Face artifacts into local model storage.",
      "Generate the registry manifest and refresh the live catalog snapshot.",
    ];
  }
  return [
    "Re-validate the selected files and registration settings.",
    "Copy artifacts into local model storage and write the manifest.",
    "Refresh the live catalog snapshot so the model appears in management immediately.",
  ];
}

export function ModelUploadWizard({
  isOpen,
  domains,
  onClose,
  onSuccess,
}: ModelUploadWizardProps) {
  const [state, dispatch] = useReducer(
    wizardReducer,
    createInitialState(domains[0]?.domain ?? ""),
  );
  const fieldIdPrefix = useId();
  const stepOrder = useMemo(() => STEP_DEFS.map((step) => step.id), []);
  const branch = resolveBranchMode(state.source);
  const domainChoice = resolveDomainChoice(state, domains);
  const currentMetadata = useMemo(() => getCurrentMetadata(state, domains), [state, domains]);
  const currentFramework = state.metadata.framework_type;
  const validateControllerRef = useRef<AbortController | null>(null);
  const submitControllerRef = useRef<AbortController | null>(null);
  const fieldId = (suffix: string) => `${fieldIdPrefix}-${suffix}`;

  useEffect(() => {
    if (!isOpen) {
      return;
    }
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
      validateControllerRef.current?.abort();
      submitControllerRef.current?.abort();
    };
  }, [isOpen]);

  useEffect(() => {
    if (!state.existingDomain && domains[0]) {
      dispatch({ type: "set-existing-domain", domain: domains[0].domain });
    }
  }, [domains, state.existingDomain]);

  if (!isOpen) {
    return null;
  }

  const currentStepIndex = stepOrder.indexOf(state.step);
  const canClose = state.submissionStatus !== "running";
  const canGoToReview =
    (isLocalPreflight(state.preflight) && state.preflight.ready) ||
    (isHuggingFacePreflight(state.preflight) && state.preflight.ready_to_import);

  const closeWizard = () => {
    if (!canClose) {
      return;
    }
    onClose();
  };

  const handleSourceNext = () => {
    const errors: Record<string, string> = {};
    if (!state.source) {
      errors.source = "Choose where the model is coming from.";
    }
    if (Object.keys(errors).length > 0) {
      dispatch({
        type: "set-errors",
        errors,
        message: "Choose an upload source to continue.",
      });
      return;
    }
    dispatch({ type: "set-step", step: "details" });
  };

  const handleDetailsNext = () => {
    const errors = validateDetailsStep(state, domains);
    if (Object.keys(errors).length > 0) {
      dispatch({
        type: "set-errors",
        errors,
        message: "Resolve the highlighted fields before continuing.",
      });
      return;
    }
    dispatch({ type: "set-step", step: "validate" });
  };

  const handleRunValidation = async () => {
    if (!branch || !currentMetadata || !domainChoice) {
      dispatch({
        type: "set-errors",
        errors: { source: "Finish the earlier steps first." },
        message: "The wizard needs source and metadata details before validation can run.",
      });
      return;
    }

    if (branch === "huggingface") {
      const errors: Record<string, string> = {};
      if (!state.hfRepoInput.trim()) {
        errors.hf_repo = "Paste a Hugging Face model URL or repo id.";
      }
      if (Object.keys(errors).length > 0) {
        dispatch({
          type: "set-errors",
          errors,
          message: "A Hugging Face repo is required before preflight can run.",
        });
        return;
      }

      dispatch({ type: "set-errors", errors: {}, message: "Inspecting the remote repo..." });
      validateControllerRef.current?.abort();
      validateControllerRef.current = new AbortController();

      try {
        const response = await preflightHuggingFaceImport(
          {
            repo: state.hfRepoInput.trim(),
            metadata: currentMetadata,
          } satisfies HuggingFacePreflightRequest,
          validateControllerRef.current.signal,
        );
        dispatch({ type: "set-preflight", preflight: response });
        dispatch({
          type: "apply-normalized-metadata",
          metadata: fromUploadMetadata(response.normalized_metadata),
        });
      } catch (error) {
        dispatch({
          type: "set-errors",
          errors: mapApiFieldErrors(error),
          message: error instanceof Error ? error.message : "Hugging Face preflight failed.",
        });
      }
      return;
    }

    const localErrors = await validateLocalFilesStep(state, currentFramework);
    if (Object.keys(localErrors).length > 0) {
      dispatch({
        type: "set-errors",
        errors: localErrors,
        message: "Resolve the highlighted file issues before validation can run.",
      });
      return;
    }

    dispatch({ type: "set-errors", errors: {}, message: "Validating files and building a config preview..." });
    validateControllerRef.current?.abort();
    validateControllerRef.current = new AbortController();

    const formData = new FormData();
    formData.append(
      "payload",
      JSON.stringify(
        buildLocalPreflightPayload(
          state.metadata,
          domainChoice,
          state.artifactFiles,
          state.dashboardFiles,
        ),
      ),
    );
    for (const file of state.registrationConfigFiles) {
      formData.append("registration_config_files", file, file.name);
    }

    try {
      const response = await preflightLocalUpload(formData, validateControllerRef.current.signal);
      dispatch({ type: "set-preflight", preflight: response });
      dispatch({
        type: "apply-normalized-metadata",
        metadata: fromUploadMetadata(response.normalized_metadata),
      });
    } catch (error) {
      dispatch({
        type: "set-errors",
        errors: mapApiFieldErrors(error),
        message: error instanceof Error ? error.message : "Validation failed.",
      });
    }
  };

  const handleSubmit = async () => {
    if (!branch || !currentMetadata || !state.preflight) {
      dispatch({
        type: "set-errors",
        errors: {},
        message: "Run validation first so the wizard has a reviewed import plan.",
      });
      return;
    }

    submitControllerRef.current?.abort();
    submitControllerRef.current = new AbortController();
    dispatch({ type: "start-submit" });

    try {
      let response: ModelRegistrationResponse;
      if (branch === "huggingface") {
        response = await importHuggingFaceModel(
          {
            repo: state.hfRepoInput.trim(),
            metadata: currentMetadata,
          },
          submitControllerRef.current.signal,
        );
      } else {
        const formData = new FormData();
        formData.append(
          "payload",
          JSON.stringify(
            buildLocalPreflightPayload(
              state.metadata,
              domainChoice!,
              state.artifactFiles,
              state.dashboardFiles,
            ),
          ),
        );
        for (const [slot, files] of Object.entries(state.artifactFiles)) {
          for (const file of files) {
            formData.append("artifact_files", file, `${slot}/${file.name}`);
          }
        }
        for (const file of state.dashboardFiles) {
          formData.append("dashboard_files", file, normalizeDashboardRelativePath(file));
        }
        for (const file of state.registrationConfigFiles) {
          formData.append("registration_config_files", file, file.name);
        }
        response = await importLocalModel(formData, submitControllerRef.current.signal);
      }

      onSuccess(response.snapshot, response.result.model_id);
      dispatch({ type: "submit-success", result: response });
    } catch (error) {
      dispatch({
        type: "submit-error",
        message: error instanceof Error ? error.message : "Import failed.",
        errors: mapApiFieldErrors(error),
      });
    }
  };

  const renderSourceStep = () => (
    <div className="model-upload-sheet__content model-upload-sheet__content--source">
      <div className="model-upload-sheet__section model-upload-sheet__section--source">
        <h3 className="model-upload-sheet__source-question">Where is the model coming from?</h3>
        <div className="model-upload-sheet__choice-grid model-upload-sheet__choice-grid--source">
          <button
            type="button"
            className={`model-upload-sheet__choice model-upload-sheet__choice--source${state.source === "local" ? " model-upload-sheet__choice--active" : ""}`}
            onClick={() => dispatch({ type: "set-source", source: "local" })}
            data-testid="upload-source-local"
          >
            <strong>Local computer</strong>
            <span className="model-upload-sheet__choice-copy">
              Upload files from your machine and register them in the local model storage.
            </span>
            <span
              className="model-upload-sheet__choice-icon-shell model-upload-sheet__choice-icon-shell--local"
              aria-hidden="true"
            >
              <LocalSourceIcon />
            </span>
          </button>
          <button
            type="button"
            className={`model-upload-sheet__choice model-upload-sheet__choice--source${state.source === "huggingface" ? " model-upload-sheet__choice--active" : ""}`}
            onClick={() => dispatch({ type: "set-source", source: "huggingface" })}
            data-testid="upload-source-hf"
          >
            <strong>Hugging Face</strong>
            <span className="model-upload-sheet__choice-copy">
              Inspect a remote repo, run a compatibility preflight, and import it locally.
            </span>
            <span
              className="model-upload-sheet__choice-icon-shell model-upload-sheet__choice-icon-shell--huggingface"
              aria-hidden="true"
            >
              <img
                src={huggingFaceLogo}
                alt=""
                className="model-upload-sheet__choice-icon-image"
              />
            </span>
          </button>
        </div>
        {fieldErrorFor(state, "source") ? <small className="field-error">{fieldErrorFor(state, "source")}</small> : null}
      </div>
    </div>
  );

  const renderDomainSection = () => {
    const slugPreview = canonicalizeSlug(state.newDomainDisplayName);
    return (
      <div className="model-upload-sheet__section">
        <div className="panel__eyebrow">Domain</div>
        <div className="model-upload-sheet__toggle-row">
          <button
            type="button"
            className={`mini-button${state.domainMode === "existing" ? " mini-button--active" : ""}`}
            onClick={() => dispatch({ type: "set-domain-mode", mode: "existing" })}
          >
            Select existing domain
          </button>
          <button
            type="button"
            className={`mini-button${state.domainMode === "new" ? " mini-button--active" : ""}`}
            onClick={() => dispatch({ type: "set-domain-mode", mode: "new" })}
          >
            Create new domain
          </button>
        </div>

        {state.domainMode === "existing" ? (
          <label
            htmlFor={fieldId("existing-domain")}
            className={`field-shell${fieldErrorFor(state, "domain") ? " field-shell--error" : ""}`}
          >
            <span>Existing domain</span>
            <select
              id={fieldId("existing-domain")}
              value={state.existingDomain}
              onChange={(event) =>
                dispatch({ type: "set-existing-domain", domain: event.target.value })
              }
            >
              {domains.map((domain) => (
                <option key={domain.domain} value={domain.domain}>
                  {domain.display_name} ({domain.domain})
                </option>
              ))}
            </select>
            <small>{fieldErrorFor(state, "domain") ?? "Reuse an existing domain and keep its display color/grouping."}</small>
          </label>
        ) : (
          <div className="model-upload-sheet__grid">
            <label
              htmlFor={fieldId("new-domain-display-name")}
              className={`field-shell${fieldErrorFor(state, "domain") ? " field-shell--error" : ""}`}
            >
              <span>New domain display name</span>
              <input
                id={fieldId("new-domain-display-name")}
                value={state.newDomainDisplayName}
                onChange={(event) =>
                  dispatch({
                    type: "set-new-domain",
                    key: "display_name",
                    value: event.target.value,
                  })
                }
                placeholder="Cyberbullying Classification"
              />
              <small>{fieldErrorFor(state, "domain") ?? `Slug preview: ${slugPreview || "—"}`}</small>
            </label>

            <label htmlFor={fieldId("new-domain-color-token")} className="field-shell">
              <span>Color token</span>
              <input
                id={fieldId("new-domain-color-token")}
                value={state.newDomainColorToken}
                onChange={(event) =>
                  dispatch({
                    type: "set-new-domain",
                    key: "color_token",
                    value: event.target.value,
                  })
                }
                placeholder={slugPreview || "sentiment"}
              />
              <small>Optional. Defaults to the slug preview.</small>
            </label>

            <label htmlFor={fieldId("new-domain-group")} className="field-shell">
              <span>Group</span>
              <input
                id={fieldId("new-domain-group")}
                value={state.newDomainGroup}
                onChange={(event) =>
                  dispatch({
                    type: "set-new-domain",
                    key: "group",
                    value: event.target.value,
                  })
                }
                placeholder={slugPreview ? `${slugPreview}-custom` : "sentiment-custom"}
              />
              <small>Optional. Defaults to `{slugPreview || "domain"}-custom`.</small>
            </label>
          </div>
        )}
      </div>
    );
  };

  const renderMetadataSection = () => (
    <div className="model-upload-sheet__section">
      <div className="panel__eyebrow">Model Metadata</div>
      <div className="model-upload-sheet__grid">
        <label
          htmlFor={fieldId("display-name")}
          className={`field-shell${fieldErrorFor(state, "display_name") ? " field-shell--error" : ""}`}
        >
          <span>Display name</span>
          <input
            id={fieldId("display-name")}
            aria-label="Display name"
            value={state.metadata.display_name}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "display_name", value: event.target.value })
            }
            placeholder="DistilBERT 60k"
          />
          <small>{fieldErrorFor(state, "display_name") ?? "The primary name shown in management and dashboards."}</small>
        </label>

        <label
          htmlFor={fieldId("model-id")}
          className={`field-shell${fieldErrorFor(state, "model_id") ? " field-shell--error" : ""}`}
        >
          <span>Model id</span>
          <div className="model-upload-sheet__input-stack">
            <input
              id={fieldId("model-id")}
              aria-label="Model id"
              value={state.metadata.model_id}
              onChange={(event) =>
                dispatch({ type: "set-metadata", key: "model_id", value: event.target.value })
              }
              placeholder="sentiment-distilbert-60k"
            />
            <button
              type="button"
              className="mini-button"
              onClick={() =>
                dispatch({
                  type: "set-metadata",
                  key: "model_id",
                  value: buildSuggestedModelId(
                    domainChoice?.domain ?? state.newDomainDisplayName,
                    state.metadata.display_name,
                  ),
                })
              }
            >
              Suggest id
            </button>
          </div>
          <small>{fieldErrorFor(state, "model_id") ?? "Unique registry id used for routes, manifests, and storage."}</small>
        </label>

        <label htmlFor={fieldId("description")} className="field-shell field-shell--wide">
          <span>Description</span>
          <textarea
            id={fieldId("description")}
            value={state.metadata.description ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "description", value: event.target.value })
            }
            placeholder="Short technical description"
          />
          <small>Optional context for reviewers and future maintainers.</small>
        </label>

        <label htmlFor={fieldId("version")} className="field-shell">
          <span>Version</span>
          <input
            id={fieldId("version")}
            value={state.metadata.version ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "version", value: event.target.value })
            }
            placeholder="v1"
          />
          <small>Optional release or export tag.</small>
        </label>

        {branch !== "huggingface" ? (
          <label
            htmlFor={fieldId("framework-type")}
            className={`field-shell${fieldErrorFor(state, "framework_type") ? " field-shell--error" : ""}`}
          >
            <span>Model type</span>
            <select
              id={fieldId("framework-type")}
              value={state.metadata.framework_type}
              onChange={(event) =>
                dispatch({
                  type: "set-metadata",
                  key: "framework_type",
                  value: event.target.value as FrameworkType,
                })
              }
            >
              <option value="transformers">Transformer</option>
              <option value="pytorch">Deep Learning</option>
              <option value="sklearn">Classic ML</option>
            </select>
            <small>{fieldErrorFor(state, "framework_type") ?? "This controls the artifact requirements and runtime mapping."}</small>
          </label>
        ) : null}

        <label
          htmlFor={fieldId("framework-task")}
          className={`field-shell${fieldErrorFor(state, "framework_task") ? " field-shell--error" : ""}`}
        >
          <span>Framework task</span>
          <input
            id={fieldId("framework-task")}
            value={state.metadata.framework_task}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "framework_task", value: event.target.value })
            }
            placeholder="sequence-classification"
          />
          <small>{fieldErrorFor(state, "framework_task") ?? "Visible and editable because it affects runtime compatibility."}</small>
        </label>

        <label htmlFor={fieldId("framework-library")} className="field-shell">
          <span>Framework library</span>
          <input
            id={fieldId("framework-library")}
            value={state.metadata.framework_library ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "framework_library", value: event.target.value })
            }
            placeholder={frameworkLibraryDefault(state.metadata.framework_type)}
          />
          <small>Defaults to the current runtime library for the selected model type.</small>
        </label>

        <label htmlFor={fieldId("architecture")} className="field-shell">
          <span>Architecture</span>
          <input
            id={fieldId("architecture")}
            value={state.metadata.architecture ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "architecture", value: event.target.value })
            }
            placeholder="DistilBertForSequenceClassification"
          />
          <small>Important for runtime compatibility and review.</small>
        </label>

        <label htmlFor={fieldId("backbone")} className="field-shell">
          <span>Backbone</span>
          <input
            id={fieldId("backbone")}
            value={state.metadata.backbone ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "backbone", value: event.target.value })
            }
            placeholder="distilbert-base-uncased"
          />
          <small>Used when the model family matters for review and import tracking.</small>
        </label>

        <label htmlFor={fieldId("base-model")} className="field-shell">
          <span>Base model</span>
          <input
            id={fieldId("base-model")}
            value={state.metadata.base_model ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "base_model", value: event.target.value })
            }
            placeholder="bert-base-uncased"
          />
          <small>Optional but helpful when the export and the training backbone differ.</small>
        </label>

        <label htmlFor={fieldId("embeddings")} className="field-shell">
          <span>Embeddings</span>
          <input
            id={fieldId("embeddings")}
            value={state.metadata.embeddings ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "embeddings", value: event.target.value })
            }
            placeholder="fasttext-wiki-news-subwords-300"
          />
          <small>Mostly relevant for custom deep learning and classic ML flows.</small>
        </label>
      </div>
    </div>
  );

  const renderRuntimeSection = () => (
    <div className="model-upload-sheet__section">
      <div className="panel__eyebrow">Runtime & Labels</div>
      <div className="model-upload-sheet__grid">
        <label
          htmlFor={fieldId("runtime-max-sequence-length")}
          className={`field-shell${fieldErrorFor(state, "runtime_max_sequence_length") ? " field-shell--error" : ""}`}
        >
          <span>Max sequence length</span>
          <input
            id={fieldId("runtime-max-sequence-length")}
            type="number"
            min={1}
            value={state.metadata.runtime_max_sequence_length}
            onChange={(event) =>
              dispatch({
                type: "set-metadata",
                key: "runtime_max_sequence_length",
                value: Number(event.target.value),
              })
            }
          />
          <small>{fieldErrorFor(state, "runtime_max_sequence_length") ?? "Stored in the saved runtime block."}</small>
        </label>

        <label
          htmlFor={fieldId("runtime-batch-size")}
          className={`field-shell${fieldErrorFor(state, "runtime_batch_size") ? " field-shell--error" : ""}`}
        >
          <span>Batch size</span>
          <input
            id={fieldId("runtime-batch-size")}
            type="number"
            min={1}
            value={state.metadata.runtime_batch_size}
            onChange={(event) =>
              dispatch({
                type: "set-metadata",
                key: "runtime_batch_size",
                value: Number(event.target.value),
              })
            }
          />
          <small>{fieldErrorFor(state, "runtime_batch_size") ?? "Visible because it materially changes runtime behavior."}</small>
        </label>

        <label htmlFor={fieldId("runtime-device")} className="field-shell">
          <span>Runtime device</span>
          <select
            id={fieldId("runtime-device")}
            value={state.metadata.runtime_device}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "runtime_device", value: event.target.value })
            }
          >
            <option value="auto">Auto</option>
            <option value="cpu">CPU</option>
            <option value="cuda">CUDA</option>
            <option value="mps">MPS</option>
          </select>
          <small>Matches the existing manifest structure used in the registry.</small>
        </label>

        <label htmlFor={fieldId("runtime-padding")} className="field-shell">
          <span>Padding</span>
          <select
            id={fieldId("runtime-padding")}
            value={state.metadata.runtime_padding_mode}
            onChange={(event) =>
              dispatch({
                type: "set-metadata",
                key: "runtime_padding_mode",
                value: event.target.value as WizardMetadataDraft["runtime_padding_mode"],
              })
            }
          >
            <option value="true">true</option>
            <option value="false">false</option>
            <option value="max_length">max_length</option>
          </select>
          <small>The saved manifest keeps this exact runtime padding mode.</small>
        </label>

        <label className="toggle-field field-shell">
          <input
            type="checkbox"
            checked={state.metadata.runtime_truncation}
            onChange={(event) =>
              dispatch({
                type: "set-metadata",
                key: "runtime_truncation",
                value: event.target.checked,
              })
            }
          />
          <span>Enable truncation</span>
          <small>Stored in the runtime block and shown in review.</small>
        </label>

        <label className="toggle-field field-shell">
          <input
            type="checkbox"
            checked={state.metadata.enable_on_upload}
            onChange={(event) =>
              dispatch({
                type: "set-metadata",
                key: "enable_on_upload",
                value: event.target.checked,
              })
            }
          />
          <span>Enable immediately after registration</span>
          <small>Activation is still blocked if runtime compatibility checks fail.</small>
        </label>

        <label htmlFor={fieldId("runtime-preprocessing")} className="field-shell field-shell--wide">
          <span>Preprocessing</span>
          <input
            id={fieldId("runtime-preprocessing")}
            value={state.metadata.runtime_preprocessing ?? ""}
            onChange={(event) =>
              dispatch({
                type: "set-metadata",
                key: "runtime_preprocessing",
                value: event.target.value,
              })
            }
            placeholder="normalize_text + texts_to_sequences"
          />
          <small>Especially useful for custom PyTorch and classic ML registrations.</small>
        </label>

        <label
          htmlFor={fieldId("model-config-text")}
          className={`field-shell field-shell--wide${fieldErrorFor(state, "model_config_text") ? " field-shell--error" : ""}`}
        >
          <span>Model config payload (JSON)</span>
          <textarea
            id={fieldId("model-config-text")}
            value={state.metadata.model_config_text}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "model_config_text", value: event.target.value })
            }
            placeholder='{"hidden_dim": 128, "num_layers": 2}'
          />
          <small>{fieldErrorFor(state, "model_config_text") ?? "This is stored under the `model` block in the saved manifest."}</small>
        </label>

        <div className={`field-shell field-shell--wide${fieldErrorFor(state, "labels") ? " field-shell--error" : ""}`}>
          <span>Labels</span>
          <div className="label-editor">
            {state.metadata.labels.map((label, index) => (
              <div key={`${label.id}-${index}`} className="label-editor__row">
                <div className="model-upload-sheet__input-stack">
                  <input
                    type="number"
                    value={label.id}
                    onChange={(event) =>
                      dispatch({
                        type: "update-label",
                        index,
                        patch: { id: Number(event.target.value) },
                      })
                    }
                  />
                  {fieldErrorFor(state, `label-id-${index}`) ? (
                    <small className="field-error">{fieldErrorFor(state, `label-id-${index}`)}</small>
                  ) : null}
                </div>
                <div className="model-upload-sheet__input-stack">
                  <input
                    value={label.name}
                    onChange={(event) =>
                      dispatch({
                        type: "update-label",
                        index,
                        patch: { name: event.target.value },
                      })
                    }
                    placeholder="label_name"
                  />
                  {fieldErrorFor(state, `label-name-${index}`) ? (
                    <small className="field-error">{fieldErrorFor(state, `label-name-${index}`)}</small>
                  ) : null}
                </div>
                <input
                  value={label.display_name ?? ""}
                  onChange={(event) =>
                    dispatch({
                      type: "update-label",
                      index,
                      patch: { display_name: event.target.value },
                    })
                  }
                  placeholder="Display label"
                />
                <button
                  type="button"
                  className="mini-button"
                  onClick={() => dispatch({ type: "remove-label", index })}
                  disabled={state.metadata.labels.length === 1}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
          <div className="label-editor__actions">
            <button type="button" className="mini-button" onClick={() => dispatch({ type: "add-label" })}>
              Add label
            </button>
            <span>{fieldErrorFor(state, "labels") ?? "HF preflight can replace the placeholder labels when remote metadata exposes a label map."}</span>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDetailsStep = () => (
    <div className="model-upload-sheet__content">
      <div className="model-upload-sheet__intro-card">
        <div className="panel__eyebrow">Step 2</div>
        <h3>{branchHeading(branch)}</h3>
        <p>
          {branch === "local"
            ? "Set the local registry metadata you want first. On the next step you can optionally upload an existing registration config, or let the system generate the final saved manifest from your answers and artifacts."
            : "Set the registry metadata you want before we inspect the remote repo and generate the final manifest."}
        </p>
      </div>

      {renderDomainSection()}
      {renderMetadataSection()}
      {renderRuntimeSection()}
    </div>
  );

  const renderLocalValidateStep = () => {
    const requirements = ARTIFACT_REQUIREMENTS[currentFramework];
    return (
      <div className="model-upload-sheet__content">
        <div className="model-upload-sheet__section">
          <div className="panel__eyebrow">Registration Config</div>
          <div className="model-upload-sheet__section-header">
            <div>
              <h3>Optional existing registration config</h3>
              <p>
                Upload a saved manifest if you already have one and we’ll validate and normalize
                it. If you leave this empty, the system will generate the final saved manifest from
                your answers and uploaded artifacts.
              </p>
            </div>
          </div>

          <label
            htmlFor={fieldId("registration-config")}
            className={`field-shell${fieldErrorFor(state, "registration_config") ? " field-shell--error" : ""}`}
          >
            <span>Registration config file</span>
            <input
              id={fieldId("registration-config")}
              aria-label="Registration config file"
              type="file"
              accept=".yaml,.yml,.json"
              multiple={false}
              onChange={(event) =>
                dispatch({
                  type: "set-registration-config-files",
                  files: Array.from(event.target.files ?? []),
                })
              }
            />
            <small>
              {fieldErrorFor(state, "registration_config")
                ?? "Optional. Upload the main saved manifest you already have (YAML or JSON), or skip this and we’ll generate one."}
            </small>
            <div className="artifact-slot__list">
              {state.registrationConfigFiles.length > 0 ? (
                state.registrationConfigFiles.map((file) => <span key={file.name}>{file.name}</span>)
              ) : (
                <span>No config file selected.</span>
              )}
            </div>
          </label>

          <details className="model-upload-sheet__example-config" open>
            <summary>Example config</summary>
            <pre>{exampleConfigTemplate(currentFramework)}</pre>
          </details>
        </div>

        <div className="model-upload-sheet__section">
          <div className="panel__eyebrow">Artifacts</div>
          <div className="model-upload-sheet__grid">
            {requirements.map((requirement) => {
              const files = state.artifactFiles[requirement.slot] ?? [];
              return (
                <label
                  key={requirement.slot}
                  htmlFor={fieldId(`artifact-${requirement.slot}`)}
                  className={`artifact-slot${fieldErrorFor(state, `artifacts.${requirement.slot}`) ? " artifact-slot--error" : ""}`}
                >
                  <div className="artifact-slot__header">
                    <strong>{requirement.title}</strong>
                    <span>{requirement.required ? "Required" : "Optional"}</span>
                  </div>
                  <p>{requirement.hint}</p>
                  <input
                    id={fieldId(`artifact-${requirement.slot}`)}
                    aria-label={requirement.title}
                    type="file"
                    multiple
                    accept={requirement.accepts}
                    onChange={(event) =>
                      dispatch({
                        type: "set-artifact-files",
                        slot: requirement.slot,
                        files: Array.from(event.target.files ?? []),
                      })
                    }
                  />
                  <small>{fieldErrorFor(state, `artifacts.${requirement.slot}`) ?? "Selected files are validated before the import runs."}</small>
                  <div className="artifact-slot__list">
                    {files.length > 0 ? (
                      files.map((file) => <span key={`${requirement.slot}-${file.name}`}>{file.name}</span>)
                    ) : (
                      <span>No files selected.</span>
                    )}
                  </div>
                </label>
              );
            })}
          </div>
        </div>

        <div className="model-upload-sheet__section">
          <div className="panel__eyebrow">Optional Dashboard</div>
          <label
            htmlFor={fieldId("dashboard-bundle")}
            className={`field-shell${fieldErrorFor(state, "dashboard") ? " field-shell--error" : ""}`}
          >
            <span>Dashboard bundle</span>
            <input
              id={fieldId("dashboard-bundle")}
              aria-label="Dashboard bundle"
              ref={(input) => {
                if (!input) {
                  return;
                }
                input.setAttribute("webkitdirectory", "");
                input.setAttribute("directory", "");
              }}
              type="file"
              multiple
              onChange={(event) =>
                dispatch({
                  type: "set-dashboard-files",
                  files: Array.from(event.target.files ?? []),
                })
              }
            />
            <small>{fieldErrorFor(state, "dashboard") ?? "Optional. If attached, the bundle must include dashboard-manifest.json."}</small>
            <div className="artifact-slot__list">
              {state.dashboardFiles.length > 0 ? (
                state.dashboardFiles.slice(0, 6).map((file) => (
                  <span key={normalizeDashboardRelativePath(file)}>
                    {normalizeDashboardRelativePath(file)}
                  </span>
                ))
              ) : (
                <span>No dashboard bundle selected.</span>
              )}
            </div>
          </label>
        </div>

        {isLocalPreflight(state.preflight) ? (
          <div className="model-upload-sheet__section" data-testid="local-preflight-summary">
            <div className="panel__eyebrow">Validation Result</div>
            <div className="upload-review">
              {state.preflight.artifact_checks.map((check) => (
                <div key={check.slot} className="upload-review__item">
                  <strong>{check.title}</strong>
                  <span className={`status-chip status-chip--${check.valid ? "ready" : "missing"}`}>
                    {check.valid ? "valid" : "needs attention"}
                  </span>
                  <p>{check.message ?? "Looks good."}</p>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    );
  };

  const renderHuggingFaceValidateStep = () => (
    <div className="model-upload-sheet__content">
      <div className="model-upload-sheet__section">
        <div className="panel__eyebrow">Hugging Face Repo</div>
        <label
          htmlFor={fieldId("hf-repo")}
          className={`field-shell${fieldErrorFor(state, "hf_repo") ? " field-shell--error" : ""}`}
        >
          <span>Repo URL or repo id</span>
          <input
            id={fieldId("hf-repo")}
            aria-label="Repo URL or repo id"
            value={state.hfRepoInput}
            onChange={(event) => dispatch({ type: "set-hf-repo-input", value: event.target.value })}
            placeholder="org/model-name or https://huggingface.co/org/model-name"
            data-testid="hf-repo-input"
          />
          <small>{fieldErrorFor(state, "hf_repo") ?? "We accept a full huggingface.co model URL or a repo id like org/model-name."}</small>
        </label>
      </div>

      {isHuggingFacePreflight(state.preflight) ? (
        <div className="model-upload-sheet__section" data-testid="hf-preflight-summary">
          <div className="panel__eyebrow">Preflight Result</div>
          <div className="upload-review">
            <div className="upload-review__item">
              <strong>Detected type</strong>
              <span>{state.preflight.detected_framework_type ? frameworkLabel(state.preflight.detected_framework_type as FrameworkType) : "Unknown"}</span>
            </div>
            <div className="upload-review__item">
              <strong>Task</strong>
              <span>{state.preflight.detected_task ?? "Unknown"}</span>
            </div>
            <div className="upload-review__item">
              <strong>Estimated size</strong>
              <span>{formatBytes(state.preflight.estimated_download_size_bytes)}</span>
            </div>
            <div className="upload-review__item">
              <strong>Disk free</strong>
              <span>{formatBytes(state.preflight.disk_free_bytes)}</span>
            </div>
            <div className="upload-review__item">
              <strong>Memory estimate</strong>
              <span>{formatBytes(state.preflight.memory_estimate_bytes)}</span>
            </div>
            <div className="upload-review__item">
              <strong>Compatibility</strong>
              <span className={`status-chip status-chip--${state.preflight.ready_to_import ? "ready" : "incompatible"}`}>
                {state.preflight.ready_to_import ? "ready to import" : "blocked"}
              </span>
            </div>
          </div>

          <div className="model-upload-sheet__preflight-list">
            {state.preflight.required_files.map((file) => (
              <div key={file.path} className="upload-review__item">
                <strong>{file.path}</strong>
                <span className={`status-chip status-chip--${file.available ? "ready" : "missing"}`}>
                  {file.available ? "found" : "missing"}
                </span>
                <p>{file.message ?? `${file.category} · ${formatBytes(file.size_bytes)}`}</p>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );

  const renderValidateStep = () =>
    branch === "huggingface" ? renderHuggingFaceValidateStep() : renderLocalValidateStep();

  const renderReviewStep = () => {
    const metadata = state.preflight?.normalized_metadata ?? currentMetadata;
    if (!metadata || !branch || !domainChoice) {
      return null;
    }

    const warnings = state.preflight?.warnings ?? [];
    const configSourceLabel =
      isLocalPreflight(state.preflight) && state.preflight.config_source === "uploaded"
        ? "Uploaded by user"
        : "Generated by system";
    return (
      <div className="model-upload-sheet__content" data-testid="review-step">
        <div className="model-upload-sheet__section">
          <div className="panel__eyebrow">Review</div>
          <div className="upload-review">
            <div className="upload-review__item">
              <strong>Source</strong>
              <span>{state.source === "huggingface" ? "Hugging Face" : "Local computer"}</span>
            </div>
            <div className="upload-review__item">
              <strong>Upload mode</strong>
              <span>{branchHeading(branch)}</span>
            </div>
            <div className="upload-review__item">
              <strong>Model type</strong>
              <span>{frameworkLabel(metadata.framework_type)}</span>
            </div>
            <div className="upload-review__item">
              <strong>Domain</strong>
              <span>
                {metadata.ui_display_name} ({metadata.domain})
              </span>
            </div>
            <div className="upload-review__item">
              <strong>Display name</strong>
              <span>{metadata.display_name}</span>
            </div>
            <div className="upload-review__item">
              <strong>Model id</strong>
              <span>{metadata.model_id}</span>
            </div>
            <div className="upload-review__item">
              <strong>Config source</strong>
              <span>{configSourceLabel}</span>
            </div>
            <div className="upload-review__item">
              <strong>Runtime</strong>
              <span>
                {metadata.runtime_device} · max {metadata.runtime_max_sequence_length} · batch {metadata.runtime_batch_size}
              </span>
            </div>
            <div className="upload-review__item">
              <strong>Labels</strong>
              <span>{metadata.labels.length} configured</span>
            </div>
            <div className="upload-review__item">
              <strong>Dashboard</strong>
              <span>{state.dashboardFiles.length > 0 ? "Attached" : "Not attached"}</span>
            </div>
            <div className="upload-review__item">
              <strong>{branch === "huggingface" ? "Repo" : "Artifacts"}</strong>
              <span>
                {branch === "huggingface"
                  ? state.hfRepoInput.trim()
                  : Object.values(state.artifactFiles).flat().length}{" "}
                {branch === "huggingface" ? "" : "files selected"}
              </span>
            </div>
          </div>
        </div>

        {warnings.length > 0 ? (
          <div className="inline-alert">
            {warnings.join(" ")}
          </div>
        ) : null}

        <div className="model-upload-sheet__section">
          <div className="panel__eyebrow">Saved Config Preview</div>
          <details className="model-upload-sheet__example-config" open>
            <summary>View generated / normalized config</summary>
            <pre>{state.preflight?.config_preview}</pre>
          </details>
        </div>
      </div>
    );
  };

  const renderProgressStep = () => (
    <div className="model-upload-sheet__content" data-testid="progress-step">
      <div className="model-upload-sheet__intro-card">
        <div className="panel__eyebrow">Progress</div>
        <h3>
          {state.submissionStatus === "running"
            ? branch === "huggingface"
              ? "Importing the model"
              : "Registering the model"
            : "The request finished with an error"}
        </h3>
        <p>
          {state.submissionStatus === "running"
            ? "Back and close are locked until the backend returns. This avoids fake cancellation while the import is still running."
            : state.message ?? "The request ended early. You can return to the review step and try again."}
        </p>
      </div>

      <div className="model-upload-sheet__progress-list">
        {progressStages(branch).map((stage, index) => (
          <div key={stage} className="upload-review__item">
            <strong>{index + 1}</strong>
            <p>{stage}</p>
          </div>
        ))}
      </div>

      {state.submissionStatus === "error" ? (
        <div className="inline-alert">{state.message}</div>
      ) : null}
    </div>
  );

  const renderResultStep = () => {
    if (!state.result) {
      return null;
    }
    const { result } = state.result;
    return (
      <div className="model-upload-sheet__content" data-testid="result-step">
        <div className="model-upload-sheet__intro-card">
          <div className="panel__eyebrow">Result</div>
          <h3>{result.display_name} is registered</h3>
          <p>The catalog has already been refreshed behind the modal, so the model is ready to inspect in Model Management.</p>
        </div>

        <div className="upload-review">
          <div className="upload-review__item">
            <strong>Model id</strong>
            <span>{result.model_id}</span>
          </div>
          <div className="upload-review__item">
            <strong>Domain</strong>
            <span>{result.domain}</span>
          </div>
          <div className="upload-review__item">
            <strong>Active</strong>
            <span>{result.is_active ? "Yes" : "No"}</span>
          </div>
          <div className="upload-review__item">
            <strong>Runtime status</strong>
            <span>{result.status}</span>
          </div>
          <div className="upload-review__item">
            <strong>Dashboard</strong>
            <span>{result.dashboard_status}</span>
          </div>
          <div className="upload-review__item">
            <strong>Config source</strong>
            <span>{result.config_source === "uploaded" ? "Uploaded by user" : "Generated by system"}</span>
          </div>
        </div>

        {result.warnings.length > 0 ? <div className="inline-alert">{result.warnings.join(" ")}</div> : null}
      </div>
    );
  };

  const renderBody = () => {
    if (state.step === "source") {
      return renderSourceStep();
    }
    if (state.step === "details") {
      return renderDetailsStep();
    }
    if (state.step === "validate") {
      return renderValidateStep();
    }
    if (state.step === "review") {
      return renderReviewStep();
    }
    if (state.step === "progress") {
      return renderProgressStep();
    }
    return renderResultStep();
  };

  const renderFooter = () => {
    if (state.step === "source") {
      return (
        <>
          <button type="button" className="mini-button" onClick={closeWizard}>
            Cancel
          </button>
          <button type="button" className="primary-button" onClick={handleSourceNext}>
            Continue
          </button>
        </>
      );
    }
    if (state.step === "details") {
      return (
        <>
          <button type="button" className="mini-button" onClick={() => dispatch({ type: "set-step", step: "source" })}>
            Back
          </button>
          <button type="button" className="primary-button" onClick={handleDetailsNext}>
            Continue
          </button>
        </>
      );
    }
    if (state.step === "validate") {
      return (
        <>
          <button type="button" className="mini-button" onClick={() => dispatch({ type: "set-step", step: "details" })}>
            Back
          </button>
          {canGoToReview ? (
            <button
              type="button"
              className="primary-button"
              onClick={() => dispatch({ type: "set-step", step: "review" })}
            >
              Continue to review
            </button>
          ) : (
            <button type="button" className="primary-button" onClick={() => void handleRunValidation()}>
              {branch === "huggingface" ? "Run preflight" : "Validate files"}
            </button>
          )}
        </>
      );
    }
    if (state.step === "review") {
      return (
        <>
          <button type="button" className="mini-button" onClick={() => dispatch({ type: "set-step", step: "validate" })}>
            Back
          </button>
          <button
            type="button"
            className="primary-button"
            onClick={() => void handleSubmit()}
            data-testid="review-submit"
          >
            {branch === "huggingface" ? "Import model" : "Upload model"}
          </button>
        </>
      );
    }
    if (state.step === "progress") {
      return state.submissionStatus === "error" ? (
        <>
          <button type="button" className="mini-button" onClick={() => dispatch({ type: "set-step", step: "review" })}>
            Back to review
          </button>
        </>
      ) : (
        <>
          <span className="dashboard-mini-chip">Request in progress</span>
        </>
      );
    }
    return (
      <>
        <button type="button" className="mini-button" onClick={closeWizard}>
          View model in management
        </button>
        {state.result?.result.dashboard_status === "available" ? (
          <button type="button" className="primary-button" onClick={closeWizard}>
            Open dashboard
          </button>
        ) : null}
      </>
    );
  };

  return (
    <div className="model-upload-sheet__overlay" onClick={closeWizard}>
      <section
        className="model-upload-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby="model-upload-sheet-title"
        onClick={(event) => event.stopPropagation()}
      >
        <aside className="model-upload-sheet__rail">
          <div className="panel__eyebrow">Upload Model</div>
          <h2 id="model-upload-sheet-title">{branchHeading(branch)}</h2>

          <div className="model-upload-sheet__stepper">
            {STEP_DEFS.map((step, index) => {
              const isActive = step.id === state.step;
              const isComplete = currentStepIndex > index;
              return (
                <div
                  key={step.id}
                  className={`model-upload-sheet__step${isActive ? " model-upload-sheet__step--active" : ""}${isComplete ? " model-upload-sheet__step--complete" : ""}`}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <strong>{step.label}</strong>
                </div>
              );
            })}
          </div>
        </aside>

        <div className="model-upload-sheet__main">
          <header className="model-upload-sheet__header">
            <div>
              <div className="panel__eyebrow">Wizard</div>
              {state.step === "source" ? null : <h3>{STEP_DEFS.find((step) => step.id === state.step)?.label}</h3>}
            </div>
            <button type="button" className="mini-button" onClick={closeWizard} disabled={!canClose}>
              Close
            </button>
          </header>

          {state.message ? <div className="inline-alert">{state.message}</div> : null}

          {renderBody()}

          <footer className="model-upload__actions">{renderFooter()}</footer>
        </div>
      </section>
    </div>
  );
}
