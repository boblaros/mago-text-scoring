import { useEffect, useId, useMemo, useReducer, useRef, useState } from "react";
import huggingFaceLogo from "../../assets/huggingface-logo.svg";
import {
  ApiRequestError,
  importHuggingFaceModel,
  importLocalModel,
  LOCAL_UPLOAD_LIMIT_MB,
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
  TASK_OPTIONS,
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
  type ArtifactRequirement,
  type BranchMode,
  type DomainChoice,
  type FrameworkType,
  type UploadSource,
  type WizardMetadataDraft,
  type WizardStep,
} from "./modelUploadWizardSchema";

type PreflightResult = LocalUploadPreflightResponse | HuggingFacePreflightResponse;
type LocalConfigMode = "unknown" | "uploaded-config" | "manual-metadata";
type LocalDetailsStage = "question" | "details";
type LabelInputMode = "manual" | "file";

interface WizardState {
  step: WizardStep;
  source: UploadSource | null;
  metadata: WizardMetadataDraft;
  localConfigMode: LocalConfigMode;
  localDetailsStage: LocalDetailsStage;
  labelInputMode: LabelInputMode;
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
  | { type: "set-local-config-mode"; mode: LocalConfigMode }
  | { type: "set-local-details-stage"; stage: LocalDetailsStage }
  | { type: "set-label-input-mode"; mode: LabelInputMode }
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
  { id: "details", label: "Model Metadata" },
  { id: "validate", label: "Model Artifacts" },
  { id: "review", label: "Review" },
  { id: "result", label: "Results" },
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

function ReadyConfigIcon() {
  return (
    <svg viewBox="0 0 48 48" className="model-upload-sheet__choice-icon" aria-hidden="true">
      <path d="M15 10h13l5 5v23H15z" />
      <path d="M28 10v5h5" />
      <path d="M18.5 23.5l3.8 3.8L29.5 20" />
      <path d="M19 34h10" />
    </svg>
  );
}

function ManualMetadataIcon() {
  return (
    <svg viewBox="0 0 48 48" className="model-upload-sheet__choice-icon" aria-hidden="true">
      <path d="M14 14h20" />
      <path d="M14 22h20" />
      <path d="M14 30h12" />
      <circle cx="19" cy="14" r="2.5" />
      <circle cx="27" cy="22" r="2.5" />
      <path d="M30.5 28.5l6 6" />
      <path d="M33.5 25.5l3 3" />
    </svg>
  );
}

function UploadSuccessIcon() {
  return (
    <svg viewBox="0 0 48 48" className="model-upload-sheet__success-icon-svg" aria-hidden="true">
      <circle cx="24" cy="24" r="18" />
      <path d="M16.5 24.5l5.2 5.2L31.5 18.8" />
    </svg>
  );
}

type ArtifactVisualKind = "weights" | "tokenizer" | "config" | "dashboard" | "generic";

function resolveArtifactVisualKind(slot: string): ArtifactVisualKind {
  if (slot === "weights") {
    return "weights";
  }
  if (slot === "tokenizer" || slot === "vocabulary") {
    return "tokenizer";
  }
  if (slot === "config") {
    return "config";
  }
  if (slot === "dashboard") {
    return "dashboard";
  }
  return "generic";
}

function ArtifactCardIcon({ kind }: { kind: ArtifactVisualKind }) {
  if (kind === "weights") {
    return (
      <svg viewBox="0 0 24 24" className="artifact-slot__icon" aria-hidden="true">
        <rect x="5" y="5" width="14" height="4" rx="2" />
        <rect x="3.5" y="10" width="17" height="4" rx="2" />
        <rect x="5" y="15" width="14" height="4" rx="2" />
      </svg>
    );
  }
  if (kind === "tokenizer") {
    return (
      <svg viewBox="0 0 24 24" className="artifact-slot__icon" aria-hidden="true">
        <path d="M7 4.5h7l3 3v12H7z" />
        <path d="M14 4.5v3h3" />
        <path d="M9.5 11h5" />
        <path d="M9.5 14h5" />
        <path d="M9.5 17h3.5" />
      </svg>
    );
  }
  if (kind === "config") {
    return (
      <svg viewBox="0 0 24 24" className="artifact-slot__icon" aria-hidden="true">
        <path d="M5 7.5h14" />
        <path d="M5 12h14" />
        <path d="M5 16.5h14" />
        <circle cx="9" cy="7.5" r="1.8" />
        <circle cx="15" cy="12" r="1.8" />
        <circle cx="11" cy="16.5" r="1.8" />
      </svg>
    );
  }
  if (kind === "dashboard") {
    return (
      <svg viewBox="0 0 24 24" className="artifact-slot__icon" aria-hidden="true">
        <rect x="4.5" y="4.5" width="6" height="6" rx="1.6" />
        <rect x="13.5" y="4.5" width="6" height="10" rx="1.6" />
        <rect x="4.5" y="13.5" width="6" height="6" rx="1.6" />
        <rect x="13.5" y="17.5" width="6" height="2" rx="1" />
      </svg>
    );
  }
  return (
    <svg viewBox="0 0 24 24" className="artifact-slot__icon" aria-hidden="true">
      <path d="M7 4.5h7l3 3v12H7z" />
      <path d="M14 4.5v3h3" />
    </svg>
  );
}

function createInitialState(existingDomain: string): WizardState {
  return {
    step: "source",
    source: null,
    metadata: createDefaultMetadataDraft(),
    localConfigMode: "unknown",
    localDetailsStage: "question",
    labelInputMode: "manual",
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

function detectFrameworkTypeFromConfigText(text: string): FrameworkType | null {
  try {
    const parsed = JSON.parse(text) as {
      framework?: { type?: string };
      framework_type?: string;
    };
    const frameworkType = parsed.framework?.type ?? parsed.framework_type;
    if (frameworkType === "transformers" || frameworkType === "pytorch" || frameworkType === "sklearn") {
      return frameworkType;
    }
  } catch {
    // Fall through to a lightweight YAML-style scan.
  }

  const lines = text.split(/\r?\n/);
  let inFrameworkBlock = false;
  let frameworkIndent = 0;
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const indent = line.length - line.trimStart().length;
    if (!inFrameworkBlock && trimmed === "framework:") {
      inFrameworkBlock = true;
      frameworkIndent = indent;
      continue;
    }
    if (!inFrameworkBlock) {
      continue;
    }
    if (indent <= frameworkIndent) {
      inFrameworkBlock = false;
      continue;
    }

    const typeMatch = trimmed.match(/^type:\s*["']?([a-z-]+)["']?\s*$/i);
    if (!typeMatch) {
      continue;
    }
    const frameworkType = typeMatch[1];
    if (frameworkType === "transformers" || frameworkType === "pytorch" || frameworkType === "sklearn") {
      return frameworkType;
    }
    return null;
  }

  return null;
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
      state.step === "review" || state.step === "result"
        ? "details"
        : state.step,
  };
}

function defaultLabelDisplayName(label: UploadLabelClass) {
  return `Class ${label.id}`;
}

function applyLabelPatch(
  label: UploadLabelClass,
  patch: Partial<UploadLabelClass>,
): UploadLabelClass {
  const nextLabel = { ...label, ...patch };

  if (typeof patch.name === "string") {
    const keepsDefaultDisplayName =
      !label.display_name ||
      label.display_name === label.name ||
      label.display_name === defaultLabelDisplayName(label);
    if (keepsDefaultDisplayName) {
      nextLabel.display_name = patch.name;
    }
  }

  return nextLabel;
}

function normalizeLabelClasses(labels: UploadLabelClass[]): UploadLabelClass[] {
  return labels.map((label) => ({
    ...label,
    display_name: label.display_name ?? label.name,
  }));
}

function isNonNull<T>(value: T | null): value is T {
  return value !== null;
}

function parseLabelMapPayload(payload: unknown): UploadLabelClass[] | null {
  if (Array.isArray(payload)) {
    const labels = payload
      .map((value, index) =>
        typeof value === "string" && value.trim()
          ? {
              id: index,
              name: value.trim(),
              display_name: value.trim(),
            }
          : null,
      )
      .filter(isNonNull);
    return labels.length > 0 ? labels : null;
  }

  if (!payload || typeof payload !== "object") {
    return null;
  }

  const record = payload as Record<string, unknown>;
  if (record.id2label) {
    return parseLabelMapPayload(record.id2label);
  }
  if (record.label2id) {
    return parseLabelMapPayload(record.label2id);
  }

  const entries = Object.entries(record);
  if (entries.length === 0) {
    return null;
  }

  const numericKeyEntries = entries
    .map(([key, value]) => {
      const id = Number(key);
      if (!Number.isInteger(id) || typeof value !== "string" || !value.trim()) {
        return null;
      }
      return {
        id,
        name: value.trim(),
        display_name: value.trim(),
      } satisfies UploadLabelClass;
    })
    .filter(isNonNull)
    .sort((left, right) => left.id - right.id);
  if (numericKeyEntries.length === entries.length) {
    return numericKeyEntries;
  }

  const numericValueEntries = entries
    .map(([key, value]) => {
      if (typeof key !== "string" || !key.trim()) {
        return null;
      }
      const id = Number(value);
      if (!Number.isInteger(id)) {
        return null;
      }
      return {
        id,
        name: key.trim(),
        display_name: key.trim(),
      } satisfies UploadLabelClass;
    })
    .filter(isNonNull)
    .sort((left, right) => left.id - right.id);
  if (numericValueEntries.length === entries.length) {
    return numericValueEntries;
  }

  return null;
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
        localConfigMode: action.source === "local" ? "unknown" : state.localConfigMode,
        localDetailsStage: action.source === "local" ? "question" : "details",
        step: "source",
      };
    case "set-local-config-mode":
      return {
        ...clearExecutionState(state),
        localConfigMode: action.mode,
        labelInputMode: action.mode === "uploaded-config" ? "manual" : state.labelInputMode,
        artifactFiles:
          action.mode === "uploaded-config"
            ? {
                ...state.artifactFiles,
                label_map_file: [],
              }
            : state.artifactFiles,
        registrationConfigFiles:
          action.mode === "manual-metadata" ? [] : state.registrationConfigFiles,
      };
    case "set-local-details-stage":
      return {
        ...state,
        localDetailsStage: action.stage,
        fieldErrors: {},
        message: null,
      };
    case "set-label-input-mode": {
      const nextState = clearExecutionState(state);
      return {
        ...nextState,
        labelInputMode: action.mode,
        artifactFiles:
          action.mode === "file"
            ? nextState.artifactFiles
            : {
                ...nextState.artifactFiles,
                label_map_file: [],
              },
      };
    }
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
            index === action.index ? applyLabelPatch(label, action.patch) : label,
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
        step: "review",
        submissionStatus: "running",
        fieldErrors: {},
        message: null,
      };
    case "submit-error":
      return {
        ...state,
        step: "review",
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
    } else if (key === "metadata") {
      mapped.local_config_mode = value;
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
  const canUploadLabelMapFile =
    state.source === "local" && state.metadata.framework_type === "transformers";
  if (state.source === "local") {
    if (state.localConfigMode === "unknown") {
      errors.local_config_mode = "Choose whether you already have a ready upload config.";
      return errors;
    }
    if (state.localConfigMode === "uploaded-config") {
      if (state.registrationConfigFiles.length === 0) {
        errors.registration_config = "Upload the existing registration config file to continue.";
      }
      return errors;
    }
  }

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
  if (
    state.labelInputMode === "file" &&
    canUploadLabelMapFile &&
    (state.artifactFiles.label_map_file ?? []).length === 0
  ) {
    errors["artifacts.label_map_file"] =
      "Upload a label map file or switch to manual labels.";
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
): Promise<Record<string, string>> {
  const errors: Record<string, string> = {};
  const frameworkType = state.metadata.framework_type;
  const requirements = ARTIFACT_REQUIREMENTS[frameworkType];
  for (const requirement of requirements) {
    const files = state.artifactFiles[requirement.slot] ?? [];
    const error =
      frameworkType === "transformers" &&
      state.metadata.framework_task === "sequence-classification" &&
      (requirement.slot === "weights"
        || requirement.slot === "tokenizer"
        || requirement.slot === "config")
        ? validateTransformersSequenceArtifactFiles(requirement.slot, files)
        : validateArtifactFilesByExtension(requirement, files);
    if (error) {
      errors[`artifacts.${requirement.slot}`] = error;
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

const TRANSFORMER_SEQUENCE_WEIGHT_FILENAMES = new Set([
  "model.safetensors",
  "pytorch_model.bin",
]);
const TRANSFORMER_SEQUENCE_TOKENIZER_SINGLE_FILE_FILENAMES = new Set([
  "tokenizer.json",
  "vocab.txt",
  "tokenizer.model",
  "spiece.model",
  "sentencepiece.bpe.model",
]);
const TRANSFORMER_SEQUENCE_TOKENIZER_PAIR_FILENAMES = new Set([
  "vocab.json",
  "merges.txt",
]);
const TRANSFORMER_SEQUENCE_TOKENIZER_OPTIONAL_FILENAMES = new Set([
  "special_tokens_map.json",
  "added_tokens.json",
]);
const TRANSFORMER_SEQUENCE_CONFIG_FILENAMES = new Set(["config.json"]);
const TRANSFORMER_SEQUENCE_SLOT_LABELS: Record<string, string> = {
  weights: "Weights",
  tokenizer: "Tokenizer Assets",
  config: "Runtime Config Assets",
};
const TRANSFORMER_SEQUENCE_FILENAME_TO_SLOT: Record<string, string> = {
  "model.safetensors": "weights",
  "pytorch_model.bin": "weights",
  "tokenizer.json": "tokenizer",
  "vocab.txt": "tokenizer",
  "tokenizer.model": "tokenizer",
  "spiece.model": "tokenizer",
  "sentencepiece.bpe.model": "tokenizer",
  "vocab.json": "tokenizer",
  "merges.txt": "tokenizer",
  "special_tokens_map.json": "tokenizer",
  "added_tokens.json": "tokenizer",
  "tokenizer_config.json": "tokenizer",
  "config.json": "config",
};

function validateArtifactFilesByExtension(
  requirement: ArtifactRequirement,
  files: File[],
): string | null {
  if (requirement.required && files.length === 0) {
    return `${requirement.title} requires at least 1 file.`;
  }

  const allowedExtensions = parseAcceptedExtensions(requirement.accepts);
  const invalidFile = files.find((file) => {
    const extension = `.${file.name.split(".").pop()?.toLowerCase() ?? ""}`;
    return !allowedExtensions.includes(extension);
  });
  if (invalidFile) {
    return `${invalidFile.name} has an unsupported file type.`;
  }

  return null;
}

function validateTransformersSequenceArtifactFiles(
  slot: string,
  files: File[],
): string | null {
  const basenames = files.map((file) => file.name.split(/[/\\]/).pop() ?? file.name);
  const normalizedNames = new Set(basenames.map((name) => name.toLowerCase()));

  if (slot === "weights") {
    if (basenames.length === 0) {
      return "Weights must include model.safetensors or pytorch_model.bin.";
    }
    const invalidNames = basenames.filter(
      (name) => !TRANSFORMER_SEQUENCE_WEIGHT_FILENAMES.has(name.toLowerCase()),
    );
    if (invalidNames.length > 0) {
      return (
        "Weights only accept model.safetensors or pytorch_model.bin for transformer sequence-classification uploads. "
        + describeTransformersSequenceFileIssue(invalidNames[0], slot)
      );
    }
    if (basenames.length > 2) {
      return "Weights accept at most model.safetensors and pytorch_model.bin.";
    }
    return null;
  }

  if (slot === "config") {
    const invalidNames = basenames.filter(
      (name) => !TRANSFORMER_SEQUENCE_CONFIG_FILENAMES.has(name.toLowerCase()),
    );
    if (!normalizedNames.has("config.json")) {
      if (invalidNames.length > 0) {
        return `Runtime Config Assets must include config.json. ${describeTransformersSequenceFileIssue(invalidNames[0], slot)}`;
      }
      return "Runtime Config Assets must include config.json.";
    }
    if (invalidNames.length > 0) {
      return (
        "Runtime Config Assets only accept config.json for transformer sequence-classification uploads. "
        + describeTransformersSequenceFileIssue(invalidNames[0], slot)
      );
    }
    return null;
  }

  if (slot === "tokenizer") {
    if (basenames.length === 0) {
      return (
        "Tokenizer Assets are incomplete. Upload tokenizer_config.json plus tokenizer.json, vocab.txt, tokenizer.model, spiece.model, sentencepiece.bpe.model, or vocab.json with merges.txt."
      );
    }

    const allowedNames = new Set([
      ...TRANSFORMER_SEQUENCE_TOKENIZER_SINGLE_FILE_FILENAMES,
      ...TRANSFORMER_SEQUENCE_TOKENIZER_PAIR_FILENAMES,
      ...TRANSFORMER_SEQUENCE_TOKENIZER_OPTIONAL_FILENAMES,
      "tokenizer_config.json",
    ]);
    const invalidNames = basenames.filter((name) => !allowedNames.has(name.toLowerCase()));
    if (invalidNames.length > 0) {
      return (
        "Tokenizer Assets only accept tokenizer_config.json, tokenizer.json, vocab.txt, tokenizer.model, spiece.model, sentencepiece.bpe.model, vocab.json, merges.txt, special_tokens_map.json, or added_tokens.json. "
        + describeTransformersSequenceFileIssue(invalidNames[0], slot)
      );
    }

    const missingParts: string[] = [];
    const hasTokenizerConfig = normalizedNames.has("tokenizer_config.json");
    const hasVocabJson = normalizedNames.has("vocab.json");
    const hasMergesTxt = normalizedNames.has("merges.txt");
    const hasTokenizerDefinition =
      Array.from(TRANSFORMER_SEQUENCE_TOKENIZER_SINGLE_FILE_FILENAMES).some((name) =>
        normalizedNames.has(name),
      ) || (hasVocabJson && hasMergesTxt);

    if (!hasTokenizerConfig) {
      missingParts.push("tokenizer_config.json");
    }
    if (hasVocabJson && !hasMergesTxt) {
      missingParts.push("merges.txt");
    } else if (hasMergesTxt && !hasVocabJson) {
      missingParts.push("vocab.json");
    } else if (!hasTokenizerDefinition) {
      missingParts.push(
        "one tokenizer definition file (tokenizer.json, vocab.txt, tokenizer.model, spiece.model, sentencepiece.bpe.model, or vocab.json with merges.txt)",
      );
    }

    if (missingParts.length > 0) {
      return `Tokenizer Assets are incomplete. Missing ${formatMissingItems(missingParts)}.`;
    }
  }

  return null;
}

function describeTransformersSequenceFileIssue(filename: string, currentSlot: string): string {
  const normalizedName = (filename.split(/[/\\]/).pop() ?? filename).toLowerCase();
  const expectedSlot = TRANSFORMER_SEQUENCE_FILENAME_TO_SLOT[normalizedName];
  if (expectedSlot && expectedSlot !== currentSlot) {
    return `${filename} is misplaced. Move it to ${TRANSFORMER_SEQUENCE_SLOT_LABELS[expectedSlot]}.`;
  }
  if (currentSlot === "weights") {
    return `${filename} is not a valid transformer weight file.`;
  }
  if (currentSlot === "config") {
    return `${filename} is not a supported runtime config file for this slot.`;
  }
  if (currentSlot === "tokenizer") {
    return `${filename} is not a supported tokenizer asset for this slot.`;
  }
  return `${filename} is not supported in this artifact slot.`;
}

function formatMissingItems(items: string[]): string {
  const uniqueItems = Array.from(new Set(items));
  if (uniqueItems.length === 1) {
    return uniqueItems[0]!;
  }
  if (uniqueItems.length === 2) {
    return `${uniqueItems[0]} and ${uniqueItems[1]}`;
  }
  return `${uniqueItems.slice(0, -1).join(", ")}, and ${uniqueItems[uniqueItems.length - 1]}`;
}

function getCurrentMetadata(state: WizardState, domains: DomainChoice[]) {
  if (state.source === "local" && state.localConfigMode === "uploaded-config") {
    return null;
  }
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

const HUGGING_FACE_SIZE_ESTIMATE_WARNING =
  "Download size could not be estimated precisely because one or more repo files do not report a size.";

function humanizeHuggingFacePreflightWarning(warning: string) {
  if (warning === HUGGING_FACE_SIZE_ESTIMATE_WARNING) {
    return "Some Hugging Face files do not publish their size, so the download estimate is approximate. The import can still work, but the final size may be larger than shown.";
  }
  return warning;
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
  const branch = resolveBranchMode(state.source);
  const domainChoice = resolveDomainChoice(state, domains);
  const currentMetadata = useMemo(() => getCurrentMetadata(state, domains), [state, domains]);
  const currentFramework = state.metadata.framework_type;
  const canUploadLabelMapFile = branch === "local" && currentFramework === "transformers";
  const visibleArtifactRequirements = useMemo(
    () =>
      ARTIFACT_REQUIREMENTS[currentFramework].filter(
        (requirement) =>
          requirement.slot !== "label_map_file" && requirement.slot !== "label_classes_file",
      ),
    [currentFramework],
  );
  const mainPanelRef = useRef<HTMLDivElement | null>(null);
  const validateControllerRef = useRef<AbortController | null>(null);
  const submitControllerRef = useRef<AbortController | null>(null);
  const [optionalDetailsOpen, setOptionalDetailsOpen] = useState(false);
  const [reviewConfigOpen, setReviewConfigOpen] = useState(false);
  const fieldId = (suffix: string) => `${fieldIdPrefix}-${suffix}`;
  const localUploadTooltip =
    `Local upload max: ${LOCAL_UPLOAD_LIMIT_MB} MB total. For transformers, upload only weights, tokenizer files, and config.json.`;
  const renderInfoLabel = (label: string, tooltip: string) => (
    <span className="field-shell__label-row">
      <span>{label}</span>
      <span
        className="field-info-badge"
        tabIndex={0}
        role="note"
        aria-label={tooltip}
        data-tooltip={tooltip}
      >
        i
      </span>
    </span>
  );
  const renderWarningBadge = (tooltip: string) => (
    <span
      className="field-warning-badge"
      tabIndex={0}
      role="note"
      aria-label={tooltip}
      data-tooltip={tooltip}
    >
      !
    </span>
  );
  const handleRegistrationConfigFilesChange = async (files: File[]) => {
    dispatch({ type: "set-registration-config-files", files });
    const primaryFile = files[0];
    if (!primaryFile) {
      return;
    }

    const detectedFrameworkType = detectFrameworkTypeFromConfigText(await primaryFile.text());
    if (!detectedFrameworkType) {
      return;
    }
    dispatch({
      type: "set-metadata",
      key: "framework_type",
      value: detectedFrameworkType,
    });
  };
  const handleLabelMapFilesChange = async (files: File[]) => {
    dispatch({ type: "set-artifact-files", slot: "label_map_file", files });
    const primaryFile = files[0];
    if (!primaryFile || !primaryFile.name.toLowerCase().endsWith(".json")) {
      return;
    }
    try {
      const parsed = parseLabelMapPayload(JSON.parse(await primaryFile.text()));
      if (!parsed || parsed.length === 0) {
        return;
      }
      dispatch({
        type: "patch-metadata",
        patch: {
          labels: normalizeLabelClasses(parsed),
        },
      });
    } catch {
      // Keep the uploaded file even if the browser cannot prefill labels from it.
    }
  };

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

  useEffect(() => {
    const container = mainPanelRef.current?.querySelector<HTMLDivElement>(
      ".model-upload-sheet__content",
    );
    if (!container) {
      return;
    }
    if (typeof container.scrollTo === "function") {
      container.scrollTo({ top: 0, left: 0, behavior: "auto" });
      return;
    }
    container.scrollTop = 0;
  }, [state.step]);

  useEffect(() => {
    if (state.labelInputMode === "file" && !canUploadLabelMapFile) {
      dispatch({ type: "set-label-input-mode", mode: "manual" });
    }
  }, [canUploadLabelMapFile, state.labelInputMode]);

  useEffect(() => {
    if (state.step !== "review" && reviewConfigOpen) {
      setReviewConfigOpen(false);
    }
  }, [reviewConfigOpen, state.step]);

  useEffect(() => {
    if (!reviewConfigOpen) {
      return;
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setReviewConfigOpen(false);
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [reviewConfigOpen]);

  if (!isOpen) {
    return null;
  }

  const canClose = state.submissionStatus !== "running";
  const isSubmitting = state.submissionStatus === "running";
  const huggingFacePreflight = isHuggingFacePreflight(state.preflight) ? state.preflight : null;
  const isHuggingFaceValidationRunning =
    state.step === "validate" &&
    branch === "huggingface" &&
    state.message === "Inspecting the remote repo...";
  const validationErrors = Object.entries(state.fieldErrors)
    .filter(([key]) =>
      branch === "huggingface"
        ? key === "hf_repo"
        : key === "dashboard" || key.startsWith("artifacts."),
    )
    .map(([, value]) => value);
  const localValidationDetail =
    validationErrors[0]
    ?? (isLocalPreflight(state.preflight)
      ? state.preflight.artifact_checks.find((check) => !check.valid)?.message ?? null
      : null)
    ?? state.message;
  const huggingFaceValidateDetail =
    validationErrors[0]
    ?? (huggingFacePreflight
      ? huggingFacePreflight.blocking_reasons[0] ?? null
      : null)
    ?? state.message;
  const huggingFaceWarnings =
    branch === "huggingface" && huggingFacePreflight
      ? huggingFacePreflight.warnings.map(humanizeHuggingFacePreflightWarning)
      : [];
  const huggingFaceCanAdvanceToReview = Boolean(huggingFacePreflight?.ready_to_import);
  const huggingFacePreflightState: "idle" | "running" | "success" | "blocked" | "error" =
    branch !== "huggingface"
      ? "idle"
      : isHuggingFaceValidationRunning
        ? "running"
        : huggingFacePreflight?.ready_to_import
          ? "success"
          : huggingFacePreflight
            ? "blocked"
            : huggingFaceValidateDetail
              ? "error"
              : "idle";
  const validateStatus =
    state.step !== "validate" || !branch
      ? null
      : branch === "huggingface"
        ? null
        : isLocalPreflight(state.preflight)
          ? state.preflight.ready
            ? null
            : {
                tone: "error",
                title: "Validation failed",
                detail:
                  localValidationDetail
                  ?? "Resolve the highlighted file issues before continuing.",
              }
          : state.message === "Validating files and building a config preview..."
            ? null
            : localValidationDetail
              ? {
                  tone: "error",
                  title: "Validation failed",
                  detail: localValidationDetail,
                }
              : null;

  const closeWizard = () => {
    if (!canClose) {
      return;
    }
    onClose();
  };

  const handleSourceSelection = (source: UploadSource) => {
    dispatch({ type: "set-source", source });
    dispatch({ type: "set-step", step: "details" });
  };

  const handleLocalConfigModeSelection = (mode: Exclude<LocalConfigMode, "unknown">) => {
    dispatch({ type: "set-local-config-mode", mode });
    if (state.localDetailsStage === "question") {
      dispatch({ type: "set-local-details-stage", stage: "details" });
    }
  };

  const handleDetailsNext = () => {
    if (branch === "local" && state.localDetailsStage === "question") {
      if (state.localConfigMode === "unknown") {
        dispatch({
          type: "set-errors",
          errors: { local_config_mode: "Choose whether you already have a ready upload config." },
          message: "Choose how you want to continue.",
        });
        return;
      }
      dispatch({ type: "set-local-details-stage", stage: "details" });
      return;
    }

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

  const handleDetailsBack = () => {
    dispatch({ type: "set-step", step: "source" });
  };

  const handleRunValidation = async () => {
    if (!branch) {
      dispatch({
        type: "set-errors",
        errors: { source: "Finish the earlier steps first." },
        message: "The wizard needs source and metadata details before validation can run.",
      });
      return;
    }

    if (branch === "huggingface") {
      if (!currentMetadata || !domainChoice) {
        dispatch({
          type: "set-errors",
          errors: { source: "Finish the earlier steps first." },
          message: "The wizard needs source and metadata details before validation can run.",
        });
        return;
      }
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

    const localErrors = await validateLocalFilesStep(state);
    if (Object.keys(localErrors).length > 0) {
      dispatch({
        type: "set-errors",
        errors: localErrors,
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
          state.localConfigMode === "uploaded-config" ? null : state.metadata,
          state.localConfigMode === "uploaded-config" ? null : domainChoice,
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
      if (response.ready) {
        dispatch({ type: "set-step", step: "review" });
      }
    } catch (error) {
      dispatch({
        type: "set-errors",
        errors: mapApiFieldErrors(error),
        message: error instanceof Error ? error.message : "Validation failed.",
      });
    }
  };

  const handleSubmit = async () => {
    if (!branch || !state.preflight) {
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
        if (!currentMetadata) {
          throw new Error("The wizard lost the model metadata before import.");
        }
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
              state.localConfigMode === "uploaded-config"
                ? fromUploadMetadata(state.preflight.normalized_metadata)
                : state.metadata,
              state.localConfigMode === "uploaded-config" ? {
                domain: state.preflight.normalized_metadata.domain,
                display_name: state.preflight.normalized_metadata.ui_display_name ?? "",
                color_token: state.preflight.normalized_metadata.color_token ?? "",
                group: state.preflight.normalized_metadata.group ?? null,
              } : domainChoice,
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
            className="model-upload-sheet__choice model-upload-sheet__choice--source"
            onClick={() => handleSourceSelection("local")}
            data-testid="upload-source-local"
          >
            <strong>Local computer</strong>
            <span
              className="model-upload-sheet__choice-icon-shell model-upload-sheet__choice-icon-shell--local"
              aria-hidden="true"
            >
              <LocalSourceIcon />
            </span>
          </button>
          <button
            type="button"
            className="model-upload-sheet__choice model-upload-sheet__choice--source"
            onClick={() => handleSourceSelection("huggingface")}
            data-testid="upload-source-hf"
          >
            <strong>Hugging Face</strong>
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
                  {domain.display_name}
                </option>
              ))}
            </select>
            {fieldErrorFor(state, "domain") ? <small>{fieldErrorFor(state, "domain")}</small> : null}
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
      <div className="model-upload-sheet__section-title-row">
        <div className="panel__eyebrow">Model Name &amp; Task</div>
        <span className="model-upload-sheet__section-title-note">Required</span>
      </div>
      <div className="model-upload-sheet__subsection">
        <div className="model-upload-sheet__grid">
        <label
          htmlFor={fieldId("display-name")}
          className={`field-shell field-shell--compact${fieldErrorFor(state, "display_name") ? " field-shell--error" : ""}`}
        >
          <span className="field-shell__label-row">
            <span>Display name</span>
            <span
              className="field-info-badge"
              tabIndex={0}
              role="note"
              aria-label="The primary name shown in management and dashboards."
              data-tooltip="The primary name shown in management and dashboards."
            >
              i
            </span>
          </span>
          <input
            id={fieldId("display-name")}
            aria-label="Display name"
            value={state.metadata.display_name}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "display_name", value: event.target.value })
            }
            placeholder="DistilBERT 60k"
          />
          {fieldErrorFor(state, "display_name") ? <small>{fieldErrorFor(state, "display_name")}</small> : null}
        </label>

        <label
          htmlFor={fieldId("model-id")}
          className={`field-shell field-shell--compact${fieldErrorFor(state, "model_id") ? " field-shell--error" : ""}`}
        >
          <span className="field-shell__label-row">
            <span>Model id</span>
            <span
              className="field-info-badge field-info-badge--wide"
              tabIndex={0}
              role="note"
              aria-label="Unique registry id used for routes, manifests, and storage."
              data-tooltip="Unique registry id used for routes, manifests, and storage."
            >
              i
            </span>
          </span>
          <div className="model-upload-sheet__input-row">
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
              className="mini-button model-upload-sheet__inline-action"
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
          {fieldErrorFor(state, "model_id") ? <small>{fieldErrorFor(state, "model_id")}</small> : null}
        </label>

        <label
          htmlFor={fieldId("framework-task")}
          className={`field-shell${fieldErrorFor(state, "framework_task") ? " field-shell--error" : ""}`}
        >
          <span className="field-shell__label-row">
            <span>Task</span>
            {state.metadata.framework_task !== "sequence-classification"
              ? renderWarningBadge(
                  "This task selector is future-facing for now. Only Sequence Classification is currently runtime-supported.",
                )
              : null}
          </span>
          <select
            id={fieldId("framework-task")}
            value={state.metadata.framework_task}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "framework_task", value: event.target.value })
            }
          >
            {TASK_OPTIONS.map((task) => (
              <option key={task.value} value={task.value}>
                {task.label}
              </option>
            ))}
          </select>
          {fieldErrorFor(state, "framework_task") ? <small>{fieldErrorFor(state, "framework_task")}</small> : null}
        </label>

        {branch !== "huggingface" ? (
          <label
            htmlFor={fieldId("framework-type")}
            className={`field-shell${fieldErrorFor(state, "framework_type") ? " field-shell--error" : ""}`}
          >
            <span className="field-shell__label-row">
              <span>Model type</span>
              {state.metadata.framework_type !== "transformers"
                ? renderWarningBadge(
                    "You can upload this model, but it will not be runtime-compatible yet. Only transformer-based encoder models are currently supported for activation and live analysis.",
                  )
                : null}
            </span>
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
            {fieldErrorFor(state, "framework_type") ? <small>{fieldErrorFor(state, "framework_type")}</small> : null}
          </label>
        ) : null}
        </div>
      </div>
    </div>
  );

  const renderLabelsSection = () => (
    <div className="model-upload-sheet__section">
      <div className="panel__eyebrow">Labels</div>
      <div className="model-upload-sheet__subsection">
        <div className="model-upload-sheet__grid">
          <div className={`field-shell field-shell--wide${fieldErrorFor(state, "labels") ? " field-shell--error" : ""}`}>
            <div className="model-upload-sheet__label-mode-row">
              <span className="model-upload-sheet__label-mode-copy">
                How do you want to define the labels?
              </span>
              <div className="model-upload-sheet__toggle-row" role="group" aria-label="Label input mode">
                <button
                  type="button"
                  className={`mini-button${state.labelInputMode === "manual" ? " mini-button--active" : ""}`}
                  onClick={() => dispatch({ type: "set-label-input-mode", mode: "manual" })}
                  data-testid="label-input-mode-manual"
                >
                  Manual
                </button>
                {canUploadLabelMapFile ? (
                  <button
                    type="button"
                    className={`mini-button${state.labelInputMode === "file" ? " mini-button--active" : ""}`}
                    onClick={() => dispatch({ type: "set-label-input-mode", mode: "file" })}
                    data-testid="label-input-mode-file"
                  >
                    Upload file
                  </button>
                ) : null}
              </div>
            </div>

            {state.labelInputMode === "file" ? (
              <div
                className={`field-shell field-shell--nested${
                  fieldErrorFor(state, "artifacts.label_map_file") ? " field-shell--error" : ""
                }`}
              >
                <label htmlFor={fieldId("runtime-label-map-file")} className="field-shell__nested-control">
                  {renderInfoLabel(
                    "Label map file",
                    "Upload a JSON label map to prefill names, or keep a PKL file as an artifact for runtime use.",
                  )}
                  <input
                    id={fieldId("runtime-label-map-file")}
                    aria-label="Label map file"
                    type="file"
                    accept=".json,.pkl"
                    multiple={false}
                    onChange={(event) =>
                      void handleLabelMapFilesChange(Array.from(event.target.files ?? []))
                    }
                  />
                </label>
                <small>
                  {fieldErrorFor(state, "artifacts.label_map_file")
                    ?? "JSON files auto-fill labels when possible. PKL files are kept as uploaded artifacts."}
                </small>
                <div className="artifact-slot__list artifact-slot__list--preview">
                  {(state.artifactFiles.label_map_file ?? []).length > 0 ? (
                    (state.artifactFiles.label_map_file ?? []).map((file) => (
                      <span key={`label-map-${file.name}`}>{file.name}</span>
                    ))
                  ) : (
                    <span>No label map file selected.</span>
                  )}
                </div>
                <div className="artifact-slot__list artifact-slot__list--preview">
                  {state.metadata.labels.map((label) => (
                    <span key={`label-preview-${label.id}`}>
                      {label.display_name || label.name}
                    </span>
                  ))}
                </div>
              </div>
            ) : (
              <>
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
                  {fieldErrorFor(state, "labels") ? <span>{fieldErrorFor(state, "labels")}</span> : null}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  const renderOptionalDetailsSection = () => (
    <div className="model-upload-sheet__section">
      <details
        className="model-upload-sheet__subsection model-upload-sheet__subsection--collapsible"
        open={optionalDetailsOpen}
        onToggle={(event) => setOptionalDetailsOpen((event.currentTarget as HTMLDetailsElement).open)}
      >
        <summary className="model-upload-sheet__subsection-summary">
          <span className="model-upload-sheet__subsection-header model-upload-sheet__subsection-header--summary-only">
            <small>Advanced metadata and runtime settings</small>
          </span>
          <span className="model-upload-sheet__subsection-chevron" aria-hidden="true" />
        </summary>
        <div className="model-upload-sheet__grid">
        <label
          htmlFor={fieldId("runtime-max-sequence-length")}
          className={`field-shell${fieldErrorFor(state, "runtime_max_sequence_length") ? " field-shell--error" : ""}`}
        >
          {renderInfoLabel("Max sequence length", "Used during tokenization to truncate or pad input.")}
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
          {fieldErrorFor(state, "runtime_max_sequence_length") ? (
            <small>{fieldErrorFor(state, "runtime_max_sequence_length")}</small>
          ) : null}
        </label>

        <label
          htmlFor={fieldId("runtime-batch-size")}
          className={`field-shell${fieldErrorFor(state, "runtime_batch_size") ? " field-shell--error" : ""}`}
        >
          {renderInfoLabel("Batch size", "Reserved for future functionality.")}
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
          {fieldErrorFor(state, "runtime_batch_size") ? <small>{fieldErrorFor(state, "runtime_batch_size")}</small> : null}
        </label>

        <label htmlFor={fieldId("description")} className="field-shell field-shell--wide">
          <span className="field-shell__label-row">
            <span>Description</span>
            <span
              className="field-info-badge"
              tabIndex={0}
              role="note"
              aria-label="Optional context for reviewers and future maintainers."
              data-tooltip="Optional context for reviewers and future maintainers."
            >
              i
            </span>
          </span>
          <textarea
            className="field-shell__textarea--compact"
            id={fieldId("description")}
            value={state.metadata.description ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "description", value: event.target.value })
            }
            placeholder="Short technical description"
          />
        </label>

        <label htmlFor={fieldId("version")} className="field-shell">
          {renderInfoLabel("Version", "Optional release or export tag.")}
          <input
            id={fieldId("version")}
            value={state.metadata.version ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "version", value: event.target.value })
            }
            placeholder="v1"
          />
        </label>

        <label htmlFor={fieldId("framework-library")} className="field-shell">
          {renderInfoLabel("Framework library", "Defaults to the current runtime library for the selected model type.")}
          <input
            id={fieldId("framework-library")}
            value={state.metadata.framework_library ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "framework_library", value: event.target.value })
            }
            placeholder={frameworkLibraryDefault(state.metadata.framework_type)}
          />
        </label>

        <label htmlFor={fieldId("architecture")} className="field-shell">
          {renderInfoLabel("Architecture", "Important for runtime compatibility and review.")}
          <input
            id={fieldId("architecture")}
            value={state.metadata.architecture ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "architecture", value: event.target.value })
            }
            placeholder="DistilBertForSequenceClassification"
          />
        </label>

        <label htmlFor={fieldId("backbone")} className="field-shell">
          {renderInfoLabel("Backbone", "Used when the model family matters for review and import tracking.")}
          <input
            id={fieldId("backbone")}
            value={state.metadata.backbone ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "backbone", value: event.target.value })
            }
            placeholder="distilbert-base-uncased"
          />
        </label>

        <label htmlFor={fieldId("base-model")} className="field-shell">
          {renderInfoLabel("Base model", "Optional but helpful when the export and the training backbone differ.")}
          <input
            id={fieldId("base-model")}
            value={state.metadata.base_model ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "base_model", value: event.target.value })
            }
            placeholder="bert-base-uncased"
          />
        </label>

        <label htmlFor={fieldId("embeddings")} className="field-shell">
          {renderInfoLabel("Embeddings", "Mostly relevant for custom deep learning and classic ML flows.")}
          <input
            id={fieldId("embeddings")}
            value={state.metadata.embeddings ?? ""}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "embeddings", value: event.target.value })
            }
            placeholder="fasttext-wiki-news-subwords-300"
          />
        </label>

        <label htmlFor={fieldId("runtime-device")} className="field-shell">
          {renderInfoLabel("Runtime device", "Matches the existing manifest structure used in the registry.")}
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
        </label>

        <label htmlFor={fieldId("runtime-padding")} className="field-shell">
          {renderInfoLabel("Padding", "The saved manifest keeps this exact runtime padding mode.")}
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
        </label>

        <label htmlFor={fieldId("runtime-truncation")} className="toggle-field field-shell">
          <div className="toggle-field__copy">
            {renderInfoLabel("Enable truncation", "Stored in the runtime block and shown in review.")}
          </div>
          <div className="toggle-field__control">
            <input
              id={fieldId("runtime-truncation")}
              className="toggle-field__input"
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
            <span className="toggle-field__switch" aria-hidden="true" />
            <span className="toggle-field__state">{state.metadata.runtime_truncation ? "On" : "Off"}</span>
          </div>
        </label>

        <label htmlFor={fieldId("enable-on-upload")} className="toggle-field field-shell">
          <div className="toggle-field__copy">
            {renderInfoLabel(
              "Enable after registration",
              "Activation is still blocked if runtime compatibility checks fail.",
            )}
          </div>
          <div className="toggle-field__control">
            <input
              id={fieldId("enable-on-upload")}
              className="toggle-field__input"
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
            <span className="toggle-field__switch" aria-hidden="true" />
            <span className="toggle-field__state">{state.metadata.enable_on_upload ? "On" : "Off"}</span>
          </div>
        </label>

        <label htmlFor={fieldId("runtime-preprocessing")} className="field-shell field-shell--wide">
          {renderInfoLabel("Preprocessing", "Especially useful for custom PyTorch and classic ML registrations.")}
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
        </label>

        <label
          htmlFor={fieldId("model-config-text")}
          className={`field-shell field-shell--wide${fieldErrorFor(state, "model_config_text") ? " field-shell--error" : ""}`}
        >
          {renderInfoLabel("Model config payload (JSON)", "This is stored under the `model` block in the saved manifest.")}
          <textarea
            id={fieldId("model-config-text")}
            value={state.metadata.model_config_text}
            onChange={(event) =>
              dispatch({ type: "set-metadata", key: "model_config_text", value: event.target.value })
            }
            placeholder='{"hidden_dim": 128, "num_layers": 2}'
          />
          {fieldErrorFor(state, "model_config_text") ? <small>{fieldErrorFor(state, "model_config_text")}</small> : null}
        </label>
        </div>
      </details>
    </div>
  );

  const renderRegistrationConfigSection = () => (
    <div className="model-upload-sheet__section">
      <div className="panel__eyebrow">Config Upload</div>

      <label
        htmlFor={fieldId("registration-config")}
        className={`field-shell${fieldErrorFor(state, "registration_config") ? " field-shell--error" : ""}`}
      >
        <input
          id={fieldId("registration-config")}
          aria-label="Registration config file"
          type="file"
          accept=".yaml,.yml,.json"
          multiple={false}
          onChange={(event) =>
            void handleRegistrationConfigFilesChange(Array.from(event.target.files ?? []))
          }
        />
        <small>
          {fieldErrorFor(state, "registration_config")
            ?? "Upload the saved manifest you already have (YAML or JSON). We’ll use it to drive the remaining upload flow."}
        </small>
        <div className="artifact-slot__list">
          {state.registrationConfigFiles.length > 0 ? (
            state.registrationConfigFiles.map((file) => <span key={file.name}>{file.name}</span>)
          ) : (
            <span>No config file selected.</span>
          )}
        </div>
      </label>

      <details className="model-upload-sheet__example-config">
        <summary>View config example</summary>
        <pre>{exampleConfigTemplate(currentFramework)}</pre>
      </details>
    </div>
  );

  const renderLocalConfigQuestionSection = () => (
    <div className="model-upload-sheet__section model-upload-sheet__section--source">
      <h3 className="model-upload-sheet__source-question">
        Do you already have a ready upload config file?
      </h3>
      <div className="model-upload-sheet__choice-grid model-upload-sheet__choice-grid--source">
        <button
          type="button"
          className="model-upload-sheet__choice model-upload-sheet__choice--source"
          onClick={() => handleLocalConfigModeSelection("uploaded-config")}
          data-testid="local-config-mode-uploaded"
        >
          <strong>Yes</strong>
          <span
            className="model-upload-sheet__choice-icon-shell model-upload-sheet__choice-icon-shell--config-ready"
            aria-hidden="true"
          >
            <ReadyConfigIcon />
          </span>
        </button>
        <button
          type="button"
          className="model-upload-sheet__choice model-upload-sheet__choice--source"
          onClick={() => handleLocalConfigModeSelection("manual-metadata")}
          data-testid="local-config-mode-manual"
        >
          <strong>No</strong>
          <span
            className="model-upload-sheet__choice-icon-shell model-upload-sheet__choice-icon-shell--manual-entry"
            aria-hidden="true"
          >
            <ManualMetadataIcon />
          </span>
        </button>
      </div>
      {fieldErrorFor(state, "local_config_mode") ? (
        <small className="field-error">{fieldErrorFor(state, "local_config_mode")}</small>
      ) : null}
    </div>
  );

  const renderDetailsStep = () => {
    const detailsContentClassName =
      branch === "local" && state.localDetailsStage === "question"
        ? "model-upload-sheet__content model-upload-sheet__content--source"
        : "model-upload-sheet__content model-upload-sheet__content--details";

    if (branch === "local") {
      if (state.localDetailsStage === "question") {
        return (
          <div className={detailsContentClassName}>
            {renderLocalConfigQuestionSection()}
          </div>
        );
      }

      return (
        <div className={detailsContentClassName}>
          {state.localConfigMode === "uploaded-config" ? renderRegistrationConfigSection() : null}
          {state.localConfigMode === "manual-metadata" ? (
            <>
              {renderDomainSection()}
              {renderMetadataSection()}
              {renderLabelsSection()}
              {renderOptionalDetailsSection()}
            </>
          ) : null}
        </div>
      );
    }

    return (
      <div className={detailsContentClassName}>
        {renderDomainSection()}
        {renderMetadataSection()}
        {renderLabelsSection()}
        {renderOptionalDetailsSection()}
      </div>
    );
  };

  const renderArtifactUploadState = (fileCount: number, emptyLabel: string) =>
    fileCount === 0 ? (
      <div className="artifact-slot__upload-state-row">
        <span className="artifact-slot__upload-state">{emptyLabel}</span>
      </div>
    ) : null;

  const renderArtifactHint = (requirement: ArtifactRequirement) => (
    <div className="artifact-slot__hint">
      <p>{requirement.hint}</p>
      {requirement.hintGroups?.length ? (
        <div className="artifact-slot__hint-groups">
          {requirement.hintGroups.map((group) => (
            <div key={`${requirement.slot}-${group.label}`} className="artifact-slot__hint-group">
              <strong>{group.label}</strong>
              <div className="artifact-slot__hint-items">
                {group.items.map((item) => (
                  <span key={`${requirement.slot}-${group.label}-${item}`}>{item}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );

  const renderLocalValidateStep = () => {
    return (
      <div className="model-upload-sheet__content">
        <div className="model-upload-sheet__section model-upload-sheet__section--artifacts">
          <div className="model-upload-sheet__grid model-upload-sheet__artifact-grid">
            {visibleArtifactRequirements.map((requirement) => {
              const files = state.artifactFiles[requirement.slot] ?? [];
              const artifactError = fieldErrorFor(state, `artifacts.${requirement.slot}`);
              const visualKind = resolveArtifactVisualKind(requirement.slot);
              return (
                <label
                  key={requirement.slot}
                  htmlFor={fieldId(`artifact-${requirement.slot}`)}
                  className={`artifact-slot artifact-slot--compact${
                    artifactError ? " artifact-slot--error" : ""
                  }`}
                >
                  <div className="artifact-slot__header">
                    <div className="artifact-slot__header-main">
                      <span
                        className={`artifact-slot__icon-shell artifact-slot__icon-shell--${visualKind}`}
                      >
                        <ArtifactCardIcon kind={visualKind} />
                      </span>
                      <strong>{requirement.title}</strong>
                    </div>
                    <span>{requirement.required ? "Required" : "Optional"}</span>
                  </div>
                  {renderArtifactHint(requirement)}
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
                  {renderArtifactUploadState(files.length, "No files uploaded yet")}
                  {files.length > 0 ? (
                    <div className="artifact-slot__list artifact-slot__list--selected">
                      {files.map((file) => (
                        <span key={`${requirement.slot}-${file.name}`}>{file.name}</span>
                      ))}
                    </div>
                  ) : null}
                  {artifactError ? <small className="field-error">{artifactError}</small> : null}
                </label>
              );
            })}
            {(() => {
              const dashboardError = fieldErrorFor(state, "dashboard");
              return (
            <label
              htmlFor={fieldId("dashboard-bundle")}
              className={`artifact-slot artifact-slot--compact${
                dashboardError ? " artifact-slot--error" : ""
              }`}
            >
              <div className="artifact-slot__header">
                <div className="artifact-slot__header-main">
                  <span className="artifact-slot__icon-shell artifact-slot__icon-shell--dashboard">
                    <ArtifactCardIcon kind="dashboard" />
                  </span>
                  <strong>Dashboard Bundle</strong>
                </div>
                <span>Optional</span>
              </div>
              <p>e.g. dashboard-manifest.json, index.html</p>
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
              {renderArtifactUploadState(
                state.dashboardFiles.length,
                "No dashboard bundle uploaded yet",
              )}
              {state.dashboardFiles.length > 0 ? (
                <div className="artifact-slot__list artifact-slot__list--selected">
                  {state.dashboardFiles.slice(0, 6).map((file) => (
                    <span key={normalizeDashboardRelativePath(file)}>
                      {normalizeDashboardRelativePath(file)}
                    </span>
                  ))}
                </div>
              ) : null}
              {dashboardError ? <small className="field-error">{dashboardError}</small> : null}
            </label>
              );
            })()}
          </div>
        </div>
      </div>
    );
  };

  const renderHuggingFaceValidateStep = () => {
    const requiredFiles = huggingFacePreflight?.required_files.filter((file) => file.required) ?? [];
    const availableRequiredFiles = requiredFiles.filter((file) => file.available).length;
    const fileSummary = huggingFacePreflight
      ? `${availableRequiredFiles}/${requiredFiles.length || huggingFacePreflight.required_files.length} required files found`
      : isHuggingFaceValidationRunning
        ? "Checking required files..."
        : "Not checked yet";
    const sizeSummary = huggingFacePreflight
      ? formatBytes(huggingFacePreflight.estimated_download_size_bytes)
      : isHuggingFaceValidationRunning
        ? "Estimating..."
        : "Will appear after preflight";
    const preflightHighlights = [
      {
        label: "Files",
        value: fileSummary,
        pending: !huggingFacePreflight,
      },
      {
        label: "Size",
        value: sizeSummary,
        pending: !huggingFacePreflight,
      },
    ];
    return (
      <div className="model-upload-sheet__content model-upload-sheet__content--hf-validate">
        <div className="model-upload-sheet__hf-validate-shell">
          <div className="model-upload-sheet__section model-upload-sheet__section--hf-repo">
            <div className="model-upload-sheet__hf-repo-card">
              <div className="model-upload-sheet__hf-repo-topline">
                <div className="model-upload-sheet__hf-repo-lead">
                  <span className="model-upload-sheet__hf-repo-icon-shell" aria-hidden="true">
                    <img
                      src={huggingFaceLogo}
                      alt=""
                      className="model-upload-sheet__hf-repo-icon-image"
                    />
                  </span>
                  <div className="model-upload-sheet__hf-repo-copy">
                    <div className="panel__eyebrow">Hugging Face Repo</div>
                  </div>
                </div>
                {huggingFacePreflightState === "success" ? (
                  <span className="model-upload-sheet__hf-preflight-badge model-upload-sheet__hf-preflight-badge--success">
                    Ready
                  </span>
                ) : null}
              </div>

              <div className="model-upload-sheet__hf-repo-main">
                <label
                  htmlFor={fieldId("hf-repo")}
                  className={`field-shell field-shell--compact model-upload-sheet__hf-repo-field${
                    fieldErrorFor(state, "hf_repo") ? " field-shell--error" : ""
                  }`}
                >
                  <span>Repo URL or repo ID</span>
                <input
                  id={fieldId("hf-repo")}
                  aria-label="Repo URL or repo id"
                  value={state.hfRepoInput}
                  onChange={(event) => dispatch({ type: "set-hf-repo-input", value: event.target.value })}
                  placeholder="org/model-name or https://huggingface.co/org/model-name"
                  data-testid="hf-repo-input"
                />
                {fieldErrorFor(state, "hf_repo") ? (
                  <small>{fieldErrorFor(state, "hf_repo")}</small>
                ) : null}
                </label>

                <div className="model-upload-sheet__hf-preflight-highlights model-upload-sheet__hf-preflight-highlights--inline">
                  {preflightHighlights.map((fact) => (
                    <div key={fact.label} className="model-upload-sheet__hf-preflight-highlight">
                      <span>{fact.label}</span>
                      <strong
                        className={
                          fact.pending ? "model-upload-sheet__hf-preflight-highlight-value--pending" : undefined
                        }
                      >
                        {fact.value}
                      </strong>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {huggingFaceWarnings.length > 0 ? (
            <div
              className="inline-alert inline-alert--warning model-upload-sheet__hf-preflight-warning model-upload-sheet__hf-preflight-warning-bar"
              role="status"
            >
              {huggingFaceWarnings.join(" ")}
            </div>
          ) : null}
        </div>
      </div>
    );
  };

  const renderValidateStep = () =>
    branch === "huggingface" ? renderHuggingFaceValidateStep() : renderLocalValidateStep();

  const renderReviewStep = () => {
    const metadata = state.preflight?.normalized_metadata ?? currentMetadata;
    if (!metadata || !branch) {
      return null;
    }

    const warnings = branch === "huggingface" ? [] : state.preflight?.warnings ?? [];
    const configSourceLabel =
      isLocalPreflight(state.preflight) && state.preflight.config_source === "uploaded"
        ? "Uploaded by user"
        : "Generated by system";
    const reviewCards = [
      {
        key: "model",
        title: "Model",
        accentClassName: "model-upload-sheet__review-card--model",
        facts: [
          { label: "Display name", value: metadata.display_name },
          { label: "Domain", value: `${metadata.ui_display_name} (${metadata.domain})` },
          { label: "Model id", value: metadata.model_id },
        ],
      },
      {
        key: "runtime",
        title: "Runtime",
        accentClassName: "model-upload-sheet__review-card--runtime",
        facts: [
          { label: "Model type", value: frameworkLabel(metadata.framework_type) },
          {
            label: "Runtime",
            value: `${metadata.runtime_device} · max ${metadata.runtime_max_sequence_length} · batch ${metadata.runtime_batch_size}`,
          },
          { label: "Labels", value: `${metadata.labels.length} configured` },
        ],
      },
      {
        key: "registration",
        title: "Registration",
        accentClassName: "model-upload-sheet__review-card--registration",
        facts: [
          { label: "Source", value: state.source === "huggingface" ? "Hugging Face" : "Local computer" },
          { label: "Upload mode", value: branchHeading(branch) },
          { label: "Config source", value: configSourceLabel },
        ],
      },
      {
        key: "packaging",
        title: "Packaging",
        accentClassName: "model-upload-sheet__review-card--packaging",
        facts: [
          { label: "Dashboard", value: state.dashboardFiles.length > 0 ? "Attached" : "Not attached" },
          {
            label: branch === "huggingface" ? "Repo" : "Artifacts",
            value:
              branch === "huggingface"
                ? state.hfRepoInput.trim()
                : `${Object.values(state.artifactFiles).flat().length} files selected`,
          },
        ],
      },
    ] as const;
    return (
      <div className="model-upload-sheet__content model-upload-sheet__content--review" data-testid="review-step">
        <div className="model-upload-sheet__review-grid">
          {reviewCards.map((card) => (
            <section
              key={card.key}
              className={`model-upload-sheet__review-card ${card.accentClassName}`}
            >
              <div className="model-upload-sheet__review-card-header">
                <span className="model-upload-sheet__review-card-kicker">{card.title}</span>
              </div>
              <div className="model-upload-sheet__review-facts">
                {card.facts.map((fact) => (
                  <div key={`${card.key}-${fact.label}`} className="model-upload-sheet__review-fact">
                    <span>{fact.label}</span>
                    <strong>{fact.value}</strong>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>

        {warnings.length > 0 ? (
          <div className="inline-alert">
            {warnings.join(" ")}
          </div>
        ) : null}

        <section
          className="model-upload-sheet__example-config model-upload-sheet__example-config--review"
          data-testid="review-config-preview"
        >
          <button
            type="button"
            className="model-upload-sheet__example-config-toggle model-upload-sheet__example-config-toggle--dialog"
            aria-haspopup="dialog"
            aria-expanded={reviewConfigOpen}
            onClick={() => setReviewConfigOpen(true)}
          >
            <span className="model-upload-sheet__example-config-copy">
              <strong>Config Preview</strong>
              <span>Open generated config</span>
            </span>
          </button>
        </section>
      </div>
    );
  };

  const renderResultStep = () => {
    if (!state.result) {
      return null;
    }
    const { result } = state.result;
    return (
      <div className="model-upload-sheet__content model-upload-sheet__content--result" data-testid="result-step">
        <section className="model-upload-sheet__success-card">
          <span className="model-upload-sheet__success-icon-shell" aria-hidden="true">
            <UploadSuccessIcon />
          </span>
          <div className="model-upload-sheet__success-copy">
            <h3>Model uploaded successfully</h3>
            <p>{result.display_name} is now available in Model Management section.</p>
          </div>
        </section>
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
    return renderResultStep();
  };

  const renderFooter = () => {
    if (state.step === "source") {
      return null;
    }
    if (state.step === "details") {
      if (branch === "local" && state.localDetailsStage === "question") {
        return (
          <>
            <button type="button" className="mini-button" onClick={handleDetailsBack}>
              Back
            </button>
          </>
        );
      }
      return (
        <>
          <button type="button" className="mini-button" onClick={handleDetailsBack}>
            Back
          </button>
          <button type="button" className="primary-button" onClick={handleDetailsNext}>
            Continue
          </button>
        </>
      );
    }
    if (state.step === "validate") {
      const huggingFacePrimaryLabel = isHuggingFaceValidationRunning
        ? "Running Preflight..."
        : huggingFaceCanAdvanceToReview
          ? "Final Review"
          : "Run Preflight";
      const handleValidatePrimaryAction =
        branch === "huggingface" && huggingFaceCanAdvanceToReview
          ? () => dispatch({ type: "set-step", step: "review" })
          : () => void handleRunValidation();
      return (
        <>
          {validateStatus ? (
            <div
              className={`model-upload__status model-upload__status--${validateStatus.tone}${
                validateStatus.detail ? "" : " model-upload__status--compact"
              }`}
              data-testid="validate-status"
              role={validateStatus.tone === "error" ? "alert" : "status"}
              aria-live="polite"
            >
              <span className="model-upload__status-indicator" aria-hidden="true" />
              <div className="model-upload__status-copy">
                <strong>{validateStatus.title}</strong>
                {validateStatus.detail ? <span>{validateStatus.detail}</span> : null}
              </div>
            </div>
          ) : null}
          <button
            type="button"
            className="mini-button"
            onClick={() => dispatch({ type: "set-step", step: "details" })}
            disabled={branch === "huggingface" && isHuggingFaceValidationRunning}
          >
            Back
          </button>
          <button
            type="button"
            className="primary-button"
            onClick={handleValidatePrimaryAction}
            disabled={branch === "huggingface" && isHuggingFaceValidationRunning}
          >
            {branch === "huggingface" ? huggingFacePrimaryLabel : "Validate files"}
          </button>
        </>
      );
    }
    if (state.step === "review") {
      return (
        <>
          {isSubmitting ? (
            <span
              className="model-upload__spinner-indicator"
              data-testid="upload-running-indicator"
              role="status"
              aria-live="polite"
              aria-label={branch === "huggingface" ? "Importing model" : "Uploading model"}
            >
              <span className="model-upload__spinner-dot" aria-hidden="true" />
            </span>
          ) : null}
          <button
            type="button"
            className="mini-button"
            onClick={() => dispatch({ type: "set-step", step: "validate" })}
            disabled={isSubmitting}
          >
            Back
          </button>
          <button
            type="button"
            className="primary-button"
            onClick={() => void handleSubmit()}
            data-testid="review-submit"
            disabled={isSubmitting}
          >
            {isSubmitting
              ? branch === "huggingface"
                ? "Importing..."
                : "Uploading..."
              : branch === "huggingface"
                ? "Import model"
                : "Upload model"}
          </button>
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

  const footerContent = renderFooter();
  const hasFloatingFooter =
    state.step === "details" && branch === "local" && state.localDetailsStage === "question";

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
              return (
                <div
                  key={step.id}
                  className={`model-upload-sheet__step${isActive ? " model-upload-sheet__step--active" : ""}`}
                >
                  <span>{String(index + 1).padStart(2, "0")}</span>
                  <strong>{step.label}</strong>
                </div>
              );
            })}
          </div>
        </aside>

        <div ref={mainPanelRef} className="model-upload-sheet__main">
          <header className="model-upload-sheet__header">
            <div>
              {state.step === "details" || state.step === "source" ? null : (
                <div className="model-upload-sheet__header-title">
                  <h3>{STEP_DEFS.find((step) => step.id === state.step)?.label}</h3>
                  {state.step === "validate" ? (
                    <span
                      className="field-info-badge field-info-badge--wide"
                      tabIndex={0}
                      role="note"
                      aria-label={localUploadTooltip}
                      data-tooltip={localUploadTooltip}
                      data-testid="model-artifacts-info-badge"
                    >
                      i
                    </span>
                  ) : null}
                </div>
              )}
            </div>
            <button type="button" className="mini-button" onClick={closeWizard} disabled={!canClose}>
              Close
            </button>
          </header>

          {state.message && state.step !== "source" && state.step !== "details" && state.step !== "validate" ? (
            <div className="inline-alert">{state.message}</div>
          ) : null}

          {renderBody()}

          {footerContent ? (
            <footer
              className={`model-upload__actions${
                hasFloatingFooter ? " model-upload__actions--floating" : ""
              }`}
            >
              {footerContent}
            </footer>
          ) : null}
        </div>
      </section>
      {reviewConfigOpen ? (
        <div
          className="artifact-modal model-upload-sheet__config-modal-overlay"
          data-testid="review-config-modal"
          onClick={(event) => {
            event.stopPropagation();
            setReviewConfigOpen(false);
          }}
        >
          <div
            className="artifact-modal__card model-info-modal model-upload-sheet__config-modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby={fieldId("review-config-modal-title")}
            aria-describedby={fieldId("review-config-modal-body")}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="artifact-modal__header">
              <h3 id={fieldId("review-config-modal-title")}>Config Preview</h3>
              <button
                type="button"
                className="mini-button"
                onClick={() => setReviewConfigOpen(false)}
              >
                Close
              </button>
            </div>
            <pre
              id={fieldId("review-config-modal-body")}
              className="model-upload-sheet__config-modal-body"
            >
              {state.preflight?.config_preview}
            </pre>
          </div>
        </div>
      ) : null}
    </div>
  );
}
