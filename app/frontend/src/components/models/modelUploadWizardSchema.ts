import type {
  LocalUploadPreflightRequest,
  UploadFileDescriptor,
  UploadLabelClass,
  UploadModelMetadata,
} from "../../types/contracts";

export type WizardStep =
  | "source"
  | "details"
  | "validate"
  | "review"
  | "progress"
  | "result";

export type UploadSource = "local" | "huggingface";
export type BranchMode = "local" | "huggingface";
export type FrameworkType = UploadModelMetadata["framework_type"];

export interface ArtifactRequirement {
  slot: string;
  title: string;
  required: boolean;
  accepts: string;
  hint: string;
}

export interface DomainChoice {
  domain: string;
  display_name: string;
  color_token: string;
  group?: string | null;
}

export interface WizardMetadataDraft
  extends Omit<UploadModelMetadata, "runtime_padding" | "model_config"> {
  model_config_text: string;
  runtime_padding_mode: "true" | "false" | "max_length";
}

export const ARTIFACT_REQUIREMENTS: Record<FrameworkType, ArtifactRequirement[]> = {
  transformers: [
    {
      slot: "weights",
      title: "Weights",
      required: true,
      accepts: ".safetensors,.bin,.pt,.pth",
      hint: "Primary checkpoint weights used by the transformer runtime.",
    },
    {
      slot: "tokenizer",
      title: "Tokenizer Assets",
      required: true,
      accepts: ".json,.txt,.model,.vocab",
      hint: "Tokenizer files like tokenizer.json, tokenizer_config.json, merges.txt, or vocab.txt.",
    },
    {
      slot: "config",
      title: "Runtime Config Assets",
      required: true,
      accepts: ".json,.yaml,.yml,.bin",
      hint: "Model runtime files such as config.json or training_args.bin.",
    },
    {
      slot: "label_map_file",
      title: "Label Map",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional encoder or label-id mapping artifact.",
    },
    {
      slot: "label_classes_file",
      title: "Label Classes",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional label classes export used by dashboards or custom runtimes.",
    },
  ],
  pytorch: [
    {
      slot: "weights",
      title: "Weights",
      required: true,
      accepts: ".pt,.pth,.bin",
      hint: "Primary state dict or checkpoint file for the deep learning model.",
    },
    {
      slot: "vocabulary",
      title: "Vocabulary / Preprocessing Assets",
      required: true,
      accepts: ".pkl,.json,.txt",
      hint: "Vocabulary, lookup tables, or preprocessing assets required at runtime.",
    },
    {
      slot: "config",
      title: "Runtime Config Assets",
      required: true,
      accepts: ".json,.yaml,.yml,.bin",
      hint: "Architecture or runtime config assets the model depends on.",
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
      hint: "Optional encoder artifact for class decoding.",
    },
  ],
  sklearn: [
    {
      slot: "weights",
      title: "Serialized Model",
      required: true,
      accepts: ".pkl,.joblib,.bin",
      hint: "Serialized estimator or pipeline artifact.",
    },
    {
      slot: "config",
      title: "Feature / Runtime Config",
      required: true,
      accepts: ".json,.yaml,.yml,.txt",
      hint: "Feature extraction, vectorizer, or runtime configuration files.",
    },
    {
      slot: "label_classes_file",
      title: "Label Classes",
      required: false,
      accepts: ".json,.pkl",
      hint: "Optional label classes export.",
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

export const DEFAULT_LABELS: UploadLabelClass[] = [
  { id: 0, name: "class_0", display_name: "Class 0" },
];

export const TASK_OPTIONS = [
  { value: "sequence-classification", label: "Sequence Classification" },
  { value: "token-classification", label: "NER / Token Classification" },
  { value: "translation", label: "Translation" },
  { value: "text-generation", label: "Generation" },
  { value: "knowledge-graph-construction", label: "Knowledge Graph Construction" },
] as const;

export function createDefaultMetadataDraft(): WizardMetadataDraft {
  return {
    model_id: "",
    domain: "",
    display_name: "",
    description: "",
    version: "",
    enable_on_upload: false,
    framework_type: "transformers",
    framework_task: "sequence-classification",
    framework_library: "huggingface",
    framework_problem_type: "single_label_classification",
    backbone: "",
    architecture: "",
    base_model: "",
    embeddings: "",
    output_type: "single-label-classification",
    runtime_device: "auto",
    runtime_max_sequence_length: 256,
    runtime_batch_size: 1,
    runtime_truncation: true,
    runtime_padding_mode: "true",
    runtime_preprocessing: "",
    ui_display_name: "",
    color_token: "",
    group: "",
    labels: DEFAULT_LABELS.map((label) => ({ ...label })),
    model_config_text: "{}",
  };
}

export function resolveBranchMode(
  source: UploadSource | null,
): BranchMode | null {
  if (source === "huggingface") {
    return "huggingface";
  }
  if (source === "local") {
    return "local";
  }
  return null;
}

export function frameworkLabel(framework: FrameworkType): string {
  if (framework === "transformers") {
    return "Transformer";
  }
  if (framework === "pytorch") {
    return "Deep Learning";
  }
  return "Classic ML";
}

export function frameworkLibraryDefault(framework: FrameworkType): string {
  if (framework === "transformers") {
    return "huggingface";
  }
  if (framework === "pytorch") {
    return "torch";
  }
  return "sklearn";
}

export function canonicalizeSlug(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "");
}

export function buildSuggestedModelId(domain: string, displayName: string): string {
  const domainSlug = canonicalizeSlug(domain);
  const nameSlug = canonicalizeSlug(displayName);
  return [domainSlug, nameSlug].filter(Boolean).join("-").slice(0, 120);
}

export function nextLabelId(labels: UploadLabelClass[]): number {
  return labels.reduce((current, label) => Math.max(current, label.id), -1) + 1;
}

export function createNextLabel(labels: UploadLabelClass[]): UploadLabelClass {
  const id = nextLabelId(labels);
  return {
    id,
    name: `class_${id}`,
    display_name: `Class ${id}`,
  };
}

export function buildFileDescriptor(
  file: File,
  relativePath?: string,
): UploadFileDescriptor {
  return {
    name: file.name,
    size_bytes: file.size,
    relative_path: relativePath ?? null,
  };
}

export function toUploadMetadata(
  draft: WizardMetadataDraft,
  domain: DomainChoice,
): UploadModelMetadata {
  let parsedConfig: Record<string, unknown> = {};
  if (draft.model_config_text.trim()) {
    parsedConfig = JSON.parse(draft.model_config_text);
  }

  return {
    model_id: draft.model_id.trim(),
    domain: domain.domain,
    display_name: draft.display_name.trim(),
    description: draft.description?.trim() || null,
    version: draft.version?.trim() || null,
    enable_on_upload: draft.enable_on_upload,
    framework_type: draft.framework_type,
    framework_task: draft.framework_task.trim(),
    framework_library: draft.framework_library?.trim() || null,
    framework_problem_type: draft.framework_problem_type?.trim() || null,
    backbone: draft.backbone?.trim() || null,
    architecture: draft.architecture?.trim() || null,
    base_model: draft.base_model?.trim() || null,
    embeddings: draft.embeddings?.trim() || null,
    output_type: draft.output_type?.trim() || null,
    runtime_device: draft.runtime_device,
    runtime_max_sequence_length: draft.runtime_max_sequence_length,
    runtime_batch_size: draft.runtime_batch_size,
    runtime_truncation: draft.runtime_truncation,
    runtime_padding:
      draft.runtime_padding_mode === "max_length"
        ? "max_length"
        : draft.runtime_padding_mode === "false"
          ? false
          : true,
    runtime_preprocessing: draft.runtime_preprocessing?.trim() || null,
    ui_display_name: domain.display_name,
    color_token: domain.color_token,
    group: domain.group ?? null,
    labels: draft.labels,
    model_config: parsedConfig,
  };
}

export function fromUploadMetadata(metadata: UploadModelMetadata): WizardMetadataDraft {
  return {
    model_id: metadata.model_id,
    domain: metadata.domain,
    display_name: metadata.display_name,
    description: metadata.description ?? "",
    version: metadata.version ?? "",
    enable_on_upload: metadata.enable_on_upload,
    framework_type: metadata.framework_type,
    framework_task: metadata.framework_task,
    framework_library: metadata.framework_library ?? frameworkLibraryDefault(metadata.framework_type),
    framework_problem_type: metadata.framework_problem_type ?? "",
    backbone: metadata.backbone ?? "",
    architecture: metadata.architecture ?? "",
    base_model: metadata.base_model ?? "",
    embeddings: metadata.embeddings ?? "",
    output_type: metadata.output_type ?? "single-label-classification",
    runtime_device: metadata.runtime_device,
    runtime_max_sequence_length: metadata.runtime_max_sequence_length,
    runtime_batch_size: metadata.runtime_batch_size,
    runtime_truncation: metadata.runtime_truncation,
    runtime_padding_mode:
      metadata.runtime_padding === "max_length"
        ? "max_length"
        : metadata.runtime_padding === false
          ? "false"
          : "true",
    runtime_preprocessing: metadata.runtime_preprocessing ?? "",
    ui_display_name: metadata.ui_display_name ?? "",
    color_token: metadata.color_token ?? "",
    group: metadata.group ?? "",
    labels: metadata.labels.map((label) => ({ ...label })),
    model_config_text: JSON.stringify(metadata.model_config ?? {}, null, 2),
  };
}

export function buildLocalPreflightPayload(
  draft: WizardMetadataDraft,
  domain: DomainChoice,
  artifactFiles: Record<string, File[]>,
  dashboardFiles: File[],
): LocalUploadPreflightRequest {
  return {
    metadata: toUploadMetadata(draft, domain),
    artifact_manifest: Object.fromEntries(
      Object.entries(artifactFiles)
        .filter(([, files]) => files.length > 0)
        .map(([slot, files]) => [slot, files.map((file) => buildFileDescriptor(file))]),
    ),
    dashboard_manifest: dashboardFiles.map((file) =>
      buildFileDescriptor(file, normalizeDashboardRelativePath(file)),
    ),
  };
}

export function formatBytes(value?: number | null): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Unknown";
  }
  if (value < 1024) {
    return `${value} B`;
  }
  if (value < 1024 * 1024) {
    return `${(value / 1024).toFixed(1)} KB`;
  }
  if (value < 1024 * 1024 * 1024) {
    return `${(value / (1024 * 1024)).toFixed(1)} MB`;
  }
  return `${(value / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

export function normalizeDashboardRelativePath(file: File) {
  const raw = file.webkitRelativePath || file.name;
  return raw.replace(/\\/g, "/");
}

export function parseAcceptedExtensions(accepts: string): string[] {
  return accepts
    .split(",")
    .map((item) => item.trim().toLowerCase())
    .filter(Boolean);
}

export function summarizeDashboardReference(fileRef: string) {
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

export function branchHeading(branch: BranchMode | null): string {
  if (branch === "local") {
    return "Local upload";
  }
  if (branch === "huggingface") {
    return "Import from Hugging Face";
  }
  return "Choose an upload path";
}

export function exampleConfigTemplate(framework: FrameworkType): string {
  const base = {
    model_id: "sentiment-demo",
    domain: "sentiment",
    display_name: "Sentiment Demo",
    framework: {
      type: framework,
      task: "sequence-classification",
      library: frameworkLibraryDefault(framework),
    },
    artifacts:
      framework === "transformers"
        ? {
            weights: ["model.safetensors"],
            tokenizer: ["tokenizer.json", "tokenizer_config.json"],
            config: ["config.json"],
          }
        : framework === "pytorch"
          ? {
              weights: ["model.pt"],
              vocabulary: ["vocab.pkl"],
              config: ["config.json"],
            }
          : {
              weights: ["model.pkl"],
              config: ["features.json"],
            },
    runtime: {
      max_sequence_length: 256,
      truncation: true,
      padding: true,
      batch_size: 1,
      device: "auto",
    },
    labels: {
      type: "single-label-classification",
      classes: [
        { id: 0, name: "negative", display_name: "Negative" },
        { id: 1, name: "positive", display_name: "Positive" },
      ],
    },
    ui: {
      domain_display_name: "Sentiment",
      color_token: "sentiment",
      group: "sentiment-custom",
    },
  };

  return JSON.stringify(base, null, 2);
}
