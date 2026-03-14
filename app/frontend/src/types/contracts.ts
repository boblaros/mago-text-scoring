export interface AnalysisRequest {
  text: string;
  domains?: string[];
}

export interface ProbabilityItem {
  label: string;
  score: number;
}

export interface DomainResult {
  domain: string;
  display_name: string;
  predicted_label: string;
  confidence: number;
  probabilities?: ProbabilityItem[];
  model_id: string;
  model_name: string;
  model_version?: string | null;
  latency_ms: number;
  sequence_length_used?: number | null;
  was_truncated?: boolean | null;
}

export interface AggregateResult {
  summary?: string | null;
  mean_confidence?: number | null;
  highest_confidence_domain?: string | null;
}

export interface TextProfile {
  char_count: number;
  word_count: number;
  line_count: number;
}

export interface RoutingOverview {
  mode: "broadcast-active-domains";
  requested_domains: string[];
  resolved_domains: string[];
}

export interface AnalysisResponse {
  request_id: string;
  text_profile: TextProfile;
  routing: RoutingOverview;
  results: DomainResult[];
  aggregate: AggregateResult;
}

export interface DomainCatalogModel {
  model_id: string;
  domain: string;
  display_name: string;
  description?: string | null;
  version?: string | null;
  framework_type: string;
  framework_task?: string | null;
  framework_library?: string | null;
  backbone?: string | null;
  architecture?: string | null;
  runtime_device?: string | null;
  runtime_max_sequence_length?: number | null;
  output_type?: string | null;
  is_active: boolean;
  priority: number;
  notes: string[];
  missing_artifacts: string[];
  status: "ready" | "missing_artifacts" | "incompatible";
  status_reason?: string | null;
  can_activate: boolean;
  dashboard_status: "missing" | "partial" | "available";
  dashboard_sections_available: number;
  dashboard_sections_total: number;
  dashboard_generated_at?: string | null;
}

export interface DomainCatalogEntry {
  domain: string;
  display_name: string;
  color_token: string;
  group?: string | null;
  active_model_id?: string | null;
  active_model_name?: string | null;
  active_model_version?: string | null;
  model_count: number;
  models: DomainCatalogModel[];
}

export interface DomainCatalogResponse {
  domains: DomainCatalogEntry[];
}

export interface CatalogSnapshotResponse {
  active_domains: DomainCatalogEntry[];
  management_domains: DomainCatalogEntry[];
}

export interface DashboardSectionSummary {
  id: string;
  title: string;
  status: "available" | "missing" | "image_only" | "not_applicable";
  description?: string | null;
  reason?: string | null;
  files: string[];
  charts: string[];
}

export interface DashboardSourceItem {
  category: string;
  path: string;
  reason?: string | null;
}

export interface DashboardManifestSummary {
  schema_version: string;
  generated_at?: string | null;
  dashboard_root: string;
  model: Record<string, unknown>;
  entrypoints: Record<string, string>;
  sections: DashboardSectionSummary[];
  selected_sources: DashboardSourceItem[];
  notes: string[];
}

export interface DashboardFigure {
  id: string;
  path: string;
  title?: string | null;
  section_id?: string | null;
  figure: Record<string, unknown>;
}

export interface DashboardImageAsset {
  title: string;
  path: string;
  url: string;
  section_id?: string | null;
}

export interface ModelDashboardResponse {
  model_id: string;
  available: boolean;
  manifest?: DashboardManifestSummary | null;
  overview?: Record<string, unknown> | null;
  source_audit?: Record<string, unknown> | null;
  documents: Record<string, unknown>;
  figures: DashboardFigure[];
  images: DashboardImageAsset[];
}

export interface ModelPatchRequest {
  display_name?: string;
  is_active?: boolean;
}

export interface ModelReorderRequest {
  ordered_model_ids: string[];
}

export interface UploadLabelClass {
  id: number;
  name: string;
  display_name?: string | null;
}

export interface UploadModelMetadata {
  model_id: string;
  domain: string;
  display_name: string;
  description?: string | null;
  version?: string | null;
  enable_on_upload: boolean;
  framework_type: "transformers" | "pytorch" | "sklearn";
  framework_task: string;
  framework_library?: string | null;
  framework_problem_type?: string | null;
  backbone?: string | null;
  architecture?: string | null;
  base_model?: string | null;
  embeddings?: string | null;
  output_type?: string | null;
  runtime_device: string;
  runtime_max_sequence_length: number;
  runtime_batch_size: number;
  runtime_truncation: boolean;
  runtime_padding: boolean | string;
  runtime_preprocessing?: string | null;
  ui_display_name?: string | null;
  color_token?: string | null;
  group?: string | null;
  labels: UploadLabelClass[];
  model_config: Record<string, unknown>;
}

export interface UploadFileDescriptor {
  name: string;
  size_bytes?: number | null;
  relative_path?: string | null;
}

export interface ArtifactValidationSummary {
  slot: string;
  title: string;
  required: boolean;
  valid: boolean;
  message?: string | null;
  files: string[];
}

export interface LocalUploadPreflightRequest {
  registration_mode: "uploaded" | "generated";
  metadata: UploadModelMetadata;
  artifact_manifest: Record<string, UploadFileDescriptor[]>;
  dashboard_manifest: UploadFileDescriptor[];
}

export interface LocalUploadPreflightResponse {
  ready: boolean;
  config_source: "uploaded" | "generated";
  normalized_metadata: UploadModelMetadata;
  config_preview: string;
  artifact_checks: ArtifactValidationSummary[];
  dashboard_attached: boolean;
  warnings: string[];
}

export interface HuggingFacePreflightRequest {
  repo: string;
  metadata: UploadModelMetadata;
}

export interface HuggingFaceArtifactCheck {
  path: string;
  category: string;
  required: boolean;
  available: boolean;
  size_bytes?: number | null;
  message?: string | null;
}

export interface HuggingFacePreflightResponse {
  normalized_repo_id: string;
  repo_url: string;
  detected_framework_type?: string | null;
  detected_task?: string | null;
  framework_library?: string | null;
  architecture?: string | null;
  backbone?: string | null;
  base_model?: string | null;
  estimated_download_size_bytes?: number | null;
  disk_free_bytes: number;
  memory_total_bytes?: number | null;
  memory_estimate_bytes?: number | null;
  runtime_supported: boolean;
  compatible: boolean;
  ready_to_import: boolean;
  required_files: HuggingFaceArtifactCheck[];
  warnings: string[];
  blocking_reasons: string[];
  normalized_metadata: UploadModelMetadata;
  config_preview: string;
}

export interface ModelRegistrationResult {
  model_id: string;
  source: "local" | "huggingface";
  branch: "local-config-upload" | "local-generated-config" | "huggingface";
  config_source: "uploaded" | "generated";
  framework_type: string;
  display_name: string;
  domain: string;
  is_active: boolean;
  status: "ready" | "missing_artifacts" | "incompatible";
  status_reason?: string | null;
  dashboard_status: "missing" | "partial" | "available";
  warnings: string[];
}

export interface ModelRegistrationResponse {
  snapshot: CatalogSnapshotResponse;
  result: ModelRegistrationResult;
}

export type PipelinePhase =
  | "idle"
  | "routing"
  | "running"
  | "revealing"
  | "done"
  | "error";

export type LaneStatus = "idle" | "queued" | "running" | "resolved" | "error";
export type LogStage =
  | "system"
  | "accept"
  | "queued"
  | "dispatch"
  | "resolved"
  | "aggregate"
  | "error";

export interface LaneState {
  domain: string;
  status: LaneStatus;
  result?: DomainResult;
}

export interface LogEntry {
  id: string;
  timestamp: number;
  tone: "neutral" | "info" | "success" | "error";
  stage: LogStage;
  message: string;
  domain?: string;
  latency_ms?: number | null;
}
