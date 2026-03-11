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
  ui_display_name?: string | null;
  color_token?: string | null;
  group?: string | null;
  labels: UploadLabelClass[];
  model_config: Record<string, unknown>;
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
