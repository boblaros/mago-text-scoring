import type {
  AnalysisRequest,
  AnalysisResponse,
  CatalogSnapshotResponse,
  DomainCatalogResponse,
  HuggingFacePreflightRequest,
  HuggingFacePreflightResponse,
  LocalUploadPreflightRequest,
  LocalUploadPreflightResponse,
  ModelRegistrationResponse,
  ModelDashboardResponse,
  ModelPatchRequest,
  ModelReorderRequest,
} from "../types/contracts";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000/api/v1";

function joinApiUrl(path: string): string {
  const normalizedBase = API_BASE_URL.endsWith("/")
    ? API_BASE_URL.slice(0, -1)
    : API_BASE_URL;
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${normalizedBase}${normalizedPath}`;
}

function encodePathSegments(path: string): string {
  return path
    .split("/")
    .filter(Boolean)
    .map((segment) => encodeURIComponent(segment))
    .join("/");
}

function buildModelDashboardAssetUrl(modelId: string, assetPath: string): string {
  return joinApiUrl(
    `/models/${encodeURIComponent(modelId)}/dashboard/assets/${encodePathSegments(assetPath)}`,
  );
}

export interface ApiValidationDetail {
  message?: string;
  field_errors?: Record<string, string>;
}

export class ApiRequestError extends Error {
  status: number;
  detail?: unknown;

  constructor(message: string, status: number, detail?: unknown) {
    super(message);
    this.name = "ApiRequestError";
    this.status = status;
    this.detail = detail;
  }
}

function extractErrorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") {
    return detail;
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    const message = (detail as ApiValidationDetail).message;
    if (typeof message === "string" && message.trim()) {
      return message;
    }
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const isFormData = init?.body instanceof FormData;
  let response: Response;
  try {
    response = await fetch(joinApiUrl(path), {
      headers: isFormData
        ? init?.headers
        : {
            "Content-Type": "application/json",
            ...(init?.headers ?? {}),
          },
      ...init,
    });
  } catch (error) {
    throw new Error(
      error instanceof Error && error.message === "Failed to fetch"
        ? "Backend API is unreachable. Restart the backend and refresh the page."
        : error instanceof Error
          ? error.message
          : "Backend API is unreachable.",
    );
  }

  if (!response.ok) {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("application/json")) {
      const body = (await response.json()) as { detail?: unknown };
      const detail = body.detail;
      throw new ApiRequestError(
        extractErrorMessage(detail, `Request failed with status ${response.status}`),
        response.status,
        detail,
      );
    }
    const body = await response.text();
    throw new ApiRequestError(
      body || `Request failed with status ${response.status}`,
      response.status,
      body,
    );
  }

  return (await response.json()) as T;
}

export async function fetchDomainCatalog(): Promise<DomainCatalogResponse> {
  return request<DomainCatalogResponse>("/domains");
}

export async function fetchCatalogSnapshot(): Promise<CatalogSnapshotResponse> {
  return request<CatalogSnapshotResponse>("/models/catalog");
}

export async function analyzeText(
  payload: AnalysisRequest,
): Promise<AnalysisResponse> {
  return request<AnalysisResponse>("/analyze", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function updateModel(
  modelId: string,
  payload: ModelPatchRequest,
): Promise<CatalogSnapshotResponse> {
  return request<CatalogSnapshotResponse>(`/models/${modelId}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
}

export async function reorderModels(
  payload: ModelReorderRequest,
): Promise<CatalogSnapshotResponse> {
  return request<CatalogSnapshotResponse>("/models/reorder", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function deleteModel(modelId: string): Promise<CatalogSnapshotResponse> {
  return request<CatalogSnapshotResponse>(`/models/${modelId}`, {
    method: "DELETE",
  });
}

export async function preflightLocalUpload(
  formData: FormData,
  signal?: AbortSignal,
): Promise<LocalUploadPreflightResponse> {
  return request<LocalUploadPreflightResponse>("/models/local/preflight", {
    method: "POST",
    body: formData,
    signal,
  });
}

export async function importLocalModel(
  formData: FormData,
  signal?: AbortSignal,
): Promise<ModelRegistrationResponse> {
  return request<ModelRegistrationResponse>("/models/local/import", {
    method: "POST",
    body: formData,
    signal,
  });
}

export async function preflightHuggingFaceImport(
  payload: HuggingFacePreflightRequest,
  signal?: AbortSignal,
): Promise<HuggingFacePreflightResponse> {
  return request<HuggingFacePreflightResponse>("/models/huggingface/preflight", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export async function importHuggingFaceModel(
  payload: HuggingFacePreflightRequest,
  signal?: AbortSignal,
): Promise<ModelRegistrationResponse> {
  return request<ModelRegistrationResponse>("/models/huggingface/import", {
    method: "POST",
    body: JSON.stringify(payload),
    signal,
  });
}

export async function fetchModelDashboard(
  modelId: string,
): Promise<ModelDashboardResponse> {
  const response = await request<ModelDashboardResponse>(`/models/${modelId}/dashboard`);
  return {
    ...response,
    images: response.images.map((image) => ({
      ...image,
      url: buildModelDashboardAssetUrl(modelId, image.path),
    })),
  };
}
