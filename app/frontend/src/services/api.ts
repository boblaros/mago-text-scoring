import type {
  AnalysisRequest,
  AnalysisResponse,
  CatalogSnapshotResponse,
  DomainCatalogResponse,
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
      if (typeof detail === "string") {
        throw new Error(detail);
      }
      if (detail && typeof detail === "object") {
        throw new Error(JSON.stringify(detail));
      }
    }
    const body = await response.text();
    throw new Error(body || `Request failed with status ${response.status}`);
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

export async function uploadModel(formData: FormData): Promise<CatalogSnapshotResponse> {
  return request<CatalogSnapshotResponse>("/models/upload", {
    method: "POST",
    body: formData,
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
