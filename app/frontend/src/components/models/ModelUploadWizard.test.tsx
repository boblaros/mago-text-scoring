import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ApiRequestError,
  importHuggingFaceModel,
  importLocalModel,
  preflightHuggingFaceImport,
  preflightLocalUpload,
} from "../../services/api";
import type {
  CatalogSnapshotResponse,
  HuggingFacePreflightResponse,
  LocalUploadPreflightResponse,
  ModelRegistrationResponse,
  UploadModelMetadata,
} from "../../types/contracts";
import { ModelUploadWizard } from "./ModelUploadWizard";

vi.mock("../../services/api", async () => {
  const actual = await vi.importActual<typeof import("../../services/api")>(
    "../../services/api",
  );
  return {
    ...actual,
    preflightLocalUpload: vi.fn(),
    importLocalModel: vi.fn(),
    preflightHuggingFaceImport: vi.fn(),
    importHuggingFaceModel: vi.fn(),
  };
});

const mockedPreflightLocalUpload = vi.mocked(preflightLocalUpload);
const mockedImportLocalModel = vi.mocked(importLocalModel);
const mockedPreflightHuggingFaceImport = vi.mocked(preflightHuggingFaceImport);
const mockedImportHuggingFaceModel = vi.mocked(importHuggingFaceModel);

const domains = [
  {
    domain: "sentiment",
    display_name: "Sentiment",
    color_token: "sentiment",
    group: "sentiment-core",
  },
];

function buildMetadata(
  overrides: Partial<UploadModelMetadata> = {},
): UploadModelMetadata {
  return {
    model_id: "sentiment-demo",
    domain: "sentiment",
    display_name: "Sentiment Demo",
    description: "Demo model",
    version: "v1",
    enable_on_upload: false,
    framework_type: "transformers",
    framework_task: "sequence-classification",
    framework_library: "huggingface",
    framework_problem_type: "single_label_classification",
    backbone: "distilbert-base-uncased",
    architecture: "DistilBertForSequenceClassification",
    base_model: "distilbert-base-uncased",
    embeddings: null,
    output_type: "single-label-classification",
    runtime_device: "auto",
    runtime_max_sequence_length: 128,
    runtime_batch_size: 1,
    runtime_truncation: true,
    runtime_padding: true,
    runtime_preprocessing: null,
    ui_display_name: "Sentiment",
    color_token: "sentiment",
    group: "sentiment-core",
    labels: [
      { id: 0, name: "negative", display_name: "Negative" },
      { id: 1, name: "positive", display_name: "Positive" },
    ],
    model_config: {},
    ...overrides,
  };
}

function buildLocalPreflightResponse(
  overrides: Partial<LocalUploadPreflightResponse> = {},
): LocalUploadPreflightResponse {
  return {
    ready: true,
    config_source: "generated",
    normalized_metadata: buildMetadata(),
    config_preview: "model_id: sentiment-demo\nframework:\n  type: transformers\n",
    artifact_checks: [
      { slot: "weights", title: "Weights", required: true, valid: true, files: ["model.safetensors"] },
      { slot: "tokenizer", title: "Tokenizer Assets", required: true, valid: true, files: ["tokenizer.json"] },
      { slot: "config", title: "Runtime Config Assets", required: true, valid: true, files: ["config.json"] },
    ],
    dashboard_attached: false,
    warnings: [],
    ...overrides,
  };
}

function buildHuggingFacePreflightResponse(): HuggingFacePreflightResponse {
  return {
    normalized_repo_id: "org/demo-model",
    repo_url: "https://huggingface.co/org/demo-model",
    detected_framework_type: "transformers",
    detected_task: "sequence-classification",
    framework_library: "huggingface",
    architecture: "DistilBertForSequenceClassification",
    backbone: "distilbert-base-uncased",
    base_model: "distilbert-base-uncased",
    estimated_download_size_bytes: 2048,
    disk_free_bytes: 99999999,
    memory_total_bytes: 99999999,
    memory_estimate_bytes: 4096,
    runtime_supported: true,
    compatible: true,
    ready_to_import: true,
    required_files: [
      {
        path: "model.safetensors",
        category: "weights",
        required: true,
        available: true,
        size_bytes: 1024,
      },
      {
        path: "tokenizer.json",
        category: "tokenizer",
        required: true,
        available: true,
        size_bytes: 128,
      },
      {
        path: "config.json",
        category: "config",
        required: true,
        available: true,
        size_bytes: 128,
      },
    ],
    warnings: [],
    blocking_reasons: [],
    normalized_metadata: buildMetadata(),
    config_preview: "model_id: sentiment-demo\nmodel:\n  source_repo: org/demo-model\n",
  };
}

function buildRegistrationResponse(
  overrides: Partial<ModelRegistrationResponse["result"]> = {},
): ModelRegistrationResponse {
  const snapshot: CatalogSnapshotResponse = {
    active_domains: [],
    management_domains: [],
  };
  return {
    snapshot,
    result: {
      model_id: "sentiment-demo",
      source: "local",
      branch: "local",
      config_source: "generated",
      framework_type: "transformers",
      display_name: "Sentiment Demo",
      domain: "sentiment",
      is_active: false,
      status: "ready",
      status_reason: null,
      dashboard_status: "available",
      warnings: [],
      ...overrides,
    },
  };
}

function renderWizard() {
  const onClose = vi.fn();
  const onSuccess = vi.fn();
  render(
    <ModelUploadWizard
      isOpen
      domains={domains}
      onClose={onClose}
      onSuccess={onSuccess}
    />,
  );
  return {
    onClose,
    onSuccess,
    user: userEvent.setup(),
  };
}

async function advanceToLocalValidate(user: ReturnType<typeof userEvent.setup>) {
  await user.click(screen.getByTestId("upload-source-local"));
  await user.click(screen.getByRole("button", { name: "Continue" }));
  await user.type(screen.getByLabelText("Display name"), "Sentiment Demo");
  await user.type(screen.getByLabelText("Model id"), "sentiment-demo");
  await user.click(screen.getByRole("button", { name: "Continue" }));
}

async function fillLocalFiles(user: ReturnType<typeof userEvent.setup>) {
  await user.upload(screen.getByLabelText("Weights"), new File(["weights"], "model.safetensors"));
  await user.upload(screen.getByLabelText("Tokenizer Assets"), new File(["{}"], "tokenizer.json"));
  await user.upload(screen.getByLabelText("Runtime Config Assets"), new File(["{}"], "config.json"));
}

describe("ModelUploadWizard", () => {
  beforeEach(() => {
    mockedPreflightLocalUpload.mockReset();
    mockedImportLocalModel.mockReset();
    mockedPreflightHuggingFaceImport.mockReset();
    mockedImportHuggingFaceModel.mockReset();
  });

  it("branches into the unified local path", async () => {
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(screen.getAllByText("Local upload").length).toBeGreaterThan(0);
    expect(screen.getByText("Select existing domain")).toBeInTheDocument();
  });

  it("runs local validation and reaches the review step", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(buildLocalPreflightResponse());
    const { user } = renderWizard();

    await advanceToLocalValidate(user);
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    await waitFor(() =>
      expect(mockedPreflightLocalUpload).toHaveBeenCalledTimes(1),
    );
    expect(screen.getByTestId("local-preflight-summary")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Continue to review" }));

    expect(screen.getByTestId("review-step")).toBeInTheDocument();
    expect(screen.getByText("Generated by system")).toBeInTheDocument();
    expect(screen.getByText("Sentiment Demo")).toBeInTheDocument();
  });

  it("supports an optional uploaded config inside the unified local flow", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(
      buildLocalPreflightResponse({ config_source: "uploaded" }),
    );
    const { user } = renderWizard();

    await advanceToLocalValidate(user);
    await user.upload(
      screen.getByLabelText("Registration config file"),
      new File(["model_id: sentiment-demo\n"], "model-config.yaml"),
    );
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    await waitFor(() => expect(mockedPreflightLocalUpload).toHaveBeenCalledTimes(1));
    await user.click(screen.getByRole("button", { name: "Continue to review" }));

    expect(screen.getByText("Uploaded by user")).toBeInTheDocument();
  });

  it("shows the success result summary after import", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(buildLocalPreflightResponse());
    mockedImportLocalModel.mockResolvedValue(buildRegistrationResponse());
    const { user, onSuccess } = renderWizard();

    await advanceToLocalValidate(user);
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));
    await waitFor(() => expect(mockedPreflightLocalUpload).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "Continue to review" }));
    await user.click(screen.getByTestId("review-submit"));

    await waitFor(() => expect(screen.getByTestId("result-step")).toBeInTheDocument());
    expect(screen.getByText("Sentiment Demo is registered")).toBeInTheDocument();
    expect(onSuccess).toHaveBeenCalledWith(
      buildRegistrationResponse().snapshot,
      "sentiment-demo",
    );
  });

  it("renders repo validation errors in the Hugging Face flow", async () => {
    mockedPreflightHuggingFaceImport.mockRejectedValue(
      new ApiRequestError("Paste a full model URL or a repo id like org/model-name.", 422, {
        message: "Paste a full model URL or a repo id like org/model-name.",
        field_errors: {
          "huggingface.repo": "Paste a full model URL or a repo id like org/model-name.",
        },
      }),
    );
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-hf"));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.type(screen.getByLabelText("Display name"), "HF Demo");
    await user.type(screen.getByLabelText("Model id"), "hf-demo");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.type(screen.getByTestId("hf-repo-input"), "broken");
    await user.click(screen.getByRole("button", { name: "Run preflight" }));

    await waitFor(() =>
      expect(
        screen.getAllByText("Paste a full model URL or a repo id like org/model-name.").length,
      ).toBeGreaterThan(0),
    );
  });

  it("locks the wizard during an active import request", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(buildLocalPreflightResponse());
    let resolveImport: (value: ModelRegistrationResponse) => void = () => undefined;
    mockedImportLocalModel.mockReturnValue(
      new Promise((resolve) => {
        resolveImport = resolve;
      }),
    );
    const { user } = renderWizard();

    await advanceToLocalValidate(user);
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));
    await waitFor(() => expect(mockedPreflightLocalUpload).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "Continue to review" }));
    await user.click(screen.getByTestId("review-submit"));

    expect(screen.getByTestId("progress-step")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close" })).toBeDisabled();
    expect(screen.getByText("Request in progress")).toBeInTheDocument();

    resolveImport(buildRegistrationResponse());
    await waitFor(() => expect(screen.getByTestId("result-step")).toBeInTheDocument());
  });

  it("supports the Hugging Face happy path review data", async () => {
    mockedPreflightHuggingFaceImport.mockResolvedValue(buildHuggingFacePreflightResponse());
    mockedImportHuggingFaceModel.mockResolvedValue(
      buildRegistrationResponse({
        source: "huggingface",
        branch: "huggingface",
      }),
    );
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-hf"));
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.type(screen.getByLabelText("Display name"), "HF Demo");
    await user.type(screen.getByLabelText("Model id"), "hf-demo");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.type(screen.getByTestId("hf-repo-input"), "org/demo-model");
    await user.click(screen.getByRole("button", { name: "Run preflight" }));

    await waitFor(() => expect(screen.getByTestId("hf-preflight-summary")).toBeInTheDocument());
    expect(screen.getByText("ready to import")).toBeInTheDocument();
  });
});
