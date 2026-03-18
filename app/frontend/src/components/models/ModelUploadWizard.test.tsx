import { render, screen, waitFor, within } from "@testing-library/react";
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
  await user.click(screen.getByTestId("local-config-mode-manual"));
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

    expect(screen.getByText("Do you already have a ready upload config file?")).toBeInTheDocument();
    expect(screen.getByText("Model Name & Task")).toBeInTheDocument();
    expect(screen.getByText("Model Artifacts")).toBeInTheDocument();
  });

  it("keeps the local config question footer out of the main layout flow", async () => {
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));

    expect(screen.getByRole("button", { name: "Back" }).closest("footer")).toHaveClass(
      "model-upload__actions--floating",
    );
  });

  it("advances immediately after choosing a source", async () => {
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    expect(screen.getByText("Do you already have a ready upload config file?")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Back" }));
    expect(screen.getByTestId("upload-source-local")).not.toHaveClass("model-upload-sheet__choice--active");
    expect(screen.getByTestId("upload-source-hf")).not.toHaveClass("model-upload-sheet__choice--active");
    expect(screen.getByTestId("upload-source-hf")).toBeInTheDocument();

    await user.click(screen.getByTestId("upload-source-hf"));
    expect(screen.getByLabelText("Display name")).toBeInTheDocument();
  });

  it("keeps completed sidebar steps visually neutral", async () => {
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));

    expect(screen.getByText("Source", { selector: "strong" }).closest("div")).not.toHaveClass(
      "model-upload-sheet__step--complete",
    );
    expect(
      screen.getByText("Model Name & Task", { selector: "strong" }).closest("div"),
    ).toHaveClass("model-upload-sheet__step--active");
  });

  it("hides the local follow-up question after choosing a branch", async () => {
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    const uploadedChoice = screen.getByTestId("local-config-mode-uploaded");
    await user.click(uploadedChoice);

    expect(screen.queryByText("Do you already have a ready upload config file?")).not.toBeInTheDocument();
    expect(screen.getByLabelText("Registration config file")).toBeInTheDocument();
    expect(screen.getByText("View config example")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "Back" }));
    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByTestId("local-config-mode-manual"));

    expect(screen.queryByText("Do you already have a ready upload config file?")).not.toBeInTheDocument();
    expect(screen.getByText("Select existing domain")).toBeInTheDocument();
    expect(screen.queryByLabelText("Registration config file")).not.toBeInTheDocument();
  });

  it("uses the standard details layout after the local branch selection", async () => {
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByTestId("local-config-mode-manual"));

    expect(document.querySelector(".model-upload-sheet__content--details")).not.toBeNull();
    expect(document.querySelector(".model-upload-sheet__content--details-local")).toBeNull();
  });

  it("renders the labels section with a single visible section title", async () => {
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByTestId("local-config-mode-manual"));

    expect(screen.getAllByText("Labels")).toHaveLength(1);
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
    await waitFor(() => expect(screen.getByTestId("review-step")).toBeInTheDocument());
    expect(screen.queryByTestId("validate-status")).not.toBeInTheDocument();
    expect(within(screen.getByTestId("review-step")).queryByText("Review")).not.toBeInTheDocument();
    expect(screen.queryByTestId("review-config-modal")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /config preview/i })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
    expect(screen.getByText("Generated by system")).toBeInTheDocument();
    expect(screen.getByText("Sentiment Demo")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /config preview/i }));

    expect(screen.getByTestId("review-config-modal")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /config preview/i })).toHaveAttribute(
      "aria-expanded",
      "true",
    );
    expect(screen.getByText(/model_id: sentiment-demo/i)).toBeInTheDocument();

    await user.click(within(screen.getByTestId("review-config-modal")).getByRole("button", { name: "Close" }));

    expect(screen.queryByTestId("review-config-modal")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: /config preview/i })).toHaveAttribute(
      "aria-expanded",
      "false",
    );
  });

  it("does not show a pending validation indicator while local validation is running", async () => {
    let resolvePreflight: (value: LocalUploadPreflightResponse) => void = () => undefined;
    mockedPreflightLocalUpload.mockReturnValue(
      new Promise((resolve) => {
        resolvePreflight = resolve;
      }),
    );
    const { user } = renderWizard();

    await advanceToLocalValidate(user);
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    expect(screen.queryByTestId("validate-status")).not.toBeInTheDocument();

    resolvePreflight(buildLocalPreflightResponse());
    await waitFor(() => expect(screen.getByTestId("review-step")).toBeInTheDocument());
  });

  it("highlights every missing required artifact card when validating empty local uploads", async () => {
    const { user } = renderWizard();

    await advanceToLocalValidate(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    expect(mockedPreflightLocalUpload).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Weights").closest("label")).toHaveClass("artifact-slot--error");
    expect(screen.getByLabelText("Tokenizer Assets").closest("label")).toHaveClass("artifact-slot--error");
    expect(screen.getByLabelText("Runtime Config Assets").closest("label")).toHaveClass("artifact-slot--error");
    expect(screen.getByLabelText("Dashboard bundle").closest("label")).not.toHaveClass("artifact-slot--error");
    expect(screen.getByTestId("validate-status")).toHaveClass("model-upload__status--error");
    expect(screen.getByTestId("validate-status")).toHaveTextContent("Validation failed");
    expect(screen.getByTestId("validate-status")).not.toHaveTextContent("Weights requires at least 1 file.");
    expect(screen.queryByText("No files selected.")).not.toBeInTheDocument();
  });

  it("shows clear uploaded file indicators inside artifact cards", async () => {
    const { user } = renderWizard();

    await advanceToLocalValidate(user);
    await fillLocalFiles(user);

    expect(screen.queryByText("1 file uploaded")).not.toBeInTheDocument();
    expect(screen.getByText("model.safetensors")).toBeInTheDocument();
    expect(screen.getByText("tokenizer.json")).toBeInTheDocument();
    expect(screen.getByText("config.json")).toBeInTheDocument();
    expect(screen.getByText("No dashboard bundle uploaded yet")).toBeInTheDocument();
  });

  it("removes empty placeholder file lines from artifact cards", async () => {
    const { user } = renderWizard();

    await advanceToLocalValidate(user);

    expect(screen.queryByText("No files selected.")).not.toBeInTheDocument();
    expect(screen.queryByText("No dashboard bundle selected.")).not.toBeInTheDocument();
    expect(screen.getAllByText("No files uploaded yet")).toHaveLength(3);
  });

  it("resets the wizard scroll position when moving to model artifacts upload", async () => {
    const scrollToSpy = vi.fn();
    const originalScrollTo = Object.getOwnPropertyDescriptor(HTMLElement.prototype, "scrollTo");
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      value: scrollToSpy,
    });

    try {
      const { user } = renderWizard();

      await user.click(screen.getByTestId("upload-source-local"));
      await user.click(screen.getByTestId("local-config-mode-manual"));
      await user.type(screen.getByLabelText("Display name"), "Sentiment Demo");
      await user.type(screen.getByLabelText("Model id"), "sentiment-demo");

      scrollToSpy.mockClear();
      await user.click(screen.getByRole("button", { name: "Continue" }));

      await waitFor(() => expect(screen.getByLabelText("Weights")).toBeInTheDocument());
      expect(scrollToSpy).toHaveBeenCalledWith({ top: 0, left: 0, behavior: "auto" });
    } finally {
      if (originalScrollTo) {
        Object.defineProperty(HTMLElement.prototype, "scrollTo", originalScrollTo);
      } else {
        delete (HTMLElement.prototype as Partial<HTMLElement>).scrollTo;
      }
    }
  });

  it("uses the typed label name as the default display name during local preflight", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(buildLocalPreflightResponse());
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByTestId("local-config-mode-manual"));
    await user.type(screen.getByLabelText("Display name"), "Sentiment Demo");
    await user.type(screen.getByLabelText("Model id"), "sentiment-demo");
    const labelNameInput = screen.getByDisplayValue("class_0");
    await user.clear(labelNameInput);
    await user.type(labelNameInput, "Negative");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    await waitFor(() => expect(mockedPreflightLocalUpload).toHaveBeenCalledTimes(1));

    const formData = mockedPreflightLocalUpload.mock.calls[0]?.[0];
    expect(formData).toBeInstanceOf(FormData);

    const payload = JSON.parse(String((formData as FormData).get("payload"))) as {
      metadata: UploadModelMetadata;
    };
    expect(payload.metadata.labels[0]).toMatchObject({
      id: 0,
      name: "Negative",
      display_name: "Negative",
    });
  });

  it("can prefill labels from a label map file before model artifacts upload", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(buildLocalPreflightResponse());
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByTestId("local-config-mode-manual"));
    await user.type(screen.getByLabelText("Display name"), "Sentiment Demo");
    await user.type(screen.getByLabelText("Model id"), "sentiment-demo");
    await user.click(screen.getByTestId("label-input-mode-file"));
    await user.upload(
      screen.getByLabelText("Label map file"),
      new File(
        [JSON.stringify({ 0: "Negative", 1: "Positive" })],
        "label_mapping.json",
        { type: "application/json" },
      ),
    );

    await waitFor(() => expect(screen.getByText("Negative")).toBeInTheDocument());
    await user.click(screen.getByRole("button", { name: "Continue" }));

    expect(screen.queryByText("Label Map")).not.toBeInTheDocument();
    expect(screen.queryByText("Label Classes")).not.toBeInTheDocument();
    expect(screen.getByText("Dashboard Bundle")).toBeInTheDocument();

    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    await waitFor(() => expect(mockedPreflightLocalUpload).toHaveBeenCalledTimes(1));

    const formData = mockedPreflightLocalUpload.mock.calls[0]?.[0];
    const payload = JSON.parse(String((formData as FormData).get("payload"))) as {
      metadata: UploadModelMetadata;
      artifact_manifest: Record<string, Array<{ name: string }>>;
    };
    expect(payload.metadata.labels).toEqual([
      { id: 0, name: "Negative", display_name: "Negative" },
      { id: 1, name: "Positive", display_name: "Positive" },
    ]);
    expect(payload.artifact_manifest.label_map_file?.[0]?.name).toBe("label_mapping.json");
  });

  it("keeps label names raw when raw numbers mode is selected", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(buildLocalPreflightResponse());
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByTestId("local-config-mode-manual"));
    await user.type(screen.getByLabelText("Display name"), "Sentiment Demo");
    await user.type(screen.getByLabelText("Model id"), "sentiment-demo");
    await user.click(screen.getByTestId("label-input-mode-raw"));
    await user.click(screen.getByRole("button", { name: "Continue" }));

    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    await waitFor(() => expect(mockedPreflightLocalUpload).toHaveBeenCalledTimes(1));

    const formData = mockedPreflightLocalUpload.mock.calls[0]?.[0];
    const payload = JSON.parse(String((formData as FormData).get("payload"))) as {
      metadata: UploadModelMetadata;
    };
    expect(payload.metadata.labels[0]).toEqual({
      id: 0,
      name: "0",
      display_name: "0",
    });
  });

  it("supports an optional uploaded config inside the unified local flow", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(
      buildLocalPreflightResponse({ config_source: "uploaded" }),
    );
    const { user } = renderWizard();

    await user.click(screen.getByTestId("upload-source-local"));
    await user.click(screen.getByTestId("local-config-mode-uploaded"));
    await user.upload(
      screen.getByLabelText("Registration config file"),
      new File(
        [
          [
            'model_id: "sentiment-demo"',
            'domain: "sentiment"',
            'display_name: "Sentiment Demo"',
            "framework:",
            '  type: "transformers"',
            '  task: "sequence-classification"',
            "labels:",
            '  type: "single-label-classification"',
            "  classes:",
            "    - id: 0",
            '      name: "negative"',
          ].join("\n"),
        ],
        "model-config.yaml",
      ),
    );
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));

    await waitFor(() => expect(mockedPreflightLocalUpload).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.getByTestId("review-step")).toBeInTheDocument());
    expect(screen.getByText("Uploaded by user")).toBeInTheDocument();
  });

  it("shows the success result summary after import", async () => {
    mockedPreflightLocalUpload.mockResolvedValue(buildLocalPreflightResponse());
    mockedImportLocalModel.mockResolvedValue(buildRegistrationResponse());
    const { user, onSuccess } = renderWizard();

    await advanceToLocalValidate(user);
    await fillLocalFiles(user);
    await user.click(screen.getByRole("button", { name: "Validate files" }));
    await waitFor(() => expect(screen.getByTestId("review-step")).toBeInTheDocument());
    await user.click(screen.getByTestId("review-submit"));

    await waitFor(() => expect(screen.getByTestId("result-step")).toBeInTheDocument());
    expect(screen.getByText("Model uploaded successfully")).toBeInTheDocument();
    expect(screen.queryByText("Upload Complete")).not.toBeInTheDocument();
    expect(screen.getByText("Results", { selector: "strong" }).closest("div")).toHaveClass(
      "model-upload-sheet__step--active",
    );
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
    await waitFor(() => expect(screen.getByTestId("review-step")).toBeInTheDocument());
    await user.click(screen.getByTestId("review-submit"));

    expect(screen.getByTestId("review-step")).toBeInTheDocument();
    expect(screen.getByTestId("upload-running-indicator")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Close" })).toBeDisabled();
    expect(screen.getByTestId("review-submit")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();

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
    await user.type(screen.getByLabelText("Display name"), "HF Demo");
    await user.type(screen.getByLabelText("Model id"), "hf-demo");
    await user.click(screen.getByRole("button", { name: "Continue" }));
    await user.type(screen.getByTestId("hf-repo-input"), "org/demo-model");
    await user.click(screen.getByRole("button", { name: "Run preflight" }));

    await waitFor(() => expect(screen.getByTestId("review-step")).toBeInTheDocument());
    expect(screen.getByText("org/demo-model")).toBeInTheDocument();
    expect(screen.getByText("Hugging Face")).toBeInTheDocument();
  });
});
