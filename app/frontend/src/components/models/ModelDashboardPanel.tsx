import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { PlotlyFigure } from "./PlotlyFigure";
import type {
  DashboardFigure,
  DashboardImageAsset,
  DashboardSourceItem,
  DashboardSectionSummary,
  DomainCatalogModel,
  ModelDashboardResponse,
} from "../../types/contracts";

type DashboardMetricCard = {
  label: string;
  value: string;
  hint?: string;
};

type ChartIndicatorKind = "plotly" | "image" | "missing";

type ChartIndicator = DashboardSectionSummary & {
  figures: DashboardFigure[];
  images: DashboardImageAsset[];
  kind: ChartIndicatorKind;
  metaLabel: string;
  actionLabel: string;
  descriptionText: string;
};

type MetadataValueGroup = {
  key: string;
  values: string[];
};

type SourceGroup = {
  category: string;
  paths: string[];
  reason: string | null;
};

type MetadataSourceAccordionItem = {
  id: string;
  title: string;
  groups: SourceGroup[];
  paths: string[];
};

type MetadataRuntimeAccordionItem = {
  id: string;
  title: string;
  groups: MetadataValueGroup[];
  values: string[];
  isLocalFiles?: boolean;
};

type DistributionTableBlock = {
  key: string;
  title: string;
  rows: Record<string, unknown>[];
};

const CHART_SECTION_PRIORITY: Record<string, number> = {
  benchmark: 0,
  confusion_matrix: 1,
  training_curves: 2,
  learning_curves: 3,
  class_distribution: 4,
  cross_dataset: 5,
  dataset_diagnostics: 6,
  topic_analysis: 7,
  additional_charts: 8,
};

const ARTEFACTS_AND_CHARTS_ORDER: Record<string, number> = {
  summary: 0,
  evaluation: 1,
  benchmark: 2,
  cross_dataset: 3,
  class_distribution: 4,
  confusion_matrix: 5,
  training_curves: 6,
  learning_curves: 7,
  samples: 8,
  dataset_diagnostics: 9,
  topic_analysis: 10,
  metadata: 11,
};

const METADATA_ARTIFACT_ORDER: Record<string, number> = {
  base_dir: 0,
  weights: 1,
  tokenizer: 2,
  config: 3,
  vocabulary: 4,
  label_map_file: 5,
  label_classes_file: 6,
  label_encoder_file: 7,
};

const SOURCE_ACCORDION_ORDER = [
  { id: "metadata", title: "Metadata", categories: ["metadata"] },
  { id: "experiment_config", title: "Experiment config", categories: ["experiment_config"] },
  { id: "primary_evaluation", title: "Primary evaluation", categories: ["primary_evaluation"] },
  { id: "benchmark", title: "Benchmark", categories: ["benchmark"] },
  { id: "training_history", title: "Training history", categories: ["training_history", "learning_curve"] },
  { id: "cross_dataset", title: "Cross dataset", categories: ["cross_dataset"] },
  {
    id: "class_distribution",
    title: "Class distribution",
    categories: ["class_distribution", "source_dataset_distribution"],
  },
  { id: "prediction_samples", title: "Prediction samples", categories: ["prediction_samples"] },
  { id: "confusion_matrix", title: "Confusion matrix", categories: ["confusion_matrix"] },
] as const;

const RUNTIME_ACCORDION_ORDER = [
  { id: "runtime_bundle", title: "Runtime bundle", keys: ["base_dir", "weights", "tokenizer"] },
  { id: "config", title: "Config", keys: ["config"] },
  { id: "vocabulary", title: "Vocabulary", keys: ["vocabulary"] },
  { id: "label_map_file", title: "Label map file", keys: ["label_map_file"] },
  { id: "label_classes_file", title: "Label classes file", keys: ["label_classes_file"] },
  { id: "label_encoder_file", title: "Label encoder file", keys: ["label_encoder_file"] },
  { id: "best_checkpoint", title: "Best checkpoint", keys: ["best_checkpoint", "best_model_dir", "checkpoints_dir"] },
  { id: "experiment_dir", title: "Experiment dir", keys: ["experiment_dir"] },
] as const;

function asRecord(value: unknown) {
  return value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asArray(value: unknown) {
  return Array.isArray(value) ? value : [];
}

function findDocument(dashboard: ModelDashboardResponse, path: string) {
  return dashboard.documents[path] ?? null;
}

function formatNumber(value: unknown) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "—";
  }
  if (value >= 0 && value <= 1) {
    return `${(value * 100).toFixed(1)}%`;
  }
  if (Number.isInteger(value)) {
    return value.toLocaleString();
  }
  return value.toFixed(3);
}

function numericValue(value: unknown) {
  return typeof value === "number" && !Number.isNaN(value) ? value : null;
}

function formatTimestamp(value?: string | null) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  const now = new Date();
  const includeYear = date.getFullYear() !== now.getFullYear();

  return date.toLocaleString([], {
    ...(includeYear ? { year: "numeric" as const } : {}),
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

function humanizeStatus(status: string) {
  return status.replace(/_/g, " ");
}

function familyLabel(model: DomainCatalogModel) {
  if (model.framework_type === "transformers") {
    return "Transformer";
  }
  if (model.framework_type === "pytorch") {
    return model.architecture ? `PyTorch · ${model.architecture}` : "PyTorch";
  }
  if (model.framework_type === "sklearn") {
    return "Classical ML";
  }
  return model.framework_type;
}

function formatLabel(value: string) {
  return value
    .replace(/_/g, " ")
    .replace(/\bf1\b/gi, "F1")
    .replace(/\bauc\b/gi, "AUC")
    .replace(/\bapi\b/gi, "API");
}

function formatSplitLabel(value: string) {
  if (value === "val") {
    return "Validation";
  }
  if (value === "test") {
    return "Test";
  }
  return formatLabel(value);
}

function formatCoverageSectionTitle(section: DashboardSectionSummary) {
  if (section.id === "training_curves") {
    return "Training Charts";
  }
  if (section.id === "cross_dataset") {
    return "Cross Dataset Eval.";
  }
  return section.title;
}

function formatCoverageSectionDescription(section: DashboardSectionSummary) {
  if (section.id === "summary") {
    return "Open compact model info and output details.";
  }
  if (section.id === "evaluation") {
    return "Jump to primary validation and test metrics.";
  }
  if (section.id === "benchmark") {
    return "Leaderboard context and production rank.";
  }
  if (section.id === "cross_dataset") {
    return "External dataset performance checks.";
  }
  if (section.id === "class_distribution") {
    return "Label balance across splits and source data.";
  }
  if (section.id === "confusion_matrix") {
    return "Class-level prediction mix from static matrices.";
  }
  if (section.id === "training_curves") {
    return "Training loss and metric trend charts.";
  }
  if (section.id === "learning_curves") {
    return "Quality versus sample size checkpoints.";
  }
  if (section.id === "samples") {
    return "Prediction examples with labels and scores.";
  }
  if (section.id === "dataset_diagnostics") {
    return "EDA snapshots and dataset-level diagnostics.";
  }
  if (section.id === "topic_analysis") {
    return "Topic-model and clustering visuals.";
  }
  if (section.id === "metadata") {
    return "Sources, runtime assets, and diagnostics.";
  }

  return String(section.reason ?? section.description ?? "No notes attached.");
}

function splitTone(value: string) {
  if (value === "validation" || value === "val") {
    return "validation";
  }
  if (value === "test") {
    return "test";
  }
  return "default";
}

function comparisonTone(label: string) {
  const normalized = label.toLowerCase();
  if (normalized.includes("benchmark leader")) {
    return "benchmark";
  }
  if (normalized.includes("rank")) {
    return "rank";
  }
  if (normalized.includes("external")) {
    return "external";
  }
  if (normalized.includes("learning")) {
    return "learning";
  }
  if (normalized.includes("training")) {
    return "training";
  }
  return "default";
}

function overviewFactTone(label: string) {
  if (label === "Model") {
    return "identity";
  }
  if (label === "Domain") {
    return "domain";
  }
  if (label === "Runtime") {
    return "runtime";
  }
  if (label === "Generated") {
    return "generated";
  }
  if (label === "Sections") {
    return "sections";
  }
  if (label === "Charts") {
    return "charts";
  }
  return "default";
}

function isOverviewFactFeatured(label: string) {
  return label === "Model";
}

function isOverviewFactNumeric(label: string) {
  return label === "Sections" || label === "Charts";
}

function stringifyStructuredValue(value: unknown) {
  try {
    return JSON.stringify(value, null, 2) ?? String(value);
  } catch {
    return String(value);
  }
}

function toRecordArray(value: unknown) {
  return asArray(value).filter((item): item is Record<string, unknown> => Boolean(asRecord(item)));
}

function toStringValues(value: unknown): string[] {
  if (typeof value === "string") {
    return value.trim() ? [value] : [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => (typeof item === "string" && item.trim() ? [item] : []));
  }

  return [];
}

function dedupeStrings(values: string[]) {
  return Array.from(new Set(values));
}

function pluralize(count: number, singular: string) {
  return `${count} ${singular}${count === 1 ? "" : "s"}`;
}

function humanizeArtifactStem(value: string) {
  return formatLabel(value.replace(/[-/]/g, "_"));
}

function buildDistributionTables(documents: Record<string, unknown>): DistributionTableBlock[] {
  return Object.entries(documents)
    .filter(([path, value]) => path.startsWith("distributions/") && Boolean(asRecord(value)))
    .sort(([leftPath], [rightPath]) => {
      const rank = (path: string) => {
        if (path.includes("class-distribution")) {
          return 0;
        }
        if (path.includes("source-dataset-distribution")) {
          return 1;
        }
        return 2;
      };

      return rank(leftPath) - rank(rightPath) || leftPath.localeCompare(rightPath);
    })
    .flatMap(([path, value]) => {
      const record = asRecord(value);
      if (!record) {
        return [];
      }

      const overallRows = toRecordArray(record.overall);
      const splitRows = toRecordArray(record.splits);
      if (!overallRows.length && !splitRows.length) {
        return [];
      }

      const stem = lastPathSegment(path).replace(/\.json$/i, "");
      const titleBase = stem.replace(/[-_]/g, " ").trim().toLowerCase();
      const blocks: DistributionTableBlock[] = [];

      if (overallRows.length) {
        blocks.push({
          key: `${path}:overall`,
          title: `${titleBase} overview`,
          rows: overallRows,
        });
      }

      if (splitRows.length) {
        blocks.push({
          key: `${path}:splits`,
          title: `${titleBase} by split`,
          rows: splitRows,
        });
      }

      return blocks;
    });
}

function lastPathSegment(value: string) {
  const segments = value.split("/").filter(Boolean);
  return segments[segments.length - 1] ?? value;
}

function summarizeListPreview(values: string[]) {
  const visibleValues = values.map((value) => value.trim()).filter(Boolean);
  if (!visibleValues.length) {
    return null;
  }

  const [first, ...rest] = visibleValues;
  return rest.length ? `${first} +${rest.length} more` : first;
}

function summarizePathPreview(values: string[]) {
  return summarizeListPreview(values.map((value) => lastPathSegment(value)));
}

function buildMetadataValueGroups(
  record: Record<string, unknown> | null,
  order: Record<string, number> = {},
): MetadataValueGroup[] {
  return Object.entries(record ?? {})
    .map(([key, value]) => ({
      key,
      values: dedupeStrings(toStringValues(value)),
    }))
    .filter((group) => group.values.length)
    .sort(
      (left, right) =>
        (order[left.key] ?? Number.MAX_SAFE_INTEGER) - (order[right.key] ?? Number.MAX_SAFE_INTEGER) ||
        left.key.localeCompare(right.key),
    );
}

function buildSourceGroups(selectedSources: DashboardSourceItem[]): SourceGroup[] {
  const grouped = new Map<string, { paths: string[]; reasons: string[] }>();

  selectedSources.forEach((source) => {
    const current = grouped.get(source.category) ?? { paths: [], reasons: [] };
    current.paths.push(source.path);
    if (source.reason) {
      current.reasons.push(source.reason);
    }
    grouped.set(source.category, current);
  });

  return Array.from(grouped.entries()).map(([category, value]) => ({
    category,
    paths: dedupeStrings(value.paths),
    reason: dedupeStrings(value.reasons)[0] ?? null,
  }));
}

function isVisibleArtifactPath(path: string) {
  return path.split("/").every((segment) => segment && !segment.startsWith("."));
}

function buildMetadataSourceAccordionItems(sourceGroups: SourceGroup[]): MetadataSourceAccordionItem[] {
  const consumedCategories = new Set<string>();

  const orderedItems: MetadataSourceAccordionItem[] = SOURCE_ACCORDION_ORDER.flatMap((definition) => {
    const groups = sourceGroups.filter((group) =>
      definition.categories.some((category) => category === group.category),
    );
    if (!groups.length) {
      return [];
    }

    groups.forEach((group) => consumedCategories.add(group.category));

    return [
      {
        id: definition.id,
        title: definition.title,
        groups,
        paths: dedupeStrings(groups.flatMap((group) => group.paths)),
      },
    ];
  });

  const extraItems = sourceGroups
    .filter((group) => !consumedCategories.has(group.category))
    .sort((left, right) => left.category.localeCompare(right.category))
    .map((group) => ({
      id: group.category,
      title: formatLabel(group.category),
      groups: [group],
      paths: group.paths,
    }));

  return [...orderedItems, ...extraItems];
}

function buildMetadataRuntimeAccordionItems(
  runtimeArtifactGroups: MetadataValueGroup[],
  experimentPathGroups: MetadataValueGroup[],
  localFiles: string[],
): MetadataRuntimeAccordionItem[] {
  const runtimeGroupMap = new Map(runtimeArtifactGroups.map((group) => [group.key, group]));
  const experimentGroupMap = new Map(experimentPathGroups.map((group) => [group.key, group]));
  const consumedRuntimeKeys = new Set<string>();
  const consumedExperimentKeys = new Set<string>();

  const orderedItems: MetadataRuntimeAccordionItem[] = RUNTIME_ACCORDION_ORDER.flatMap((definition) => {
    const groups = definition.keys.flatMap((key) => {
      const group = runtimeGroupMap.get(key) ?? experimentGroupMap.get(key);
      if (!group) {
        return [];
      }
      if (runtimeGroupMap.has(key)) {
        consumedRuntimeKeys.add(key);
      }
      if (experimentGroupMap.has(key)) {
        consumedExperimentKeys.add(key);
      }
      return [group];
    });

    if (!groups.length) {
      return [];
    }

    return [
      {
        id: definition.id,
        title: definition.title,
        groups,
        values: dedupeStrings(groups.flatMap((group) => group.values)),
      },
    ];
  });

  const extraRuntimeItems: MetadataRuntimeAccordionItem[] = runtimeArtifactGroups
    .filter((group) => !consumedRuntimeKeys.has(group.key))
    .sort((left, right) => left.key.localeCompare(right.key))
    .map((group) => ({
      id: group.key,
      title: formatLabel(group.key),
      groups: [group],
      values: group.values,
    }));

  const extraExperimentItems: MetadataRuntimeAccordionItem[] = experimentPathGroups
    .filter((group) => !consumedExperimentKeys.has(group.key))
    .sort((left, right) => left.key.localeCompare(right.key))
    .map((group) => ({
      id: group.key,
      title: formatLabel(group.key),
      groups: [group],
      values: group.values,
    }));

  const items = [...orderedItems, ...extraRuntimeItems, ...extraExperimentItems];

  if (localFiles.length) {
    items.push({
      id: "top_level_local_files",
      title: "Top-level local files",
      groups: [],
      values: localFiles,
      isLocalFiles: true,
    });
  }

  return items;
}

function getMetricFromSplits(
  splits: Record<string, unknown> | null,
  splitCandidates: string[],
  metric: string,
) {
  for (const split of splitCandidates) {
    const splitRecord = asRecord(splits?.[split]);
    const value = splitRecord ? numericValue(splitRecord[metric]) : null;
    if (value !== null) {
      return value;
    }
  }
  return null;
}

function buildPrimaryMetricCards(
  splits: Record<string, unknown> | null,
  benchmarkRows: Record<string, unknown>[],
  crossDatasetRows: Record<string, unknown>[],
) {
  const cards: DashboardMetricCard[] = [];

  const metricSpecs = [
    { label: "Test F1 macro", splitCandidates: ["test"], metric: "f1_macro" },
    { label: "Test accuracy", splitCandidates: ["test"], metric: "accuracy" },
    { label: "Validation F1 macro", splitCandidates: ["validation", "val"], metric: "f1_macro" },
    { label: "Validation accuracy", splitCandidates: ["validation", "val"], metric: "accuracy" },
  ];

  metricSpecs.forEach((spec) => {
    const value = getMetricFromSplits(splits, spec.splitCandidates, spec.metric);
    if (value !== null) {
      cards.push({
        label: spec.label,
        value: formatNumber(value),
      });
    }
  });

  if (crossDatasetRows.length) {
    const scores = crossDatasetRows
      .map((row) => numericValue(row.f1_macro) ?? numericValue(row.accuracy))
      .filter((score): score is number => score !== null);
    if (scores.length) {
      const average = scores.reduce((sum, score) => sum + score, 0) / scores.length;
      cards.push({
        label: "External eval",
        value: formatNumber(average),
      });
    }
  }

  return cards.slice(0, 5);
}

function chartKindLabel(kind: ChartIndicatorKind) {
  if (kind === "plotly") {
    return "Plotly";
  }
  if (kind === "image") {
    return "Static image";
  }
  return "Missing";
}

function chartDescriptionText(section: DashboardSectionSummary) {
  if (section.id === "benchmark") {
    return "Production model against benchmark peers.";
  }

  if (section.id === "class_distribution") {
    return "Class balance across splits and source data.";
  }

  return String(
    section.description ??
      section.reason ??
      "No visual artifact or chart payload was attached for this section.",
  );
}

function buildChartIndicators(
  sections: DashboardSectionSummary[],
  figures: DashboardFigure[],
  images: DashboardImageAsset[],
) {
  const indicators = sections
    .filter(
      (section) =>
        CHART_SECTION_PRIORITY[section.id] !== undefined ||
        section.charts.length > 0 ||
        section.status === "image_only",
    )
    .map((section): ChartIndicator => {
      const sectionFigures = figures.filter((figure) => figure.section_id === section.id);
      const sectionImages = images.filter((image) => image.section_id === section.id);
      const kind: ChartIndicatorKind = sectionFigures.length
        ? "plotly"
        : sectionImages.length
          ? "image"
          : "missing";
      const totalAssets = sectionFigures.length + sectionImages.length;

      return {
        ...section,
        figures: sectionFigures,
        images: sectionImages,
        kind,
        metaLabel:
          kind === "missing"
            ? "0 assets"
            : totalAssets
              ? `${totalAssets} asset${totalAssets === 1 ? "" : "s"}`
              : humanizeStatus(section.status),
        actionLabel: "",
        descriptionText: chartDescriptionText(section),
      };
    });

  const unsectionedFigures = figures.filter((figure) => !figure.section_id);
  const unsectionedImages = images.filter((image) => !image.section_id);
  if (unsectionedFigures.length || unsectionedImages.length) {
    const totalAssets = unsectionedFigures.length + unsectionedImages.length;
    indicators.push({
      id: "additional_charts",
      title: "Additional Charts",
      status: "available",
      description: "Charts attached without an explicit dashboard section.",
      reason: null,
      files: [],
      charts: [],
      figures: unsectionedFigures,
      images: unsectionedImages,
      kind: unsectionedFigures.length ? "plotly" : unsectionedImages.length ? "image" : "missing",
      metaLabel: `${totalAssets} asset${totalAssets === 1 ? "" : "s"}`,
      actionLabel: "",
      descriptionText: "Extra visual outputs collected outside the standard dashboard sections.",
    });
  }

  return indicators.sort(
    (left, right) =>
      (CHART_SECTION_PRIORITY[left.id] ?? 99) - (CHART_SECTION_PRIORITY[right.id] ?? 99),
  );
}

function buildMissingPreviewIndicator(section: DashboardSectionSummary): ChartIndicator {
  return {
    ...section,
    figures: [],
    images: [],
    kind: "missing",
    metaLabel: "0 assets",
    actionLabel: "",
    descriptionText: chartDescriptionText(section),
  };
}

function countFigureTraces(figure: DashboardFigure) {
  return Array.isArray(figure.figure.data) ? figure.figure.data.length : 0;
}

function summarizeFigureTraceKinds(figure: DashboardFigure) {
  const traceKinds = Array.from(
    new Set(
      (Array.isArray(figure.figure.data) ? figure.figure.data : [])
        .map((trace) => {
          const traceRecord = asRecord(trace);
          return typeof traceRecord?.type === "string" ? traceRecord.type : "trace";
        })
        .filter(Boolean),
    ),
  );

  if (!traceKinds.length) {
    return "Interactive figure payload";
  }

  return traceKinds.slice(0, 3).map((kind) => formatLabel(kind)).join(" · ");
}

function summarizeChartAssets(indicator: ChartIndicator) {
  const parts: string[] = [];

  if (indicator.figures.length) {
    parts.push(`${indicator.figures.length} figure${indicator.figures.length === 1 ? "" : "s"}`);
  }
  if (indicator.images.length) {
    parts.push(`${indicator.images.length} image${indicator.images.length === 1 ? "" : "s"}`);
  }

  return parts.join(" · ") || "No chart assets attached";
}

function DashboardDetail({
  title,
  meta,
  className,
  children,
}: {
  title: string;
  meta?: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details className={`models-dashboard__detail${className ? ` ${className}` : ""}`}>
      <summary>
        <span>{title}</span>
        {meta ? <strong>{meta}</strong> : null}
      </summary>
      <div className="models-dashboard__detail-body">{children}</div>
    </details>
  );
}

function MetadataAccordion({
  title,
  meta,
  preview,
  tone,
  level,
  defaultOpen = false,
  children,
}: {
  title: string;
  meta?: string;
  preview?: string | null;
  tone: "core" | "runtime" | "diagnostic";
  level: "top" | "nested";
  defaultOpen?: boolean;
  children: ReactNode;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <section
      className={`models-dashboard__metadata-accordion models-dashboard__metadata-accordion--${tone} models-dashboard__metadata-accordion--${level}${
        isOpen ? " is-open" : ""
      }`}
    >
      <button
        type="button"
        className="models-dashboard__metadata-accordion-toggle"
        aria-expanded={isOpen}
        onClick={() => setIsOpen((current) => !current)}
      >
        <div className="models-dashboard__metadata-accordion-summary">
          <div className="models-dashboard__metadata-accordion-title-row">
            <strong className="models-dashboard__metadata-accordion-title">{title}</strong>
            {meta ? <span className="models-dashboard__metadata-accordion-meta">{meta}</span> : null}
          </div>
          {preview ? <span className="models-dashboard__metadata-accordion-preview">{preview}</span> : null}
        </div>

        <span className="models-dashboard__metadata-accordion-caret" aria-hidden="true" />
      </button>

      <div className="models-dashboard__metadata-accordion-shell">
        <div className="models-dashboard__metadata-accordion-content">{children}</div>
      </div>
    </section>
  );
}

function DashboardHeaderInfo({
  label,
  text,
}: {
  label: string;
  text: string;
}) {
  return (
    <details className="dashboard-header__info">
      <summary aria-label={label}>
        <span>i</span>
      </summary>
      <div className="dashboard-header__popover">{text}</div>
    </details>
  );
}

function CompactTable({
  title,
  eyebrow,
  rows,
  limit,
  hideHeader = false,
  variant = "default",
}: {
  title: string;
  eyebrow?: string;
  rows: Record<string, unknown>[];
  limit?: number;
  hideHeader?: boolean;
  variant?: "default" | "distribution";
}) {
  if (!rows.length) {
    return null;
  }

  const visibleRows = typeof limit === "number" ? rows.slice(0, limit) : rows;
  const columns = Array.from(
    visibleRows.reduce((keys, row) => {
      Object.keys(row).forEach((key) => {
        if (key !== "source_file" && key !== "artifact_paths") {
          keys.add(key);
        }
      });
      return keys;
    }, new Set<string>()),
  ).slice(0, 7);

  return (
    <div
      className={`models-dashboard__table-block${
        variant === "distribution" ? " models-dashboard__table-block--distribution" : ""
      }`}
    >
      {!hideHeader ? (
        <div className="dashboard-block__header">
          <div>
            {eyebrow ? <div className="dashboard-block__eyebrow">{eyebrow}</div> : null}
            <h4>{title}</h4>
          </div>
        </div>
      ) : null}

      <div className="dashboard-table-shell">
        <table className="dashboard-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{formatLabel(column)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, rowIndex) => (
              <tr key={`${title}-${rowIndex}`}>
                {columns.map((column) => {
                  const value = row[column];
                  return (
                    <td key={column}>
                      {typeof value === "number" ? formatNumber(value) : value ? String(value) : "—"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {typeof limit === "number" && rows.length > limit ? (
        <p className="models-dashboard__caption">
          Showing {limit} of {rows.length} rows.
        </p>
      ) : null}
    </div>
  );
}

function SampleCards({
  samples,
}: {
  samples: Record<string, unknown>[];
}) {
  return (
    <div className="dashboard-table-shell models-dashboard__sample-table-shell">
      <table className="dashboard-table models-dashboard__sample-table">
        <thead>
          <tr>
            <th>№</th>
            <th>Text</th>
            <th>Production</th>
            <th>Reference</th>
          </tr>
        </thead>
        <tbody>
          {samples.map((sample, index) => {
            const productionPrediction = asRecord(sample.production_prediction);
            const referencePrediction = asRecord(sample.reference_prediction);

            return (
              <tr key={`sample-${index}`}>
                <td className="models-dashboard__sample-id">
                  {String(sample.example_id ?? index + 1)}
                </td>
                <td className="models-dashboard__sample-text">
                  <div className="models-dashboard__sample-text-copy">
                    {String(sample.text ?? "—")}
                  </div>
                </td>
                <td>
                  {productionPrediction ? (
                    <div className="models-dashboard__sample-prediction">
                      <strong>{String(productionPrediction.model ?? "Production")}</strong>
                      <span>
                        {String(productionPrediction.label ?? "—")} ·{" "}
                        {formatNumber(productionPrediction.confidence)}
                      </span>
                    </div>
                  ) : (
                    "—"
                  )}
                </td>
                <td>
                  {referencePrediction ? (
                    <div className="models-dashboard__sample-prediction">
                      <strong>{String(referencePrediction.model ?? "Reference")}</strong>
                      <span>
                        {String(referencePrediction.label ?? "—")} ·{" "}
                        {formatNumber(referencePrediction.confidence)}
                      </span>
                    </div>
                  ) : (
                    "—"
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

interface ModelDashboardPanelProps {
  model: DomainCatalogModel;
  dashboard: ModelDashboardResponse;
  onOpenModelInfo: () => void;
}

export function ModelDashboardPanel({
  model,
  dashboard,
  onOpenModelInfo,
}: ModelDashboardPanelProps) {
  const manifestSections = dashboard.manifest?.sections ?? [];
  const overview = asRecord(dashboard.overview);
  const sourceAudit = asRecord(dashboard.source_audit);
  const metadataModel = asRecord(findDocument(dashboard, "metadata/model.json"));
  const primaryEvaluation = asRecord(findDocument(dashboard, "metrics/primary-evaluation.json"));
  const benchmarkRows = toRecordArray(findDocument(dashboard, "metrics/benchmark-test.json"));
  const crossDatasetRows = toRecordArray(findDocument(dashboard, "metrics/cross-dataset.json"));
  const learningCurves = toRecordArray(findDocument(dashboard, "curves/learning-curve.json"));
  const predictionSamples = toRecordArray(findDocument(dashboard, "samples/prediction-samples.json"));
  const trainingHistory = asRecord(findDocument(dashboard, "curves/training-history.json"));

  const metadataUi = asRecord(metadataModel?.ui);
  const metadataArtifacts = asRecord(metadataModel?.artifacts);
  const artifactPaths = asRecord(primaryEvaluation?.artifact_paths);
  const artifactsPresent = asArray(metadataModel?.artifacts_present).map(String);

  const chartIndicators = useMemo(
    () => buildChartIndicators(manifestSections, dashboard.figures, dashboard.images),
    [dashboard.figures, dashboard.images, manifestSections],
  );

  const [openChartSectionId, setOpenChartSectionId] = useState<string | null>(null);
  const [chartViewerSectionId, setChartViewerSectionId] = useState<string | null>(null);
  const [chartViewerImageIndex, setChartViewerImageIndex] = useState(0);
  const overviewRef = useRef<HTMLElement | null>(null);
  const evaluationRef = useRef<HTMLElement | null>(null);
  const metadataRef = useRef<HTMLElement | null>(null);
  const chartDialogRef = useRef<HTMLDivElement | null>(null);
  const chartDialogBodyRef = useRef<HTMLDivElement | null>(null);
  const chartViewerRef = useRef<HTMLDivElement | null>(null);
  const chartViewerBodyRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setOpenChartSectionId(null);
    setChartViewerSectionId(null);
  }, [model.model_id]);

  useLayoutEffect(() => {
    if (!openChartSectionId && !chartViewerSectionId) {
      return;
    }

    const hasOpenChartSection = openChartSectionId
      ? chartIndicators.some((indicator) => indicator.id === openChartSectionId) ||
        manifestSections.some(
          (section) => section.id === openChartSectionId && section.id === "samples" && section.status === "missing",
        )
      : true;
    const hasOpenChartViewer = chartViewerSectionId
      ? chartIndicators.some((indicator) => indicator.id === chartViewerSectionId)
      : true;

    if (!hasOpenChartSection) {
      setOpenChartSectionId(null);
    }
    if (!hasOpenChartViewer) {
      setChartViewerSectionId(null);
    }
  }, [chartIndicators, chartViewerSectionId, openChartSectionId]);

  useLayoutEffect(() => {
    if (!openChartSectionId && !chartViewerSectionId) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        if (chartViewerSectionId) {
          setChartViewerSectionId(null);
          return;
        }
        setOpenChartSectionId(null);
      }
    };

    const scrollY = window.scrollY;
    const scrollbarCompensation = Math.max(0, window.innerWidth - document.documentElement.clientWidth);
    const previousBodyStyles = {
      overflow: document.body.style.overflow,
      position: document.body.style.position,
      top: document.body.style.top,
      left: document.body.style.left,
      right: document.body.style.right,
      width: document.body.style.width,
      paddingRight: document.body.style.paddingRight,
    };
    const previousHtmlOverscrollBehavior = document.documentElement.style.overscrollBehavior;

    document.body.style.overflow = "hidden";
    document.body.style.position = "fixed";
    document.body.style.top = `-${scrollY}px`;
    document.body.style.left = "0";
    document.body.style.right = "0";
    document.body.style.width = "100%";
    if (scrollbarCompensation > 0) {
      document.body.style.paddingRight = `${scrollbarCompensation}px`;
    }
    document.documentElement.style.overscrollBehavior = "none";
    window.addEventListener("keydown", handleKeyDown);

    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousBodyStyles.overflow;
      document.body.style.position = previousBodyStyles.position;
      document.body.style.top = previousBodyStyles.top;
      document.body.style.left = previousBodyStyles.left;
      document.body.style.right = previousBodyStyles.right;
      document.body.style.width = previousBodyStyles.width;
      document.body.style.paddingRight = previousBodyStyles.paddingRight;
      document.documentElement.style.overscrollBehavior = previousHtmlOverscrollBehavior;
      window.scrollTo(0, scrollY);
    };
  }, [chartViewerSectionId, openChartSectionId]);

  useEffect(() => {
    if (!openChartSectionId) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      chartDialogRef.current?.focus();
      chartDialogBodyRef.current?.scrollTo({ top: 0, left: 0, behavior: "auto" });
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [openChartSectionId]);

  useEffect(() => {
    if (!chartViewerSectionId) {
      return;
    }

    const frameId = window.requestAnimationFrame(() => {
      chartViewerRef.current?.focus();
      chartViewerBodyRef.current?.scrollTo({ top: 0, left: 0, behavior: "auto" });
    });

    return () => {
      window.cancelAnimationFrame(frameId);
    };
  }, [chartViewerSectionId]);

  const openChartIndicator =
    chartIndicators.find((indicator) => indicator.id === openChartSectionId) ??
    (() => {
      const missingSection = manifestSections.find(
        (section) => section.id === openChartSectionId && section.id === "samples" && section.status === "missing",
      );
      return missingSection ? buildMissingPreviewIndicator(missingSection) : null;
    })();
  const openChartViewerIndicator =
    chartIndicators.find((indicator) => indicator.id === chartViewerSectionId) ?? null;
  const viewerHasImageGallery = Boolean(
    openChartViewerIndicator?.images.length && !openChartViewerIndicator?.figures.length,
  );
  const viewerImageCount = openChartViewerIndicator?.images.length ?? 0;
  const activeViewerImage =
    viewerHasImageGallery && openChartViewerIndicator
      ? openChartViewerIndicator.images[Math.min(chartViewerImageIndex, openChartViewerIndicator.images.length - 1)]
      : null;
  const chartIndicatorById = useMemo(
    () => new Map(chartIndicators.map((indicator) => [indicator.id, indicator])),
    [chartIndicators],
  );
  const closeChartModal = () => setOpenChartSectionId(null);
  const closeChartViewer = () => setChartViewerSectionId(null);
  const openChartViewer = (sectionId: string) => {
    setOpenChartSectionId(null);
    setChartViewerSectionId(sectionId);
  };
  const showPreviousViewerImage = () => {
    if (!openChartViewerIndicator?.images.length) {
      return;
    }
    setChartViewerImageIndex((current) =>
      current === 0 ? openChartViewerIndicator.images.length - 1 : current - 1,
    );
  };
  const showNextViewerImage = () => {
    if (!openChartViewerIndicator?.images.length) {
      return;
    }
    setChartViewerImageIndex((current) =>
      current === openChartViewerIndicator.images.length - 1 ? 0 : current + 1,
    );
  };
  const scrollToSection = (ref: { current: HTMLElement | null }) => {
    ref.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  useEffect(() => {
    setChartViewerImageIndex(0);
  }, [chartViewerSectionId]);

  useEffect(() => {
    if (!viewerHasImageGallery || !openChartViewerIndicator || openChartViewerIndicator.images.length <= 1) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "ArrowLeft") {
        event.preventDefault();
        showPreviousViewerImage();
      } else if (event.key === "ArrowRight") {
        event.preventDefault();
        showNextViewerImage();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [openChartViewerIndicator, viewerHasImageGallery]);

  const notes = useMemo(
    () =>
      Array.from(
        new Set(
          [
            ...asArray(dashboard.manifest?.notes).map(String),
            ...asArray(overview?.notes).map(String),
            ...asArray(sourceAudit?.notes).map(String),
          ]
            .map((note) => note.trim())
            .filter(Boolean),
        ),
      ),
    [dashboard.manifest?.notes, overview?.notes, sourceAudit?.notes],
  );

  const artefactsAndChartsSections = useMemo(
    () =>
      [...manifestSections].sort(
        (left, right) =>
          (ARTEFACTS_AND_CHARTS_ORDER[left.id] ?? 99) - (ARTEFACTS_AND_CHARTS_ORDER[right.id] ?? 99),
      ),
    [manifestSections],
  );

  const primaryMetricCards = useMemo(
    () => buildPrimaryMetricCards(asRecord(primaryEvaluation?.splits), benchmarkRows, crossDatasetRows),
    [benchmarkRows, crossDatasetRows, primaryEvaluation],
  );

  const comparisonCards = useMemo<DashboardMetricCard[]>(() => {
    const cards: DashboardMetricCard[] = [];

    if (benchmarkRows.length) {
      const rankedRows = benchmarkRows
        .filter((row) => numericValue(row.f1_macro) !== null)
        .sort((left, right) => (numericValue(right.f1_macro) ?? 0) - (numericValue(left.f1_macro) ?? 0));
      const productionRow = rankedRows.find((row) => row.is_production === true);
      if (rankedRows.length) {
        cards.push({
          label: "Benchmark leader",
          value: formatNumber(rankedRows[0].f1_macro),
          hint: String(rankedRows[0].display_name ?? rankedRows[0].model ?? "Top run"),
        });
      }
      if (productionRow) {
        const rank = rankedRows.findIndex((row) => row === productionRow) + 1;
        cards.push({
          label: "Production rank",
          value: `${rank}/${rankedRows.length}`,
          hint: `${formatNumber(productionRow.f1_macro ?? productionRow.accuracy)} score`,
        });
      }
    }

    if (crossDatasetRows.length) {
      cards.push({
        label: "External datasets",
        value: String(crossDatasetRows.length),
        hint: "Cross-dataset rows loaded",
      });
    }

    if (learningCurves.length) {
      cards.push({
        label: "Learning curve",
        value: `${learningCurves.length}`,
        hint: "sample-size checkpoints",
      });
    }

    return cards.slice(0, 4);
  }, [benchmarkRows, crossDatasetRows, learningCurves.length, trainingHistory]);

  const overviewFacts = [
    {
      label: "Model",
      value: String(model.display_name || model.model_id),
    },
    {
      label: "Domain",
      value: String(metadataUi?.domain_display_name ?? model.domain),
    },
    {
      label: "Runtime",
      value: familyLabel(model),
    },
    {
      label: "Generated",
      value: formatTimestamp(dashboard.manifest?.generated_at ?? model.dashboard_generated_at),
    },
    {
      label: "Sections",
      value: `${model.dashboard_sections_available}/${model.dashboard_sections_total || manifestSections.length}`,
    },
    {
      label: "Charts",
      value: String(chartIndicators.length || "0"),
    },
  ];

  const primarySplitEntries = Object.entries(asRecord(primaryEvaluation?.splits) ?? {}).flatMap(
    ([split, metrics]) => {
      const metricRecord = asRecord(metrics);
      if (!metricRecord) {
        return [];
      }

      const visibleMetrics = Object.entries(metricRecord).filter(([, value]) => typeof value === "number");
      if (!visibleMetrics.length) {
        return [];
      }

      return [{ split, metrics: visibleMetrics }];
    },
  );

  const distributionTables = useMemo(
    () => buildDistributionTables(dashboard.documents),
    [dashboard.documents],
  );

  const description = String(
    metadataModel?.description ??
      model.description ??
      "Compact evaluation view for the currently selected production model.",
  );

  const artifactCountEntries = Object.entries(asRecord(sourceAudit?.artifact_counts) ?? {});
  const artifactInventoryTotal = artifactCountEntries.reduce((total, [, value]) => {
    return total + (typeof value === "number" ? value : 0);
  }, 0);
  const selectedSources = dashboard.manifest?.selected_sources ?? [];
  const sourceGroups = useMemo(() => buildSourceGroups(selectedSources), [selectedSources]);
  const runtimeArtifactGroups = useMemo(
    () => buildMetadataValueGroups(metadataArtifacts, METADATA_ARTIFACT_ORDER),
    [metadataArtifacts],
  );
  const experimentPathGroups = useMemo(
    () => buildMetadataValueGroups(artifactPaths),
    [artifactPaths],
  );
  const localFiles = useMemo(
    () => artifactsPresent.filter((path) => isVisibleArtifactPath(path)),
    [artifactsPresent],
  );
  const scanRoots = asArray(sourceAudit?.scanned_roots).map(String);
  const sourceAccordionItems = useMemo(
    () => buildMetadataSourceAccordionItems(sourceGroups),
    [sourceGroups],
  );
  const runtimeAccordionItems = useMemo(
    () => buildMetadataRuntimeAccordionItems(runtimeArtifactGroups, experimentPathGroups, localFiles),
    [experimentPathGroups, localFiles, runtimeArtifactGroups],
  );

  const metadataSummaryFacts = [
    { label: "Selected sources", value: String(selectedSources.length) },
    { label: "Source groups", value: String(sourceGroups.length) },
    { label: "Runtime groups", value: String(runtimeArtifactGroups.length) },
    { label: "Local files", value: String(localFiles.length) },
  ];

  const diagnosticsSummaryFacts = [
    { label: "Audit groups", value: String(artifactCountEntries.length) },
    { label: "Audited files", value: String(artifactInventoryTotal) },
    { label: "Scan roots", value: String(scanRoots.length) },
    { label: "Entrypoints", value: String(Object.keys(dashboard.manifest?.entrypoints ?? {}).length) },
  ];

  const handleArtefactsAndChartsClick = (section: DashboardSectionSummary) => {
    if (section.id === "summary") {
      onOpenModelInfo();
      return;
    }

    if (section.id === "metadata") {
      scrollToSection(metadataRef);
      return;
    }

    if (section.id === "samples" && section.status === "missing") {
      setOpenChartSectionId(section.id);
      return;
    }

    if (chartIndicatorById.has(section.id)) {
      setOpenChartSectionId(section.id);
      return;
    }

    if (section.id === "evaluation" || section.id === "benchmark" || section.id === "cross_dataset" || section.id === "samples") {
      scrollToSection(evaluationRef);
    }
  };

  return (
    <div className="models-dashboard__shell">
      <article ref={overviewRef} className="dashboard-block models-dashboard__hero-card">
        <div className="dashboard-block__header models-dashboard__hero-header">
          <div>
            <div className="dashboard-block__title-row">
              <div className="dashboard-block__eyebrow">Overview</div>
            </div>
          </div>

          <div className="dashboard-chip-row">
            <span className={`status-chip status-chip--${model.status}`}>{humanizeStatus(model.status)}</span>
          </div>
        </div>

        <div className="models-dashboard__fact-grid">
          {overviewFacts.map((fact) => (
            <div
              key={fact.label}
              className={`models-dashboard__fact-card models-dashboard__fact-card--${overviewFactTone(fact.label)}${
                isOverviewFactFeatured(fact.label) ? " models-dashboard__fact-card--featured" : ""
              }${isOverviewFactNumeric(fact.label) ? " models-dashboard__fact-card--numeric" : ""}`}
            >
              <span className="models-dashboard__fact-label">{fact.label}</span>
              <strong className="models-dashboard__fact-value">{fact.value}</strong>
            </div>
          ))}
        </div>

        {primaryMetricCards.length ? (
          <div className="dashboard-stat-grid models-dashboard__kpi-grid">
            {primaryMetricCards.map((card) => (
              <div key={card.label} className="dashboard-stat-card">
                <span>{card.label}</span>
                <strong>{card.value}</strong>
                {card.hint ? <small>{card.hint}</small> : null}
              </div>
            ))}
          </div>
        ) : null}

        {notes.length ? (
          <DashboardDetail title="Notes" meta={`${notes.length} item${notes.length === 1 ? "" : "s"}`}>
            <div className="dashboard-note-list">
              {notes.map((note) => (
                <div key={note} className="dashboard-note">
                  {note}
                </div>
              ))}
            </div>
          </DashboardDetail>
        ) : null}
      </article>

      <article className="dashboard-block models-dashboard__panel models-dashboard__panel--coverage">
        <div className="dashboard-block__header models-dashboard__coverage-header">
          <div>
            <div className="dashboard-block__title-row">
              <div className="dashboard-block__eyebrow">Artefacts &amp; Charts</div>
            </div>
          </div>
        </div>

        <div className="models-dashboard__section-grid">
          {artefactsAndChartsSections.map((section) => (
            <button
              key={section.id}
              type="button"
              className={`models-dashboard__section-item models-dashboard__section-item--${section.status}`}
              onClick={() => handleArtefactsAndChartsClick(section)}
            >
              <span className={`status-chip status-chip--${section.status} models-dashboard__section-status`}>
                {humanizeStatus(section.status)}
              </span>
              <strong>{formatCoverageSectionTitle(section)}</strong>
              <p>{formatCoverageSectionDescription(section)}</p>
            </button>
          ))}
        </div>
      </article>

      <div className="models-dashboard__body">
        <article
          ref={evaluationRef}
          className="dashboard-block models-dashboard__panel models-dashboard__panel--balanced models-dashboard__panel--evaluation"
        >
          <div className="dashboard-block__header">
            <div>
              <div className="dashboard-block__title-row">
                <div className="dashboard-block__eyebrow">Evaluation</div>
              </div>
            </div>
          </div>

          <div className="models-dashboard__panel-grid models-dashboard__panel-grid--compact">
            <div className="models-dashboard__subpanel models-dashboard__subpanel--compact models-dashboard__subpanel--evaluation-primary">
              <div className="models-dashboard__subpanel-header">
                <div className="dashboard-block__eyebrow">Primary</div>
              </div>

              {primarySplitEntries.length ? (
                <div className="models-dashboard__split-grid models-dashboard__split-grid--compact">
                  {primarySplitEntries.map((entry) => (
                    <div
                      key={entry.split}
                      className={`models-dashboard__split-card models-dashboard__split-card--compact models-dashboard__split-card--${splitTone(entry.split)}`}
                    >
                      <span>{formatSplitLabel(entry.split)}</span>
                      <div className="models-dashboard__metric-list">
                        {entry.metrics.map(([metric, value]) => (
                          <div key={`${entry.split}-${metric}`} className="models-dashboard__metric-row">
                            <span>{formatLabel(metric)}</span>
                            <strong>{formatNumber(value)}</strong>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="dashboard-empty">
                  <strong>No split-level metrics were attached.</strong>
                </div>
              )}
            </div>

            <div className="models-dashboard__subpanel models-dashboard__subpanel--compact models-dashboard__subpanel--evaluation-context">
              <div className="models-dashboard__subpanel-header">
                <div className="dashboard-block__eyebrow">Comparison</div>
              </div>

              {comparisonCards.length ? (
                <div className="dashboard-stat-grid models-dashboard__summary-grid models-dashboard__summary-grid--compact">
                  {comparisonCards.map((card) => (
                    <div
                      key={card.label}
                      className={`dashboard-stat-card models-dashboard__comparison-card models-dashboard__comparison-card--${comparisonTone(card.label)}`}
                    >
                      <span>{card.label}</span>
                      <strong>{card.value}</strong>
                      {card.hint ? <small>{card.hint}</small> : null}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="dashboard-empty">
                  <strong>No comparison artifacts attached.</strong>
                </div>
              )}
            </div>
          </div>

          <div className="models-dashboard__detail-stack models-dashboard__detail-stack--compact">
            {benchmarkRows.length ? (
              <DashboardDetail
                title="Benchmark leaderboard"
                className="models-dashboard__detail--evaluation"
              >
                <CompactTable
                  title="Benchmark leaderboard"
                  eyebrow="Benchmark"
                  rows={benchmarkRows}
                  limit={8}
                  hideHeader
                />
              </DashboardDetail>
            ) : null}

            {crossDatasetRows.length ? (
              <DashboardDetail
                title="Cross-dataset results"
                className="models-dashboard__detail--evaluation"
              >
                <CompactTable
                  title="External evaluation"
                  eyebrow="Cross dataset"
                  rows={crossDatasetRows}
                  hideHeader
                />
              </DashboardDetail>
            ) : null}

            {learningCurves.length ? (
              <DashboardDetail
                title="Learning curve points"
                className="models-dashboard__detail--evaluation"
              >
                <CompactTable
                  title="Learning curve checkpoints"
                  eyebrow="Curves"
                  rows={learningCurves}
                  limit={8}
                  hideHeader
                />
              </DashboardDetail>
            ) : null}

            {distributionTables.length ? (
              <DashboardDetail
                title="Distribution tables"
                className="models-dashboard__detail--evaluation"
              >
                <div className="models-dashboard__detail-stack">
                  {distributionTables.map((table) => (
                    <CompactTable
                      key={table.key}
                      title={table.title}
                      rows={table.rows}
                      variant="distribution"
                    />
                  ))}
                </div>
              </DashboardDetail>
            ) : null}

            {predictionSamples.length ? (
              <DashboardDetail
                title="Prediction samples"
                className="models-dashboard__detail--evaluation"
              >
                <SampleCards samples={predictionSamples} />
              </DashboardDetail>
            ) : null}
          </div>
        </article>
      </div>

      <article
        ref={metadataRef}
        className="dashboard-block models-dashboard__panel models-dashboard__panel--wide models-dashboard__panel--metadata"
      >
        <div className="dashboard-block__header">
          <div>
            <div className="dashboard-block__title-row">
              <div className="dashboard-block__eyebrow">Metadata</div>
              <DashboardHeaderInfo
                label="About Metadata"
                text="Source traceability, runtime assets, and optional diagnostics for the generated dashboard bundle."
              />
            </div>
          </div>
        </div>

        <div className="models-dashboard__metadata-grid models-dashboard__metadata-grid--summary">
          {metadataSummaryFacts.map((fact) => (
            <div key={fact.label} className="models-dashboard__mini-fact models-dashboard__mini-fact--summary">
              <span>{fact.label}</span>
              <strong>{fact.value}</strong>
            </div>
          ))}
        </div>

        <div className="models-dashboard__metadata-accordion-stack">
          <MetadataAccordion
            title="Sources"
            meta={`${sourceAccordionItems.length} group${sourceAccordionItems.length === 1 ? "" : "s"}`}
            tone="core"
            level="top"
          >
            {sourceAccordionItems.length ? (
              <div className="models-dashboard__metadata-accordion-children">
                {sourceAccordionItems.map((item) => (
                  <MetadataAccordion
                    key={item.id}
                    title={item.title}
                    meta={pluralize(item.paths.length, "source")}
                    tone="core"
                    level="nested"
                  >
                    <div className="models-dashboard__metadata-accordion-detail">
                      {item.groups.map((group) => (
                        <div
                          key={`${item.id}-${group.category}`}
                          className="models-dashboard__metadata-accordion-detail-group"
                        >
                          {item.groups.length > 1 ? (
                            <div className="dashboard-chip-row models-dashboard__label-row--metadata">
                              <span className="dashboard-mini-chip models-dashboard__label-chip">
                                {formatLabel(group.category)}
                              </span>
                            </div>
                          ) : null}
                          {group.reason ? <p className="models-dashboard__detail-copy">{group.reason}</p> : null}
                          <div className="models-dashboard__path-stack">
                            {group.paths.map((path) => (
                              <div key={`${group.category}-${path}`} className="models-dashboard__path-chip">
                                {path}
                              </div>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </MetadataAccordion>
                ))}
              </div>
            ) : (
              <div className="dashboard-empty">
                <strong>No selected source records were attached.</strong>
              </div>
            )}
          </MetadataAccordion>

          <MetadataAccordion
            title="Runtime assets"
            meta={`${runtimeAccordionItems.length} group${runtimeAccordionItems.length === 1 ? "" : "s"}`}
            tone="runtime"
            level="top"
          >
            {runtimeAccordionItems.length ? (
              <div className="models-dashboard__metadata-accordion-children">
                {runtimeAccordionItems.map((item) => (
                  <MetadataAccordion
                    key={item.id}
                    title={item.title}
                    meta={pluralize(item.values.length, item.isLocalFiles ? "file" : "path")}
                    tone="runtime"
                    level="nested"
                  >
                    {item.isLocalFiles ? (
                      <div className="models-dashboard__path-stack models-dashboard__path-stack--single-column">
                        {item.values.map((path) => (
                          <div key={path} className="models-dashboard__path-chip">
                            {path}
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="models-dashboard__metadata-accordion-detail">
                        {item.groups.map((group) => (
                          <div
                            key={`${item.id}-${group.key}`}
                            className="models-dashboard__metadata-accordion-detail-group"
                          >
                            {item.groups.length > 1 ||
                            formatLabel(group.key).toLowerCase() !== item.title.toLowerCase() ? (
                              <div className="dashboard-chip-row models-dashboard__label-row--metadata">
                                <span className="dashboard-mini-chip models-dashboard__label-chip">
                                  {formatLabel(group.key)}
                                </span>
                              </div>
                            ) : null}
                            <div className="models-dashboard__path-stack">
                              {group.values.map((value) => (
                                <div key={`${group.key}-${value}`} className="models-dashboard__path-chip">
                                  {value}
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </MetadataAccordion>
                ))}
              </div>
            ) : (
              <div className="dashboard-empty">
                <strong>No runtime assets were attached.</strong>
              </div>
            )}
          </MetadataAccordion>
          <MetadataAccordion
            title="Diagnostics"
            meta={artifactCountEntries.length ? `${artifactCountEntries.length} groups` : "advanced"}
            tone="diagnostic"
            level="top"
          >
            <div className="models-dashboard__metadata-grid models-dashboard__metadata-grid--summary">
              {diagnosticsSummaryFacts.map((fact) => (
                <div key={fact.label} className="models-dashboard__mini-fact models-dashboard__mini-fact--summary">
                  <span>{fact.label}</span>
                  <strong>{fact.value}</strong>
                </div>
              ))}
            </div>

            {artifactCountEntries.length ? (
              <div className="models-dashboard__metadata-section">
                <div className="dashboard-block__eyebrow">Artifact scan</div>
                <div className="dashboard-chip-row models-dashboard__artifact-chip-cloud">
                  {artifactCountEntries.map(([key, value]) => (
                    <span key={key} className="dashboard-mini-chip">
                      {formatLabel(key)}: {String(value)}
                    </span>
                  ))}
                </div>
              </div>
            ) : null}

            {scanRoots.length ? (
              <div className="models-dashboard__metadata-section">
                <div className="dashboard-block__eyebrow">Scan roots</div>
                <pre className="dashboard-kv__code">{stringifyStructuredValue(scanRoots)}</pre>
              </div>
            ) : null}

            {dashboard.manifest ? (
              <div className="models-dashboard__metadata-section">
                <div className="dashboard-block__eyebrow">Manifest entrypoints</div>
                <pre className="dashboard-kv__code">
                  {stringifyStructuredValue(dashboard.manifest.entrypoints)}
                </pre>
              </div>
            ) : null}

            {metadataModel ? (
              <div className="models-dashboard__metadata-section">
                <div className="dashboard-block__eyebrow">Model metadata JSON</div>
                <pre className="dashboard-kv__code">{stringifyStructuredValue(metadataModel)}</pre>
              </div>
            ) : null}
          </MetadataAccordion>
        </div>
      </article>

      {openChartIndicator && typeof document !== "undefined"
        ? createPortal(
            <div
              className="models-dashboard__chart-modal"
              role="presentation"
              onClick={(event) => {
                if (event.target === event.currentTarget) {
                  closeChartModal();
                }
              }}
            >
              <div className="models-dashboard__chart-shell">
                <div
                  ref={chartDialogRef}
                  className="models-dashboard__chart-dialog"
                  role="dialog"
                  aria-modal="true"
                  aria-labelledby="model-dashboard-chart-title"
                  tabIndex={-1}
                >
                  <div className="models-dashboard__chart-dialog-header">
                    <div className="models-dashboard__chart-dialog-title">
                      <h3 id="model-dashboard-chart-title">{openChartIndicator.title}</h3>
                      <DashboardHeaderInfo
                        label={`About ${openChartIndicator.title}`}
                        text={openChartIndicator.descriptionText}
                      />
                    </div>

                    <div className="models-dashboard__chart-dialog-actions">
                      <span className={`status-chip status-chip--${openChartIndicator.status}`}>
                        {humanizeStatus(openChartIndicator.status)}
                      </span>
                      {openChartIndicator.kind === "plotly" ? (
                        <span
                          className={`models-dashboard__chart-kind models-dashboard__chart-kind--${openChartIndicator.kind}`}
                        >
                          {chartKindLabel(openChartIndicator.kind)}
                        </span>
                      ) : null}
                      <button
                        type="button"
                        className="mini-button models-dashboard__chart-close"
                        aria-label={`Close ${openChartIndicator.title} preview`}
                        onClick={closeChartModal}
                      >
                        Close
                      </button>
                    </div>
                  </div>

                  <div ref={chartDialogBodyRef} className="models-dashboard__chart-dialog-scroll">
                    {openChartIndicator.reason ? (
                      <div className="dashboard-footnote">{openChartIndicator.reason}</div>
                    ) : null}

                    {openChartIndicator.kind === "missing" ? (
                      <div className="dashboard-empty models-dashboard__chart-empty">
                        <strong>{openChartIndicator.title} is not available.</strong>
                        <p>
                          {openChartIndicator.reason ??
                            openChartIndicator.description ??
                            "This section does not have a chart or image artifact attached yet."}
                        </p>
                      </div>
                    ) : (
                      <div className="models-dashboard__chart-modal-body">
                        <div className="models-dashboard__chart-preview-summary">
                          <span className="dashboard-mini-chip">
                            {summarizeChartAssets(openChartIndicator)}
                          </span>
                          {openChartIndicator.figures.length ? (
                            <span className="dashboard-mini-chip">
                              {openChartIndicator.figures.reduce(
                                (total, figure) => total + countFigureTraces(figure),
                                0,
                              )}{" "}
                              trace
                              {openChartIndicator.figures.reduce(
                                (total, figure) => total + countFigureTraces(figure),
                                0,
                              ) === 1
                                ? ""
                                : "s"}
                            </span>
                          ) : null}
                          {openChartIndicator.images.length ? (
                            <span className="dashboard-mini-chip">
                              {openChartIndicator.images.length} preview image
                              {openChartIndicator.images.length === 1 ? "" : "s"}
                            </span>
                          ) : null}
                        </div>

                        {openChartIndicator.figures.length ? (
                          <div className="models-dashboard__chart-preview-grid">
                            {openChartIndicator.figures.map((figure) => (
                              <article key={figure.id} className="models-dashboard__chart-preview-card">
                                <div className="plotly-card__eyebrow">Plotly figure</div>
                                <strong>{figure.title ?? figure.id}</strong>
                                <p>{summarizeFigureTraceKinds(figure)}</p>
                                <div className="dashboard-chip-row">
                                  <span className="dashboard-mini-chip">
                                    {countFigureTraces(figure)} trace
                                    {countFigureTraces(figure) === 1 ? "" : "s"}
                                  </span>
                                  <span className="dashboard-mini-chip">{figure.id}</span>
                                </div>
                              </article>
                            ))}
                          </div>
                        ) : null}

                        {openChartIndicator.images.length ? (
                          <div className="models-dashboard__chart-preview-grid models-dashboard__chart-preview-grid--images">
                            {openChartIndicator.images.map((image) => (
                              <figure key={image.path} className="dashboard-image-card models-dashboard__chart-preview-image">
                                <img src={image.url} alt={image.title} />
                                <figcaption>{image.title}</figcaption>
                              </figure>
                            ))}
                          </div>
                        ) : null}

                        <div className="models-dashboard__chart-preview-actions">
                          <button
                            type="button"
                            className="primary-button"
                            onClick={() => openChartViewer(openChartIndicator.id)}
                          >
                            Open full-screen viewer
                          </button>
                          <button
                            type="button"
                            className="mini-button"
                            onClick={closeChartModal}
                          >
                            Keep browsing dashboard
                          </button>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}

      {openChartViewerIndicator && typeof document !== "undefined"
        ? createPortal(
            <div
              className="models-dashboard__viewer-modal"
              role="presentation"
              onClick={(event) => {
                if (event.target === event.currentTarget) {
                  closeChartViewer();
                }
              }}
            >
              <div
                ref={chartViewerRef}
                className="models-dashboard__viewer-shell"
                role="dialog"
                aria-modal="true"
                aria-labelledby="model-dashboard-viewer-title"
                tabIndex={-1}
              >
                <div className="models-dashboard__viewer-header">
                  <div>
                    <div className="dashboard-block__title-row">
                      <h3 id="model-dashboard-viewer-title">{openChartViewerIndicator.title}</h3>
                      <DashboardHeaderInfo
                        label={`About ${openChartViewerIndicator.title} viewer`}
                        text={openChartViewerIndicator.descriptionText}
                      />
                    </div>
                  </div>

                  <div className="models-dashboard__viewer-actions">
                    <span className="dashboard-mini-chip">
                      {summarizeChartAssets(openChartViewerIndicator)}
                    </span>
                    <span className={`status-chip status-chip--${openChartViewerIndicator.status}`}>
                      {humanizeStatus(openChartViewerIndicator.status)}
                    </span>
                    <button
                      type="button"
                      className="mini-button models-dashboard__viewer-close"
                      onClick={closeChartViewer}
                    >
                      Close
                    </button>
                  </div>
                </div>

                <div ref={chartViewerBodyRef} className="models-dashboard__viewer-body">
                  {openChartViewerIndicator.reason ? (
                    <div className="dashboard-footnote">{openChartViewerIndicator.reason}</div>
                  ) : null}

                  {openChartViewerIndicator.figures.length ? (
                    <div className="models-dashboard__viewer-figures">
                      {openChartViewerIndicator.figures.map((figure) => (
                        <PlotlyFigure key={figure.id} figure={figure} variant="viewer" />
                      ))}
                    </div>
                  ) : null}

                  {viewerHasImageGallery && activeViewerImage ? (
                    <div className="models-dashboard__viewer-gallery">
                      {viewerImageCount > 1 ? (
                        <div className="models-dashboard__viewer-gallery-controls">
                          <button
                            type="button"
                            className="mini-button"
                            onClick={showPreviousViewerImage}
                          >
                            Previous
                          </button>
                          <span className="dashboard-mini-chip">
                            {chartViewerImageIndex + 1} / {viewerImageCount}
                          </span>
                          <button
                            type="button"
                            className="mini-button"
                            onClick={showNextViewerImage}
                          >
                            Next
                          </button>
                        </div>
                      ) : null}

                      <figure className="dashboard-image-card models-dashboard__viewer-gallery-figure">
                        <img
                          className="models-dashboard__viewer-gallery-media"
                          src={activeViewerImage.url}
                          alt={activeViewerImage.title}
                        />
                        <figcaption>{activeViewerImage.title}</figcaption>
                      </figure>

                      {viewerImageCount > 1 ? (
                        <div className="models-dashboard__viewer-gallery-strip">
                          {openChartViewerIndicator.images.map((image, index) => (
                            <button
                              key={image.path}
                              type="button"
                              className={`models-dashboard__viewer-gallery-thumb${
                                index === chartViewerImageIndex
                                  ? " models-dashboard__viewer-gallery-thumb--active"
                                  : ""
                              }`}
                              onClick={() => setChartViewerImageIndex(index)}
                              aria-label={`Show ${image.title}`}
                            >
                              <img src={image.url} alt={image.title} />
                              <span>{image.title}</span>
                            </button>
                          ))}
                        </div>
                      ) : null}
                    </div>
                  ) : null}

                  {openChartViewerIndicator.images.length && !viewerHasImageGallery ? (
                    <div className="models-dashboard__viewer-images">
                      {openChartViewerIndicator.images.map((image) => (
                        <figure key={image.path} className="dashboard-image-card">
                          <img src={image.url} alt={image.title} />
                          <figcaption>{image.title}</figcaption>
                        </figure>
                      ))}
                    </div>
                  ) : null}
                </div>
              </div>
            </div>,
            document.body,
          )
        : null}
    </div>
  );
}
