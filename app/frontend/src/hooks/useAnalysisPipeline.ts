import { startTransition, useEffect, useState } from "react";
import {
  LANE_BOOT_DELAY_MS,
  RESULT_REVEAL_DELAY_MS,
  ROUTING_DELAY_MS,
} from "../animations/pipeline";
import { analyzeText } from "../services/api";
import type {
  AnalysisResponse,
  DomainCatalogEntry,
  DomainResult,
  LaneState,
  LogEntry,
  PipelinePhase,
} from "../types/contracts";

const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

function createLogEntry(
  stage: LogEntry["stage"],
  tone: LogEntry["tone"],
  message: string,
  options?: Pick<LogEntry, "domain" | "latency_ms">,
): LogEntry {
  const timestamp = Date.now();
  return {
    id: `${timestamp}-${Math.random().toString(16).slice(2)}`,
    timestamp,
    stage,
    tone,
    message,
    domain: options?.domain,
    latency_ms: options?.latency_ms ?? null,
  };
}

function bootLogs(): LogEntry[] {
  return [
    {
      id: "boot",
      timestamp: Date.now(),
      stage: "system",
      tone: "neutral",
      message: "Ready for input.",
      latency_ms: null,
    },
  ];
}

function formatLatency(result: DomainResult) {
  return `${result.display_name} resolved`;
}

function buildIdleLanes(domains: DomainCatalogEntry[]) {
  return Object.fromEntries(
    domains.map((domain) => [
      domain.domain,
      {
        domain: domain.domain,
        status: "idle",
      } satisfies LaneState,
    ]),
  );
}

export function useAnalysisPipeline(domains: DomainCatalogEntry[]) {
  const [phase, setPhase] = useState<PipelinePhase>("idle");
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>(bootLogs);
  const [lanes, setLanes] = useState<Record<string, LaneState>>(() => buildIdleLanes(domains));

  useEffect(() => {
    setLanes((current) =>
      Object.fromEntries(
        domains.map((domain) => [
          domain.domain,
          current[domain.domain] ?? {
            domain: domain.domain,
            status: "idle",
          },
        ]),
      ),
    );
  }, [domains]);

  const reset = () => {
    setPhase("idle");
    setAnalysis(null);
    setError(null);
    setLogs(bootLogs());
    setLanes(buildIdleLanes(domains));
  };

  const analyze = async (text: string) => {
    setPhase("idle");
    setAnalysis(null);
    setError(null);
    setLogs([
      ...bootLogs(),
      createLogEntry("accept", "info", "Text received."),
    ]);
    setLanes(buildIdleLanes(domains));
    setPhase("routing");
    await sleep(ROUTING_DELAY_MS);

    for (const domain of domains) {
      setLanes((current) => ({
        ...current,
        [domain.domain]: {
          ...(current[domain.domain] ?? {
            domain: domain.domain,
          }),
          status: "queued",
        },
      }));
      setLogs((current) => [
        ...current.slice(-11),
        createLogEntry("queued", "neutral", "Queued.", {
          domain: domain.display_name,
        }),
      ]);
      await sleep(LANE_BOOT_DELAY_MS);
    }

    try {
      const responsePromise = analyzeText({
        text,
        domains: domains.map((domain) => domain.domain),
      });
      setPhase("running");
      setLogs((current) => [
        ...current.slice(-11),
        createLogEntry(
          "dispatch",
          "info",
          `Running ${domains.length} domains.`,
        ),
      ]);

      for (const domain of domains) {
        setLanes((current) => ({
          ...current,
          [domain.domain]: {
            ...(current[domain.domain] ?? {
              domain: domain.domain,
            }),
            status: "running",
          },
        }));
        await sleep(Math.max(90, Math.round(LANE_BOOT_DELAY_MS * 0.75)));
      }

      const response = await responsePromise;
      startTransition(() => {
        setAnalysis(response);
      });
      setPhase("revealing");
      setLogs((current) => [
        ...current.slice(-11),
        createLogEntry("dispatch", "info", "Results received."),
      ]);

      for (const result of response.results) {
        setLanes((current) => ({
          ...current,
          [result.domain]: {
            domain: result.domain,
            status: "resolved",
            result,
          },
        }));
        setLogs((current) => [
          ...current.slice(-11),
          createLogEntry("resolved", "success", formatLatency(result), {
            domain: result.display_name,
            latency_ms: result.latency_ms,
          }),
        ]);
        await sleep(RESULT_REVEAL_DELAY_MS);
      }

      setPhase("done");
      setLogs((current) => [
        ...current.slice(-11),
        createLogEntry("aggregate", "success", "Analysis complete."),
      ]);
      setError(null);
    } catch (err) {
      const message =
        err instanceof Error ? err.message : "The analysis request failed.";
      setPhase("error");
      setError(message);
      setLogs((current) => [
        ...current.slice(-11),
        createLogEntry("error", "error", message),
      ]);
      setLanes((current) =>
        Object.fromEntries(
          Object.entries(current).map(([domain, lane]) => [
            domain,
            {
              ...lane,
              status: "error",
            },
          ]),
        ),
      );
    }
  };

  return {
    analysis,
    phase,
    lanes,
    logs,
    error,
    analyze,
    reset,
  };
}
