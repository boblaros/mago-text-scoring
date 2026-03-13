import { useEffect, useRef, useState } from "react";
import type { DashboardFigure } from "../../types/contracts";

type PlotlyInstance = {
  newPlot: (
    element: HTMLElement,
    data: unknown[],
    layout?: Record<string, unknown>,
    config?: Record<string, unknown>,
  ) => Promise<unknown>;
  purge: (element: HTMLElement) => void;
  Plots?: {
    resize: (element: HTMLElement) => void;
  };
};

declare global {
  interface Window {
    Plotly?: PlotlyInstance;
  }
}

const PLOTLY_SRC = "https://cdn.plot.ly/plotly-2.35.2.min.js";

let plotlyPromise: Promise<PlotlyInstance> | null = null;

function loadPlotlyLibrary() {
  if (window.Plotly) {
    return Promise.resolve(window.Plotly);
  }

  if (plotlyPromise) {
    return plotlyPromise;
  }

  plotlyPromise = new Promise<PlotlyInstance>((resolve, reject) => {
    const existing = document.querySelector<HTMLScriptElement>("script[data-plotly-loader='true']");
    if (existing) {
      existing.addEventListener("load", () => {
        if (window.Plotly) {
          resolve(window.Plotly);
          return;
        }
        reject(new Error("Plotly loaded without exposing window.Plotly."));
      });
      existing.addEventListener("error", () => {
        reject(new Error("Failed to load Plotly."));
      });
      return;
    }

    const script = document.createElement("script");
    script.src = PLOTLY_SRC;
    script.async = true;
    script.dataset.plotlyLoader = "true";
    script.onload = () => {
      if (window.Plotly) {
        resolve(window.Plotly);
        return;
      }
      reject(new Error("Plotly loaded without exposing window.Plotly."));
    };
    script.onerror = () => reject(new Error("Failed to load Plotly."));
    document.head.appendChild(script);
  });

  return plotlyPromise;
}

function buildThemedFigure(source: Record<string, unknown>) {
  const data = Array.isArray(source.data) ? source.data : [];
  const layout = (source.layout ?? {}) as Record<string, unknown>;
  const font = (layout.font ?? {}) as Record<string, unknown>;
  const { width: _width, height: _height, ...layoutWithoutFixedSize } = layout;
  const themedAxes = Object.fromEntries(
    Object.entries(layoutWithoutFixedSize).map(([key, value]) => {
      if (/^xaxis\d*$/.test(key)) {
        return [key, buildThemedAxis(value, "x")];
      }
      if (/^yaxis\d*$/.test(key)) {
        return [key, buildThemedAxis(value, "y")];
      }
      return [key, value];
    }),
  );

  return {
    data,
    layout: {
      ...themedAxes,
      autosize: true,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {
        ...font,
        family: '"Sora", sans-serif',
        color: "#d9e2f1",
      },
      margin: buildThemedMargin(layout.margin),
    },
    config: {
      displaylogo: false,
      responsive: true,
      modeBarButtonsToRemove: [
        "lasso2d",
        "select2d",
        "toggleSpikelines",
        "autoScale2d",
      ],
      ...(typeof source.config === "object" && source.config !== null ? source.config : {}),
    },
  };
}

function waitForNextFrame() {
  return new Promise<void>((resolve) => {
    window.requestAnimationFrame(() => resolve());
  });
}

async function waitForVisiblePlotContainer(element: HTMLElement) {
  for (let attempt = 0; attempt < 12; attempt += 1) {
    await waitForNextFrame();
    const { width, height } = element.getBoundingClientRect();
    if (width > 0 && height > 0) {
      return;
    }
  }
}

async function waitForPlotTypography() {
  if (typeof document === "undefined" || !("fonts" in document)) {
    return;
  }

  const fontSet = document.fonts;
  try {
    await Promise.allSettled([
      fontSet.load('400 12px "Sora"'),
      fontSet.load('600 12px "Sora"'),
      fontSet.ready,
    ]);
  } catch {
    // Plot rendering can proceed with fallback fonts if the custom face is unavailable.
  }
}

function buildThemedMargin(source: unknown) {
  const margin =
    typeof source === "object" && source !== null ? (source as Record<string, unknown>) : {};

  const numeric = (key: "l" | "r" | "t" | "b" | "pad", fallback: number) =>
    typeof margin[key] === "number" ? margin[key] : fallback;

  return {
    ...margin,
    l: Math.max(numeric("l", 0), 72),
    r: Math.max(numeric("r", 0), 24),
    t: Math.max(numeric("t", 0), 60),
    b: Math.max(numeric("b", 0), 78),
    pad: Math.max(numeric("pad", 0), 8),
  };
}

function buildThemedAxis(source: unknown, axis: "x" | "y") {
  const axisLayout =
    typeof source === "object" && source !== null ? (source as Record<string, unknown>) : {};
  const title =
    typeof axisLayout.title === "object" && axisLayout.title !== null
      ? (axisLayout.title as Record<string, unknown>)
      : typeof axisLayout.title === "string"
        ? { text: axisLayout.title }
        : null;

  return {
    ...axisLayout,
    automargin: true,
    color: "#c1cde0",
    gridcolor: "rgba(154, 175, 194, 0.12)",
    zerolinecolor: "rgba(154, 175, 194, 0.16)",
    tickfont: {
      ...(typeof axisLayout.tickfont === "object" && axisLayout.tickfont !== null
        ? (axisLayout.tickfont as Record<string, unknown>)
        : {}),
      color: "#c1cde0",
    },
    title: title
      ? {
          ...title,
          standoff:
            typeof title.standoff === "number"
              ? Math.max(title.standoff, axis === "x" ? 18 : 14)
              : axis === "x"
                ? 18
                : 14,
          font: {
            ...(typeof title.font === "object" && title.font !== null
              ? (title.font as Record<string, unknown>)
              : {}),
            color: "#d9e2f1",
          },
        }
      : axisLayout.title,
  };
}

interface PlotlyFigureProps {
  figure: DashboardFigure;
  variant?: "default" | "modal";
}

export function PlotlyFigure({
  figure,
  variant = "default",
}: PlotlyFigureProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const element = containerRef.current;
    if (!element) {
      return;
    }

    setError(null);
    setIsLoading(true);

    let resizeHandler: (() => void) | null = null;
    let resizeObserver: ResizeObserver | null = null;
    let resizeTimeoutId: number | null = null;
    let resizeFrameId: number | null = null;
    let followUpFrameId: number | null = null;

    void loadPlotlyLibrary()
      .then(async (plotly) => {
        if (cancelled || !containerRef.current) {
          return;
        }
        await waitForPlotTypography();
        await waitForVisiblePlotContainer(containerRef.current);
        if (cancelled || !containerRef.current) {
          return;
        }
        const themedFigure = buildThemedFigure(figure.figure);
        const resizePlot = () => {
          if (containerRef.current && plotly.Plots) {
            plotly.Plots.resize(containerRef.current);
          }
        };
        const scheduleResize = () => {
          if (resizeFrameId !== null) {
            window.cancelAnimationFrame(resizeFrameId);
          }
          if (followUpFrameId !== null) {
            window.cancelAnimationFrame(followUpFrameId);
          }
          if (resizeTimeoutId !== null) {
            window.clearTimeout(resizeTimeoutId);
          }

          resizeFrameId = window.requestAnimationFrame(() => {
            resizePlot();
            followUpFrameId = window.requestAnimationFrame(() => {
              resizePlot();
            });
          });
          resizeTimeoutId = window.setTimeout(() => {
            resizePlot();
          }, variant === "modal" ? 140 : 80);
        };
        return plotly
          .newPlot(
            containerRef.current,
            themedFigure.data,
            themedFigure.layout,
            themedFigure.config,
          )
          .then(() => {
            if (cancelled || !containerRef.current) {
              return;
            }
            setIsLoading(false);
            resizeHandler = () => {
              scheduleResize();
            };
            window.addEventListener("resize", resizeHandler);
            if (typeof ResizeObserver !== "undefined") {
              resizeObserver = new ResizeObserver(() => {
                scheduleResize();
              });
              if (containerRef.current.parentElement) {
                resizeObserver.observe(containerRef.current.parentElement);
              }
              resizeObserver.observe(containerRef.current);
            }
            scheduleResize();
          });
      })
      .catch((loadError) => {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to render Plotly figure.");
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
      if (resizeHandler) {
        window.removeEventListener("resize", resizeHandler);
      }
      if (resizeObserver) {
        resizeObserver.disconnect();
      }
      if (resizeFrameId !== null) {
        window.cancelAnimationFrame(resizeFrameId);
      }
      if (followUpFrameId !== null) {
        window.cancelAnimationFrame(followUpFrameId);
      }
      if (resizeTimeoutId !== null) {
        window.clearTimeout(resizeTimeoutId);
      }
      if (window.Plotly && element) {
        window.Plotly.purge(element);
      }
    };
  }, [figure, variant]);

  return (
    <article className={`plotly-card${variant === "modal" ? " plotly-card--modal" : ""}`}>
      <div className="plotly-card__header">
        <div>
          <div className="plotly-card__eyebrow">Plotly</div>
          <h4>{figure.title ?? figure.id}</h4>
        </div>
      </div>

      {error ? <div className="inline-alert">{error}</div> : null}
      <div className="plotly-card__stage">
        {isLoading ? <div className="plotly-card__placeholder">Loading chart…</div> : null}
        <div
          ref={containerRef}
          className={`plotly-card__canvas${isLoading ? " plotly-card__canvas--hidden" : ""}`}
        />
      </div>
    </article>
  );
}
