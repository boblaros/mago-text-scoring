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

  return {
    data,
    layout: {
      ...layout,
      paper_bgcolor: "transparent",
      plot_bgcolor: "transparent",
      font: {
        ...font,
        family: '"Sora", sans-serif',
        color: "#d9e2f1",
      },
      margin: {
        l: 56,
        r: 18,
        t: 52,
        b: 46,
        ...(typeof layout.margin === "object" && layout.margin !== null ? layout.margin : {}),
      },
      xaxis: {
        ...(typeof layout.xaxis === "object" && layout.xaxis !== null ? layout.xaxis : {}),
        color: "#c1cde0",
        gridcolor: "rgba(154, 175, 194, 0.12)",
        zerolinecolor: "rgba(154, 175, 194, 0.16)",
      },
      yaxis: {
        ...(typeof layout.yaxis === "object" && layout.yaxis !== null ? layout.yaxis : {}),
        color: "#c1cde0",
        gridcolor: "rgba(154, 175, 194, 0.12)",
        zerolinecolor: "rgba(154, 175, 194, 0.16)",
      },
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

interface PlotlyFigureProps {
  figure: DashboardFigure;
}

export function PlotlyFigure({ figure }: PlotlyFigureProps) {
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

    void loadPlotlyLibrary()
      .then((plotly) => {
        if (cancelled || !containerRef.current) {
          return;
        }
        const themedFigure = buildThemedFigure(figure.figure);
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
            resizeHandler = () => {
              if (containerRef.current && plotly.Plots) {
                plotly.Plots.resize(containerRef.current);
              }
            };
            window.addEventListener("resize", resizeHandler);
            setIsLoading(false);
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
      if (window.Plotly && element) {
        window.Plotly.purge(element);
      }
    };
  }, [figure]);

  return (
    <article className="plotly-card">
      <div className="plotly-card__header">
        <div>
          <div className="plotly-card__eyebrow">Plotly</div>
          <h4>{figure.title ?? figure.id}</h4>
        </div>
      </div>

      {error ? <div className="inline-alert">{error}</div> : null}
      {isLoading ? <div className="plotly-card__placeholder">Loading chart…</div> : null}
      <div
        ref={containerRef}
        className={`plotly-card__canvas${isLoading ? " plotly-card__canvas--hidden" : ""}`}
      />
    </article>
  );
}
