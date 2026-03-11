import type { DomainCatalogEntry } from "../../types/contracts";

export interface DomainTheme {
  accent: string;
  glow: string;
  panel: string;
}

const THEMES: Record<string, DomainTheme> = {
  sentiment: {
    accent: "#f8bd63",
    glow: "rgba(248, 189, 99, 0.35)",
    panel: "rgba(57, 34, 5, 0.72)",
  },
  complexity: {
    accent: "#72b6ff",
    glow: "rgba(114, 182, 255, 0.35)",
    panel: "rgba(11, 25, 46, 0.72)",
  },
  age: {
    accent: "#40d3b9",
    glow: "rgba(64, 211, 185, 0.35)",
    panel: "rgba(5, 28, 24, 0.72)",
  },
  abuse: {
    accent: "#ff7d7d",
    glow: "rgba(255, 125, 125, 0.35)",
    panel: "rgba(46, 10, 10, 0.72)",
  },
};

export function getDomainTheme(domain: Pick<DomainCatalogEntry, "domain" | "color_token">): DomainTheme {
  return THEMES[domain.domain] ?? THEMES[domain.color_token] ?? {
    accent: "#a7b3c8",
    glow: "rgba(167, 179, 200, 0.3)",
    panel: "rgba(18, 24, 36, 0.72)",
  };
}

