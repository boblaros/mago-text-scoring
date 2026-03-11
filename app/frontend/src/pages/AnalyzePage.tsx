import { useMemo, useState } from "react";
import { LogsPanel } from "../components/LogsPanel";
import { RegistryDeck } from "../components/RegistryDeck";
import { ResultsPanel } from "../components/ResultsPanel";
import { TextWorkbench } from "../components/TextWorkbench";
import { useAnalysisPipeline } from "../hooks/useAnalysisPipeline";
import { useCatalog } from "../hooks/useCatalog";

const DEFAULT_TEXT =
  "The interface is clean, the message is clear, and the overall tone feels optimistic without sounding simplistic.";

interface AnalyzePageProps {
  onNavigateToModels: () => void;
}

export function AnalyzePage({ onNavigateToModels }: AnalyzePageProps) {
  const [text, setText] = useState(DEFAULT_TEXT);
  const { domains, error: catalogError } = useCatalog();
  const noActiveDomains = domains.length === 0;
  const {
    analysis,
    phase,
    lanes,
    logs,
    error: analysisError,
    analyze,
    reset,
  } = useAnalysisPipeline(domains);

  const combinedError = useMemo(
    () => analysisError ?? catalogError,
    [analysisError, catalogError],
  );

  return (
    <div className="console-layout">
      <div className="layout-output">
        <ResultsPanel
          domains={domains}
          lanes={lanes}
          onNavigateToModels={onNavigateToModels}
        />
      </div>

      <div className="layout-sidebar">
        <RegistryDeck
          domains={domains}
          phase={phase}
          analysis={analysis}
          error={combinedError}
        />
      </div>

      <div className="layout-input">
        <TextWorkbench
          value={text}
          phase={phase}
          disabled={
            noActiveDomains ||
            phase === "routing" ||
            phase === "running" ||
            phase === "revealing"
          }
          onChange={setText}
          onAnalyze={() => void analyze(text)}
          onReset={() => {
            setText("");
            reset();
          }}
        />
      </div>

      <div className="layout-log">
        <LogsPanel logs={logs} />
      </div>
    </div>
  );
}
