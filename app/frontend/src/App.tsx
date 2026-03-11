import { useState } from "react";
import { BackgroundGrid } from "./components/BackgroundGrid";
import { TopNav, type AppView } from "./components/TopNav";
import { CatalogProvider } from "./hooks/useCatalog";
import { AnalyzePage } from "./pages/AnalyzePage";
import { InfoPage } from "./pages/InfoPage";
import { ModelsPage } from "./pages/ModelsPage";

const GITHUB_URL = "https://github.com/boblaros/mago-text-scoring";

export default function App() {
  const [view, setView] = useState<AppView>("home");

  return (
    <CatalogProvider>
      <main className="app-shell">
        <BackgroundGrid />

        <div className="app-shell__inner">
          <TopNav currentView={view} onChange={setView} githubUrl={GITHUB_URL} />

          <div className="app-stage">
            {view === "home" ? (
              <AnalyzePage onNavigateToModels={() => setView("models")} />
            ) : view === "models" ? (
              <ModelsPage />
            ) : (
              <InfoPage view={view} />
            )}
          </div>
        </div>
      </main>
    </CatalogProvider>
  );
}
