import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type PropsWithChildren,
} from "react";
import { fetchCatalogSnapshot, fetchDomainCatalog } from "../services/api";
import type { CatalogSnapshotResponse, DomainCatalogEntry } from "../types/contracts";

const FALLBACK_DOMAINS: DomainCatalogEntry[] = [
  {
    domain: "sentiment",
    display_name: "Sentiment",
    color_token: "sentiment",
    group: "sentiment-core",
    active_model_id: "sentiment",
    active_model_name: "Loading registry",
    active_model_version: null,
    model_count: 1,
    models: [],
  },
  {
    domain: "complexity",
    display_name: "Complexity",
    color_token: "complexity",
    group: "complexity-core",
    active_model_id: "complexity",
    active_model_name: "Loading registry",
    active_model_version: null,
    model_count: 1,
    models: [],
  },
  {
    domain: "age",
    display_name: "Age",
    color_token: "age",
    group: "age-core",
    active_model_id: "age",
    active_model_name: "Loading registry",
    active_model_version: null,
    model_count: 1,
    models: [],
  },
  {
    domain: "abuse",
    display_name: "Cyberbullying Classification",
    color_token: "abuse",
    group: "abuse-core",
    active_model_id: "abuse",
    active_model_name: "Loading registry",
    active_model_version: null,
    model_count: 1,
    models: [],
  },
];

interface CatalogContextValue {
  domains: DomainCatalogEntry[];
  managementDomains: DomainCatalogEntry[];
  isLoading: boolean;
  error: string | null;
  managementWarning: string | null;
  managementReady: boolean;
  usingFallbackCatalog: boolean;
  refreshCatalog: () => Promise<void>;
  applySnapshot: (snapshot: CatalogSnapshotResponse) => void;
}

const CatalogContext = createContext<CatalogContextValue | null>(null);

export function CatalogProvider({ children }: PropsWithChildren) {
  const [domains, setDomains] = useState<DomainCatalogEntry[]>([]);
  const [managementDomains, setManagementDomains] = useState<DomainCatalogEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [managementWarning, setManagementWarning] = useState<string | null>(null);
  const [managementReady, setManagementReady] = useState(false);

  const applySnapshot = (snapshot: CatalogSnapshotResponse) => {
    setDomains(snapshot.active_domains);
    setManagementDomains(snapshot.management_domains);
    setError(null);
    setManagementWarning(null);
    setManagementReady(true);
  };

  const refreshCatalog = async () => {
    try {
      const snapshot = await fetchCatalogSnapshot();
      applySnapshot(snapshot);
    } catch (snapshotError) {
      try {
        const fallback = await fetchDomainCatalog();
        setDomains(fallback.domains);
        setManagementDomains(fallback.domains);
        setError(null);
        setManagementWarning(
          snapshotError instanceof Error
            ? snapshotError.message
            : "Model management API is unavailable.",
        );
        setManagementReady(false);
      } catch (catalogError) {
        setError(
          catalogError instanceof Error ? catalogError.message : "Failed to load catalog.",
        );
        setManagementWarning(null);
        setManagementReady(false);
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    void refreshCatalog();
  }, []);

  const value = useMemo(
    () => ({
      domains: domains.length ? domains : FALLBACK_DOMAINS,
      managementDomains: managementDomains.length ? managementDomains : FALLBACK_DOMAINS,
      isLoading,
      error,
      managementWarning,
      managementReady,
      usingFallbackCatalog: !domains.length && !managementDomains.length,
      refreshCatalog,
      applySnapshot,
    }),
    [domains, error, isLoading, managementDomains, managementReady, managementWarning],
  );

  return <CatalogContext.Provider value={value}>{children}</CatalogContext.Provider>;
}

export function useCatalog() {
  const context = useContext(CatalogContext);
  if (!context) {
    throw new Error("useCatalog must be used inside CatalogProvider.");
  }
  return context;
}
