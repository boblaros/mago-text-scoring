export type AppView = "home" | "how-it-works" | "models" | "about";

interface TopNavProps {
  currentView: AppView;
  onChange: (view: AppView) => void;
  githubUrl: string;
}

const NAV_ITEMS: Array<{ id: AppView; label: string }> = [
  { id: "home", label: "Home" },
  { id: "how-it-works", label: "How it works" },
  { id: "models", label: "Models" },
  { id: "about", label: "About" },
];

export function TopNav({ currentView, onChange, githubUrl }: TopNavProps) {
  return (
    <header className="top-nav panel">
      <div className="top-nav__left">
        <button className="top-nav__brand" type="button" onClick={() => onChange("home")}>
          MAGO
        </button>

        <nav className="top-nav__links" aria-label="Primary">
          {NAV_ITEMS.map((item) => (
            <button
              key={item.id}
              className={`top-nav__link${currentView === item.id ? " top-nav__link--active" : ""}`}
              type="button"
              onClick={() => onChange(item.id)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </div>

      <a
        className="top-nav__github"
        href={githubUrl}
        target="_blank"
        rel="noreferrer"
        aria-label="Open GitHub repository"
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path
            fill="currentColor"
            d="M12 2C6.48 2 2 6.59 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.1.68-.22.68-.49 0-.24-.01-1.03-.02-1.87-2.78.62-3.37-1.21-3.37-1.21-.45-1.18-1.11-1.49-1.11-1.49-.9-.63.07-.62.07-.62 1 .07 1.52 1.05 1.52 1.05.88 1.56 2.32 1.11 2.88.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.09 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.31.1-2.72 0 0 .84-.28 2.75 1.05A9.3 9.3 0 0 1 12 6.84c.85 0 1.71.12 2.51.36 1.91-1.33 2.75-1.05 2.75-1.05.55 1.41.2 2.46.1 2.72.64.72 1.03 1.63 1.03 2.75 0 3.96-2.34 4.82-4.58 5.08.36.32.68.95.68 1.92 0 1.39-.01 2.5-.01 2.84 0 .27.18.6.69.49A10.28 10.28 0 0 0 22 12.25C22 6.59 17.52 2 12 2Z"
          />
        </svg>
        <span>GitHub</span>
      </a>
    </header>
  );
}
