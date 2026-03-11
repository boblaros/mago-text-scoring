import { useCatalog } from "../hooks/useCatalog";
import type { AppView } from "../components/TopNav";

interface InfoPageProps {
  view: Exclude<AppView, "home" | "models">;
}

type SemanticTone = "setup" | "gate" | "runtime" | "control";

const PAGE_COPY = {
  "how-it-works": {
    eyebrow: "How it works",
    title: "A modular text analysis workspace.",
    intro:
      "The service routes one text input through active language analysis modules and returns a compact result set for each domain.",
  },
  models: {
    eyebrow: "Models",
    title: "Model management surface.",
    intro:
      "This area is reserved for uploading, inspecting, and managing models, with room for technical metadata and active model details.",
  },
  about: {
    eyebrow: "About",
    title: "Built as part of a university text mining project.",
    intro:
      "The project was developed by students of Università Cattolica del Sacro Cuore (UCSC) as part of the Text Mining program.",
  },
} as const;

const HOW_IT_WORKS_STEPS = [
  {
    id: "1",
    eyebrow: "Setup",
    title: "Choose domains",
    detail: "Pick the scoring modules you want in the stack.",
    tone: "setup" as const,
  },
  {
    id: "2",
    eyebrow: "Models",
    title: "Register models",
    detail: "Add model assets and confirm they are ready.",
    tone: "gate" as const,
  },
  {
    id: "3",
    eyebrow: "Runtime",
    title: "Activate modules",
    detail: "Move approved models into the live workspace.",
    tone: "runtime" as const,
  },
  {
    id: "4",
    eyebrow: "Home",
    title: "Analyze text",
    detail: "Run one input and get a unified result.",
    tone: "runtime" as const,
  },
] as const;

const MODEL_FLOW_STEPS = [
  {
    id: "01",
    eyebrow: "Assets",
    title: "Load files",
    detail: "Weights, config, tokenizer",
    tone: "setup" as const,
  },
  {
    id: "02",
    eyebrow: "Metadata",
    title: "Inspect setup",
    detail: "Version, lineage, parameters",
    tone: "gate" as const,
  },
  {
    id: "03",
    eyebrow: "Checks",
    title: "Validate readiness",
    detail: "Run technical review",
    tone: "gate" as const,
  },
  {
    id: "04",
    eyebrow: "Activation",
    title: "Approve runtime",
    detail: "Move module to active",
    tone: "runtime" as const,
  },
  {
    id: "05",
    eyebrow: "Home",
    title: "Use in scoring",
    detail: "Available in live analysis",
    tone: "runtime" as const,
  },
] as const;

const MODEL_FLOW_SUMMARY = [
  {
    label: "Input",
    value: "Raw model assets",
    tone: "setup" as const,
  },
  {
    label: "Gate",
    value: "Validation + activation",
    tone: "gate" as const,
  },
  {
    label: "Output",
    value: "Active module in Home",
    tone: "runtime" as const,
  },
] as const;

const VALUE_PILLARS = [
  {
    eyebrow: "Low-code",
    title: "One surface, less pipeline wiring.",
    copy: "Operate multiple text-scoring domains with less glue code.",
  },
  {
    eyebrow: "Modular",
    title: "Change the stack without rebuilding the UI.",
    copy: "Keep the set narrow or grow it as new modules mature.",
  },
  {
    eyebrow: "Extensible",
    title: "Future ecosystems can plug into the workflow.",
    copy: "Hugging Face, Kaggle, and similar ecosystems can extend it later.",
  },
] as const;

const ABOUT_ROOTS_FACTS = [
  {
    label: "Institution",
    value: "Università Cattolica del Sacro Cuore",
  },
  {
    label: "Campus",
    value: "Milan campus",
  },
  {
    label: "Course",
    value: "Data Visualization and Text Mining",
  },
  {
    label: "Format",
    value: "Course project",
  },
  {
    label: "Professor",
    value: "Andrea Belli",
  },
] as const;

const ABOUT_CREATORS = [
  {
    name: "Matteo",
    href: "https://www.linkedin.com/in/matteo-moltrasio-05a927247/",
  },
  {
    name: "Aleksandra",
    href: "https://www.linkedin.com/in/aleksandra-laricheva/",
  },
  {
    name: "Georgii",
    href: "https://github.com/boblaros",
  },
  {
    name: "Orlando",
    href: "https://www.linkedin.com/in/orlandopb/",
  },
] as const;

const ABOUT_PILLARS = [
  {
    eyebrow: "NLP",
    title: "Apply language analysis in a usable interface.",
    copy:
      "MAGO was built to move beyond notebooks and isolated experiments by turning text-scoring skills into a workspace that can be operated, compared, and extended.",
  },
  {
    eyebrow: "Visualization",
    title: "Keep model output readable and decision-friendly.",
    copy:
      "Data visualization principles shape the interface so outputs stay compact, legible, and easier to interpret across multiple domains instead of disappearing into raw logs.",
  },
  {
    eyebrow: "Modularity",
    title: "Treat the product as a platform concept, not a fixed demo.",
    copy:
      "The workspace is organized around modules, so teams can keep the scope narrow, add new domains, or replace models as the system evolves.",
  },
] as const;

const ABOUT_OPEN_SOURCE_POINTS = [
  {
    id: "01",
    tone: "primary",
    copy: (
      <>
        The <span className="about-inline-accent">codebase</span> is fully open and can be
        cloned, inspected, and adapted to new text-analysis workflows.
      </>
    ),
  },
  {
    id: "02",
    tone: "green",
    copy: (
      <>
        Teams can reuse the <span className="about-inline-accent">modular registry</span>,
        interface structure, and scoring flow as a base for internal NLP systems.
      </>
    ),
  },
  {
    id: "03",
    tone: "amber",
    copy: (
      <>
        Researchers and students can treat MAGO as a{" "}
        <span className="about-inline-accent">reference implementation</span> for building their
        own model-driven workspaces.
      </>
    ),
  },
] as const;

const ABOUT_USE_CASES = [
  {
    eyebrow: "Business",
    title: "Text, message, review, and news analysis.",
    copy:
      "A modular scoring layer can support product feedback analysis, internal communication review, moderation pipelines, and monitoring of public-facing text streams.",
  },
  {
    eyebrow: "Internal tooling",
    title: "Reusable NLP building blocks.",
    copy:
      "The interface can act as a base for lightweight internal tools where multiple classifiers need to run together without rebuilding the workflow for every new task.",
  },
  {
    eyebrow: "Research",
    title: "Experimentation, teaching, and model comparison.",
    copy:
      "MAGO also works as an academic or exploratory surface for testing modules, demonstrating architecture choices, and organizing applied NLP experiments.",
  },
] as const;

function getToneClass(tone: SemanticTone) {
  return `semantic-tone--${tone}`;
}

function getDomainTone(domain: string): SemanticTone {
  if (domain === "abuse") {
    return "control";
  }

  if (domain === "age") {
    return "runtime";
  }

  if (domain === "sentiment") {
    return "gate";
  }

  return "setup";
}

export function InfoPage({ view }: InfoPageProps) {
  const { domains } = useCatalog();
  const copy = PAGE_COPY[view];
  const pageClassName = [
    "info-page",
    view === "how-it-works" ? "info-page--workflow" : "",
    view === "about" ? "info-page--about" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <section className={pageClassName}>
      {view !== "how-it-works" && view !== "about" ? (
        <div className="panel info-hero">
          <div className="panel__eyebrow">{copy.eyebrow}</div>
          <h1>{copy.title}</h1>
          <p>{copy.intro}</p>
        </div>
      ) : null}

      {view === "how-it-works" ? (
        <>
          <div className="panel info-hero info-hero--workflow">
            <div className="workflow-hero">
              <div className="workflow-hero__copy">
                <h1>Configure domains, register models, activate modules, analyze text.</h1>
                <p>
                  A modular text-analysis workspace built for low-code scoring. Think of it as a
                  Power BI-style layer for text analysis: configurable domains, one operational
                  surface, and minimal plumbing between model assets and runtime use.
                </p>

                <div className="info-badge-row">
                  <span className={`info-badge ${getToneClass("setup")}`}>Low-code workflow</span>
                  <span className={`info-badge ${getToneClass("gate")}`}>Expandable modules</span>
                  <span className={`info-badge ${getToneClass("runtime")}`}>Unified runtime</span>
                </div>
              </div>

              <div className="workflow-hero__signal panel">
                <div className="workflow-hero__label">Active domain workspace</div>
                <div className="workflow-hero__caption">
                  Active modules stay configurable, so the domain stack can stay focused, shrink,
                  or expand as the workspace evolves.
                </div>

                <div className="workflow-domain-row">
                  {domains.slice(0, 4).map((domain) => (
                    <span
                      key={domain.domain}
                      className={`workflow-domain-chip ${getToneClass(getDomainTone(domain.domain))}`}
                    >
                      {domain.display_name}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>

          <section className="panel workflow-panel">
            <div className="workflow-panel__header">
              <div className="panel__eyebrow">Workflow roadmap</div>
              <h2>A four-step roadmap from setup to scoring.</h2>
            </div>

            <div className="workflow-rail">
              {HOW_IT_WORKS_STEPS.flatMap((step, index) => {
                const nextTone = HOW_IT_WORKS_STEPS[index + 1]?.tone ?? step.tone;
                const items = [
                  <article key={step.id} className={`workflow-step ${getToneClass(step.tone)}`}>
                    <div className="workflow-step__meta">
                      <div className="workflow-step__index">{step.id}</div>
                      <div className="workflow-step__eyebrow">{step.eyebrow}</div>
                    </div>
                    <div className="workflow-step__body">
                      <h3>{step.title}</h3>
                      <p className="workflow-step__detail">{step.detail}</p>
                    </div>
                  </article>,
                ];

                if (index < HOW_IT_WORKS_STEPS.length - 1) {
                  items.push(
                    <div
                      key={`${step.id}-arrow`}
                      className={`workflow-arrow activation-arrow ${getToneClass(nextTone)}`}
                      aria-hidden="true"
                    >
                      →
                    </div>,
                  );
                }

                return items;
              })}
            </div>
          </section>

          <section className="panel models-panel">
            <div className="models-panel__header">
              <div className="panel__eyebrow">Models</div>
              <h2>One technical flow from model files to live runtime.</h2>
            </div>

            <div className="models-panel__roadmap">
              {MODEL_FLOW_STEPS.flatMap((step, index) => {
                const nextTone = MODEL_FLOW_STEPS[index + 1]?.tone ?? step.tone;
                const items = [
                  <article key={step.id} className={`models-panel__step ${getToneClass(step.tone)}`}>
                    <div className="models-panel__step-meta">
                      <div className="models-panel__step-index">{step.id}</div>
                      <div className="models-panel__step-eyebrow">{step.eyebrow}</div>
                    </div>
                    <div className="models-panel__step-body">
                      <h3>{step.title}</h3>
                      <p>{step.detail}</p>
                    </div>
                  </article>,
                ];

                if (index < MODEL_FLOW_STEPS.length - 1) {
                  items.push(
                    <div
                      key={`${step.id}-arrow`}
                      className={`models-panel__arrow activation-arrow ${getToneClass(nextTone)}`}
                      aria-hidden="true"
                    >
                      →
                    </div>,
                  );
                }

                return items;
              })}
            </div>

            <div className="models-panel__summary">
              {MODEL_FLOW_SUMMARY.map((item) => (
                <div key={item.label} className={`models-panel__signal ${getToneClass(item.tone)}`}>
                  <div className="models-panel__signal-label">{item.label}</div>
                  <div className="models-panel__signal-value">{item.value}</div>
                </div>
              ))}
            </div>
          </section>

          <section className="panel value-panel">
            <div className="value-panel__header">
              <div className="panel__eyebrow">Why it matters</div>
              <h2>One modular workspace for text scoring with minimum code.</h2>
            </div>

            <div className="value-grid">
              {VALUE_PILLARS.map((item) => (
                <article key={item.title} className="value-card">
                  <div className="value-card__eyebrow">{item.eyebrow}</div>
                  <h3>{item.title}</h3>
                  <p>{item.copy}</p>
                </article>
              ))}
            </div>
          </section>
        </>
      ) : null}

      {view === "about" ? (
        <>
          <div className="about-hero-grid">
            <section className="panel about-hero">
              <div className="about-hero__main">
                <div className="about-hero__header">
                <div className="panel__eyebrow">About MAGO</div>
                <h1 className="about-hero__title">
                  A <span className="about-hero__title-accent">modular</span> NLP{" "}
                  <span className="about-hero__title-accent">workspace</span> with academic roots
                  and product ambition.
                </h1>
              </div>

                <div className="about-hero__tail">
                  <p className="about-supporting-line about-hero__lead">
                    MAGO is named from the initials of the four people who created the project.
                  </p>

                  <div className="about-creator-row" aria-label="Project creators">
                    {ABOUT_CREATORS.map((creator) => (
                      <a
                        key={creator.name}
                        className="about-creator-pill"
                        href={creator.href}
                        target="_blank"
                        rel="noreferrer"
                      >
                        {creator.name}
                      </a>
                    ))}
                  </div>
                </div>
              </div>
            </section>

            <aside className="panel about-facts">
              <div className="about-facts__header">
                <div className="panel__eyebrow">Academic origin</div>
                <p className="about-facts__caption">
                  The context behind the first release
                </p>
              </div>

              <div className="about-facts__list">
                {ABOUT_ROOTS_FACTS.map((fact) => (
                  <div key={fact.label} className="about-fact">
                    <div className="about-fact__label">{fact.label}</div>
                    <div className="about-fact__value">{fact.value}</div>
                  </div>
                ))}
              </div>
            </aside>
          </div>

          <section className="panel about-section about-section--purpose">
            <div className="about-section__header">
              <div className="panel__eyebrow">Why MAGO exists</div>
              <h2 className="about-section__title--wide">
                To connect NLP, visualization, and modular product thinking in one workspace.
              </h2>
              <p className="about-section__caption">
                Built as a modular interface where applied NLP, readable outputs, and flexible
                product architecture can work as one coherent system.
              </p>
            </div>

            <div className="value-grid about-value-grid">
              {ABOUT_PILLARS.map((item) => (
                <article key={item.title} className="value-card about-value-card">
                  <div className="value-card__eyebrow">{item.eyebrow}</div>
                  <h3>{item.title}</h3>
                  <p>{item.copy}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel about-section about-section--open">
            <div className="about-section__header">
              <div className="panel__eyebrow">Open-source and reuse</div>
              <h2 className="about-section__title--single-line">
                Built in public and meant to be adapted.
              </h2>
              <p className="about-section__caption about-section__caption--single-line">
                MAGO is fully open-source and can serve as a reusable foundation for modular NLP
                systems.
              </p>
            </div>

            <div className="about-list">
              {ABOUT_OPEN_SOURCE_POINTS.map((item) => (
                <article
                  key={item.id}
                  className={`about-list__item about-list__item--${item.tone}`}
                >
                  <div className="about-list__meta">
                    <span className="about-list__index">{item.id}</span>
                  </div>
                  <p className="about-list__copy">{item.copy}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="panel about-section about-section--use">
            <div className="about-section__header">
              <div className="panel__eyebrow">Where it can be used</div>
              <h2 className="about-section__title--single-line">
                Practical enough for business needs, clear enough for research use.
              </h2>
              <p className="about-section__caption about-section__caption--single-line">
                Designed to read as a reusable product surface for operational analysis while still
                making sense as a research and experimentation environment.
              </p>
            </div>

            <div className="about-use-grid">
              {ABOUT_USE_CASES.map((item) => (
                <article key={item.title} className="about-use-card">
                  <div className="about-use-card__eyebrow">{item.eyebrow}</div>
                  <h3>{item.title}</h3>
                  <p>{item.copy}</p>
                </article>
              ))}
            </div>
          </section>
        </>
      ) : null}
    </section>
  );
}
