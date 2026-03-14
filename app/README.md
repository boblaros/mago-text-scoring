# Mago Text Scoring App

Docker-first text analysis application built inside `/app`, with a FastAPI backend, a React/Vite frontend, and a config-driven model registry that auto-discovers models from `/app/app-models`.

## What is included

- `backend/`: FastAPI API, model discovery, artifact resolution, inference runner plugins, aggregate summary logic, and backend tests.
- `frontend/`: React + TypeScript + Vite dashboard with animated routing, lane activation, progressive result reveal, and a premium inference-console UI.
- `app-models/`: Local model artifacts scanned at runtime through each folder's `model-config.yaml`.
- `shared/`: Contract and payload examples for the frontend and future integrations.
- `docker-compose.yml`: Local development entrypoint for backend and frontend containers.

## Alpha domains

- `sentiment`
- `complexity`
- `age`
- `abuse`

The backend is not hardcoded to those four domains. It scans manifests, groups models by canonical domain, selects the active model per domain, and exposes the registry through `GET /api/v1/domains`.

## Registry design

- Model discovery is automatic: every `model-config.yaml` under `app-models` is registered.
- Artifact paths are resolved defensively: the loader tolerates manifests that still reference nested export paths while the actual folder is flattened.
- One domain can expose multiple model definitions.
- The active model is picked by `is_active` and `priority`, so a future admin/settings UI can switch models without changing inference code.
- Inference backends are plugin-based: the current build ships a Hugging Face sequence-classification runner and a PyTorch BiLSTM+attention runner.

## Dashboard builder

- Existing dashboards still load from `model_dir/dashboard/dashboard-manifest.json`.
- New config-driven dashboards can be built with `scripts/build_model_dashboard.py`.
- The workflow and `model-config.yaml` shape are documented in [model-dashboard-workflow.md](/Users/georgijkutivadze/projects/mago-text-scoring/app/docs/model-dashboard-workflow.md).

## API

- `GET /api/v1/health`
- `GET /api/v1/domains`
- `POST /api/v1/analyze`

Example request is in [sample-request.json](/Users/georgijkutivadze/projects/mago-text-scoring/app/shared/examples/sample-request.json).

## Run with Docker

From `/Users/georgijkutivadze/projects/mago-text-scoring/app`, run:

```bash
docker compose up --build
```

Frontend: [http://localhost:5173](http://localhost:5173)  
Backend: [http://localhost:8000/api/v1/health](http://localhost:8000/api/v1/health)

Use `.env.example` only if you want to override defaults such as ports, preload behavior, or the frontend API base URL.

## Notes

- The complexity model uses a custom BiLSTM + additive attention runner reconstructed from the repository's training code.
- The shipped abuse manifest is canonicalized from `abuse-detection` to `abuse` so it participates in the shared UI lane model.
- The frontend always renders the lane architecture, even before the live registry finishes loading, so the UI remains stable during startup.
