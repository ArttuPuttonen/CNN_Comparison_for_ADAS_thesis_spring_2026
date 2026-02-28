# Frontend (React + TypeScript + Tailwind + Recharts)

Minimal one-page thesis demo UI for comparing CNN model predictions.

## Prerequisites

- Node.js 18+
- Backend API running at `http://localhost:8000` (or set `VITE_API_URL`)

## Setup

```bash
cd frontend
npm install
```

## Environment variable

Create a `.env` file in `frontend/`:

```bash
VITE_API_URL=http://localhost:8000
```

If not set, the app defaults to `http://localhost:8000`.

## Run

```bash
cd frontend
npm run dev
```

Open the printed local Vite URL (usually `http://localhost:5173`).

## Build

```bash
cd frontend
npm run build
npm run preview
```
