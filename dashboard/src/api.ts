import type { Decision, Run } from "./types";

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status}: ${await res.text()}`);
  return res.json() as Promise<T>;
}

export const api = {
  runs: () => fetch("/api/runs").then((r) => json<Run[]>(r)),
  run: (id: string) => fetch(`/api/runs/${id}`).then((r) => json<Run>(r)),
  samples: () => fetch("/api/samples").then((r) => json<string[]>(r)),
  startSample: (sheet: string, variant: "scanned" | "vector") =>
    fetch(`/api/runs/sample/${sheet}?variant=${variant}`, { method: "POST" }).then((r) => json<Run>(r)),
  upload: (file: File) => {
    const body = new FormData();
    body.append("file", file);
    return fetch("/api/runs", { method: "POST", body }).then((r) => json<Run>(r));
  },
  resume: (id: string, decisions: Decision[]) =>
    fetch(`/api/runs/${id}/resume`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(decisions),
    }).then((r) => json<Run>(r)),
  pageUrl: (id: string, page: number) => `/api/runs/${id}/pages/${page}.png`,
  cropUrl: (id: string, detId: string) => `/api/runs/${id}/crops/${detId}.png`,
  mtoUrl: (id: string) => `/api/runs/${id}/mto.csv`,
};
