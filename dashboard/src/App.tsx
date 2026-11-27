import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { PageViewer, STATUS_COLOUR } from "./PageViewer";
import { ReviewQueue } from "./ReviewQueue";
import type { Item, Run } from "./types";

export default function App() {
  const [runs, setRuns] = useState<Run[]>([]);
  const [samples, setSamples] = useState<string[]>([]);
  const [current, setCurrent] = useState<string | null>(null);

  const refresh = useCallback(() => api.runs().then(setRuns).catch(() => {}), []);
  useEffect(() => {
    refresh();
    api.samples().then(setSamples).catch(() => {});
  }, [refresh]);

  async function start(p: Promise<Run>) {
    const r = await p;
    await refresh();
    setCurrent(r.run_id);
  }

  return (
    <div className="app">
      <aside>
        <h2>P&amp;ID digitiser</h2>
        <label className="upload">
          Upload PDF
          <input type="file" accept="application/pdf" hidden
            onChange={(e) => e.target.files?.[0] && start(api.upload(e.target.files[0]))} />
        </label>
        {samples.length > 0 && (
          <select defaultValue="" onChange={(e) => {
            const [sheet, variant] = e.target.value.split("|");
            if (sheet) start(api.startSample(sheet, variant as "scanned" | "vector"));
            e.target.value = "";
          }}>
            <option value="">Run a test sheet…</option>
            {samples.flatMap((s) => [
              <option key={s + "s"} value={`${s}|scanned`}>{s} · scanned</option>,
              <option key={s + "v"} value={`${s}|vector`}>{s} · vector</option>,
            ])}
          </select>
        )}
        <ul className="runs">
          {runs.map((r) => (
            <li key={r.run_id} className={r.run_id === current ? "active" : ""} onClick={() => setCurrent(r.run_id)}>
              <span>{r.filename}</span>
              <span className={`pill ${r.status}`}>{r.status.replace("_", " ")}</span>
            </li>
          ))}
        </ul>
      </aside>
      <main>{current ? <RunView key={current} runId={current} onChange={refresh} /> : <Empty />}</main>
    </div>
  );
}

function Empty() {
  return <div className="empty">Upload a P&amp;ID PDF or pick a test sheet to start a run.</div>;
}

function RunView({ runId, onChange }: { runId: string; onChange: () => void }) {
  const [run, setRun] = useState<Run | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const load = useCallback(() => api.run(runId).then(setRun), [runId]);
  useEffect(() => {
    load();
  }, [load]);
  useEffect(() => {
    if (run?.status !== "processing") return;
    const t = setInterval(() => load().then(onChange), 1500);
    return () => clearInterval(t);
  }, [run?.status, load, onChange]);

  if (!run) return <div className="empty">Loading…</div>;
  if (run.status === "processing") return <div className="empty">Scanning, detecting and reading tags…</div>;
  if (run.status === "failed") return <div className="empty warn">Run failed: {run.error}</div>;

  const items: Item[] = run.items ?? [];
  const pending = run.pending ?? [];
  const page = run.pages?.[0];
  const counts = items.reduce<Record<string, number>>((acc, i) => ({ ...acc, [i.status]: (acc[i.status] ?? 0) + 1 }), {});
  const sel = items.find((i) => i.det.det_id === selected);

  return (
    <div className="run">
      <div className="stats">
        <h3>{run.filename}</h3>
        <span>{items.length} symbols</span>
        {Object.entries(counts).map(([s, n]) => (
          <span key={s} style={{ color: STATUS_COLOUR[s as keyof typeof STATUS_COLOUR] }}>{n} {s.replace("_", " ")}</span>
        ))}
        {run.exports && <a className="primary" href={api.mtoUrl(runId)} download>Download MTO (CSV)</a>}
      </div>
      <div className="split">
        {page && <PageViewer page={page} imageUrl={api.pageUrl(runId, 0)} items={items}
          selected={selected} onSelect={setSelected} />}
        <div className="side">
          {pending.length > 0 ? (
            <ReviewQueue runId={runId} pending={pending} selected={selected} onSelect={setSelected}
              onSubmitted={() => { load(); onChange(); }} />
          ) : (
            <div className="queue"><h3>No items waiting for review</h3></div>
          )}
          {sel && (
            <div className="detail">
              <h4>{sel.det.det_id}</h4>
              <p><b>{sel.tag ?? "no tag"}</b> · {sel.det.cls} · {sel.status}</p>
              <p className="muted">{sel.reason}</p>
              <p className="muted">line: {sel.line_number ?? "—"} · confidence {sel.confidence.toFixed(2)}</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
