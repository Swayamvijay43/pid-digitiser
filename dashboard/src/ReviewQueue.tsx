import { useState } from "react";
import { api } from "./api";
import type { Decision, Item } from "./types";
import { TAG_PATTERNS } from "./types";

interface Props {
  runId: string;
  pending: Item[];
  selected: string | null;
  onSelect: (detId: string) => void;
  onSubmitted: () => void;
}

type Draft = { action: Decision["action"]; tag: string };

export function ReviewQueue({ runId, pending, selected, onSelect, onSubmitted }: Props) {
  const [drafts, setDrafts] = useState<Record<string, Draft>>(() =>
    Object.fromEntries(pending.map((p) => [p.det.det_id, { action: "confirm", tag: p.tag ?? "" }])),
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const set = (id: string, d: Partial<Draft>) => setDrafts((all) => ({ ...all, [id]: { ...all[id], ...d } }));

  const invalid = pending.filter((p) => {
    const d = drafts[p.det.det_id];
    return d.action !== "reject" && !TAG_PATTERNS[p.det.cls].test(d.tag.trim().toUpperCase());
  });

  async function submit() {
    setBusy(true);
    setError(null);
    const decisions: Decision[] = pending.map((p) => {
      const d = drafts[p.det.det_id];
      const tag = d.tag.trim().toUpperCase();
      if (d.action === "reject") return { det_id: p.det.det_id, action: "reject" };
      return tag === (p.tag ?? "")
        ? { det_id: p.det.det_id, action: "confirm" }
        : { det_id: p.det.det_id, action: "correct", tag };
    });
    try {
      await api.resume(runId, decisions);
      onSubmitted();
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="queue">
      <div className="queue-head">
        <h3>Review queue · {pending.length}</h3>
        <button className="primary" disabled={busy || invalid.length > 0} onClick={submit}
          title={invalid.length ? `${invalid.length} tag(s) fail the tag grammar` : "Resume the paused LangGraph run"}>
          {busy ? "Resuming…" : "Submit review & resume"}
        </button>
      </div>
      {invalid.length > 0 && <p className="warn">{invalid.length} tag(s) don't match the tag grammar yet.</p>}
      {error && <p className="warn">{error}</p>}
      {pending.map((p) => {
        const id = p.det.det_id;
        const d = drafts[id];
        const bad = d.action !== "reject" && !TAG_PATTERNS[p.det.cls].test(d.tag.trim().toUpperCase());
        return (
          <div key={id} className={`card ${selected === id ? "selected" : ""}`} onClick={() => onSelect(id)}>
            <img className="crop" src={api.cropUrl(runId, id)} alt={`crop ${id}`} />
            <div className="card-body">
              <div className="row">
                <b>{p.det.cls.replace("_", " ")}</b>
                <span className="muted">det {p.det.det_conf.toFixed(2)}</span>
              </div>
              <div className="reason">{p.reason}</div>
              <div className="attempts">
                {p.attempts.map((a, i) => (
                  <span key={i} className={a.valid ? "ok" : "no"} title={a.raw ?? ""}>
                    {a.rung}: {a.tag ?? "∅"}
                  </span>
                ))}
              </div>
              <div className="row">
                <input value={d.tag} disabled={d.action === "reject"} className={bad ? "bad" : ""}
                  onChange={(e) => set(id, { tag: e.target.value, action: "confirm" })} placeholder="tag" />
                <label>
                  <input type="checkbox" checked={d.action === "reject"}
                    onChange={(e) => set(id, { action: e.target.checked ? "reject" : "confirm" })} />
                  not a symbol
                </label>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
