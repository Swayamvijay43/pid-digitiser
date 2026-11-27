import { useState } from "react";
import type { Item, Page, Status } from "./types";

export const STATUS_COLOUR: Record<Status, string> = {
  auto_accepted: "#16a34a",
  needs_review: "#ea580c",
  human_confirmed: "#2563eb",
  human_corrected: "#7c3aed",
  rejected: "#9ca3af",
};

interface Props {
  page: Page;
  imageUrl: string;
  items: Item[];
  selected: string | null;
  onSelect: (detId: string) => void;
}

// Page image with an SVG overlay in page-pixel coordinates, so boxes stay aligned at any zoom.
export function PageViewer({ page, imageUrl, items, selected, onSelect }: Props) {
  const [zoom, setZoom] = useState(0.35);
  const [showTags, setShowTags] = useState(true);
  const w = page.width * zoom;
  const h = page.height * zoom;

  return (
    <div className="viewer">
      <div className="viewer-bar">
        <button onClick={() => setZoom((z) => Math.max(0.15, z / 1.25))}>−</button>
        <span>{Math.round(zoom * 100)}%</span>
        <button onClick={() => setZoom((z) => Math.min(2, z * 1.25))}>+</button>
        <label>
          <input type="checkbox" checked={showTags} onChange={(e) => setShowTags(e.target.checked)} /> tags
        </label>
        <span className="muted">
          {page.has_text_layer ? "vector PDF (text layer)" : `scan, deskewed ${page.skew_deg.toFixed(2)}°`}
        </span>
        <span className="legend">
          {Object.entries(STATUS_COLOUR).map(([s, c]) => (
            <span key={s}><i style={{ background: c }} />{s.replace("_", " ")}</span>
          ))}
        </span>
      </div>
      <div className="viewer-canvas">
        <div style={{ position: "relative", width: w, height: h }}>
          <img src={imageUrl} width={w} height={h} alt="P&ID page" draggable={false} />
          <svg width={w} height={h} viewBox={`0 0 ${page.width} ${page.height}`} style={{ position: "absolute", inset: 0 }}>
            {items.map((it) => {
              const b = it.det.box;
              const sel = it.det.det_id === selected;
              const colour = STATUS_COLOUR[it.status];
              return (
                <g key={it.det.det_id} onClick={() => onSelect(it.det.det_id)} style={{ cursor: "pointer" }}>
                  <rect x={b.x0 - 4} y={b.y0 - 4} width={b.x1 - b.x0 + 8} height={b.y1 - b.y0 + 8}
                    fill={sel ? `${colour}33` : "transparent"} stroke={colour} strokeWidth={sel ? 8 : 4} />
                  {showTags && it.tag && (
                    <text x={b.x0} y={b.y0 - 10} fontSize={26} fill={colour} fontWeight={600}>{it.tag}</text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    </div>
  );
}
