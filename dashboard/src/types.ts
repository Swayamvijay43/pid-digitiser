// Mirrors the Pydantic contracts in pid/schemas.py.

export type SymbolClass =
  | "gate_valve" | "globe_valve" | "check_valve" | "control_valve"
  | "centrifugal_pump" | "heat_exchanger" | "vessel"
  | "instrument_field" | "instrument_panel";

export interface Box { x0: number; y0: number; x1: number; y1: number }

export interface Detection {
  det_id: string;
  page_index: number;
  cls: SymbolClass;
  box: Box;
  det_conf: number;
}

export type Rung = "text_layer" | "ocr" | "vlm_small" | "vlm_large" | "human";

export interface Attempt { rung: Rung; raw: string | null; tag: string | null; valid: boolean; score: number | null }

export type Status = "auto_accepted" | "needs_review" | "human_corrected" | "human_confirmed" | "rejected";

export interface Item {
  det: Detection;
  tag: string | null;
  status: Status;
  resolved_by: Rung | null;
  confidence: number;
  reason: string;
  attempts: Attempt[];
  line_number: string | null;
  crop_path: string | null;
}

export interface Page { page_index: number; width: number; height: number; has_text_layer: boolean; skew_deg: number }

export interface Exports { mto_csv: string; json: string; symbols: number; exported: number; rejected: number }

export interface Run {
  run_id: string;
  filename: string;
  created: number;
  status: "processing" | "awaiting_review" | "done" | "failed";
  error?: string;
  pages?: Page[];
  items?: Item[];
  pending?: Item[];
  exports?: Exports | null;
}

export interface Decision { det_id: string; action: "confirm" | "correct" | "reject"; tag?: string | null }

// Same tag grammar as the backend: the dashboard warns before a reviewer submits a tag that would fail it.
export const TAG_PATTERNS: Record<SymbolClass, RegExp> = {
  gate_valve: /^HV-\d{4}$/,
  globe_valve: /^HV-\d{4}$/,
  check_valve: /^NRV-\d{3}$/,
  control_valve: /^[FPLT]V-\d{3}$/,
  centrifugal_pump: /^P-\d{3}[AB]?$/,
  heat_exchanger: /^E-\d{3}$/,
  vessel: /^V-\d{3}$/,
  // ISA 5.1: measured variable + readout/output function
  instrument_field: /^[FPLT](T|I|G|E|IC)-\d{3}$/,
  instrument_panel: /^[FPLT](T|I|G|E|IC)-\d{3}$/,
};
