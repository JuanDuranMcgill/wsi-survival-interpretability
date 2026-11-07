// Figs. 2-4 — one zoom per leg: (a) how the check works, (b) what it found (data panel from Python).
// Build: node build_zooms.js  ->  fig2_interp_zoom.pptx, fig3_explain_zoom.pptx, fig4_reliab_zoom.pptx
const path = require("path");
const pptxgen = require("pptxgenjs");

const NAVY = "1F2A5C", CARD = "E8EDF6", PANEL = "F4F6FA", EDGE = "CDD4E2", GREY = "6B7280", INK = "222222";
const I_COL = "B9770E", E_COL = "6C3483", R_COL = "117A65", WHITE = "FFFFFF", LG = "D9DCE3";
const FONT = "Calibri";
const W = 7.0, AW = 3.45; // slide width; panel (a) width

function make(name, height, col, letter, title, steps, note, panelB, bAspect) {
  const pres = new pptxgen();
  pres.defineLayout({ name: "Z", width: W, height });
  pres.layout = "Z";
  const s = pres.addSlide();
  s.background = { color: WHITE };
  const text = (t, x, y, w, h, o = {}) => s.addText(t, { x, y, w, h, fontFace: FONT, fontSize: 7.6, color: INK, margin: 0,
    valign: "middle", isTextBox: true, ...o });
  const box = (x, y, w, h, o = {}) => s.addShape(o.round === false ? pres.ShapeType.rect : pres.ShapeType.roundRect,
    { x, y, w, h, rectRadius: o.radius ?? 0.05, fill: { color: o.fill ?? CARD },
      line: { color: o.line ?? (o.fill ?? CARD), width: o.lw ?? 0.5, dashType: o.dash ?? "solid" } });
  const circle = (x, y, d, fill, label, o = {}) => {
    s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0.5 } });
    if (label !== undefined) text(label, x, y, d, d, { align: "center", color: o.color ?? WHITE, bold: true, fontSize: o.size ?? 7.6 });
  };
  const down = (x, y1, y2, c) => s.addShape(pres.ShapeType.line, { x, y: y1, w: 0, h: y2 - y1,
    line: { color: c, width: 1.25, endArrowType: "triangle" } });
  const ctx = { s, pres, text, box, circle, down };

  // panel (a)
  box(0.08, 0.08, AW - 0.06, height - 0.16, { fill: PANEL, line: EDGE, lw: 0.75, radius: 0.06 });
  circle(0.18, 0.16, 0.36, col, letter, { size: 12 });
  text("(a) " + title, 0.62, 0.16, AW - 0.62, 0.36, { fontSize: 10, bold: true, color: NAVY });
  let y = 0.66;
  steps.forEach((st, i) => {
    circle(0.20, y + 0.02, 0.20, col, String(i + 1), { size: 7.6 });
    text(st.t, 0.48, y - 0.02, 1.75, st.h, { fontSize: 8, valign: "top" });
    st.g(ctx, 2.32, y, st.h);
    if (i < steps.length - 1) down(0.30, y + 0.26, y + st.h + 0.02, col);
    y += st.h + 0.10;
  });
  if (note) {
    box(0.16, y, AW - 0.22, note.h, { fill: WHITE, line: col, lw: 0.9 });
    text(note.label, 0.24, y + 0.04, AW - 0.4, 0.13, { fontSize: 7.2, bold: true, color: col });
    text(note.t, 0.24, y + 0.18, AW - 0.38, note.h - 0.22, { fontSize: 8, valign: "top" });
  }
  // panel (b)
  text("(b) What the audit found", AW + 0.12, 0.08, 3.3, 0.2, { fontSize: 10, bold: true, color: NAVY });
  const bw = W - AW - 0.17, bh = bw / bAspect;
  s.addImage({ path: path.join(__dirname, "panels", panelB), x: AW + 0.12, y: 0.32, w: bw, h: Math.min(bh, height - 0.36) });
  return pres.writeFile({ fileName: path.join(__dirname, name) }).then((f) => console.log("wrote", f));
}

// ----- glyphs -----
const G = {
  weights: (c, x, y) => { ["Tumour", "Necrosis", "Adipose", "…"].forEach((n, i) => {
      c.box(x, y + i * 0.09, 0.62 + (i % 2 ? 0.03 : 0), 0.06, { fill: I_COL, round: false });
      c.text(n, x + 0.70, y + i * 0.09 - 0.02, 0.45, 0.1, { fontSize: 7.2, color: GREY }); }); },
  coalitions: (c, x, y) => { for (let r = 0; r < 4; r++) for (let k = 0; k < 8; k++) {
      const on = ((r * 5 + k * 3) % 4) > 0; c.box(x + k * 0.085, y + r * 0.10, 0.07, 0.075, { fill: on ? I_COL : LG, round: false }); } },
  payoffs: (c, x, y) => { c.box(x, y, 0.95, 0.22, { fill: WHITE, line: NAVY, lw: 0.6 });
      c.text("risk score", x, y, 0.95, 0.22, { align: "center", fontSize: 7.2, bold: true });
      c.box(x, y + 0.30, 0.95, 0.22, { fill: WHITE, line: NAVY, lw: 0.6 });
      c.text("C-index vs outcome", x, y + 0.30, 0.95, 0.22, { align: "center", fontSize: 7.2, bold: true }); },
  compare: (c, x, y) => { c.text("weights ↔ use", x, y, 1.0, 0.16, { fontSize: 7.2, bold: true, color: I_COL });
      c.text("faithful?", x, y + 0.15, 1.0, 0.14, { fontSize: 7.2, italic: true, color: GREY });
      c.text("use ↔ outcome", x, y + 0.33, 1.0, 0.16, { fontSize: 7.2, bold: true, color: NAVY });
      c.text("prognostic?", x, y + 0.48, 1.0, 0.14, { fontSize: 7.2, italic: true, color: GREY }); },
  score: (c, x, y) => { for (let k = 0; k < 8; k++) c.circle(x + (k % 4) * 0.2, y + Math.floor(k / 4) * 0.2, 0.13, k % 3 ? "9AA5BF" : NAVY); },
  forest: (c, x, y) => { c.circle(x + 0.30, y, 0.14, E_COL); c.circle(x + 0.12, y + 0.30, 0.14, E_COL); c.circle(x + 0.48, y + 0.30, 0.14, E_COL);
      c.s.addShape(c.pres.ShapeType.line, { x: x + 0.19, y: y + 0.13, w: 0.14, h: 0.17, flipH: true, line: { color: E_COL, width: 0.6 } });
      c.s.addShape(c.pres.ShapeType.line, { x: x + 0.39, y: y + 0.13, w: 0.14, h: 0.17, line: { color: E_COL, width: 0.6 } }); },
  ranked: (c, x, y) => { [0.85, 0.62, 0.45, 0.30].forEach((w, i) => c.box(x, y + i * 0.11, w, 0.075, { fill: i ? "A88BB8" : E_COL, round: false })); },
  twoChecks: (c, x, y) => { c.box(x - 0.08, y, 1.12, 0.22, { fill: WHITE, line: E_COL, lw: 0.6 });
      c.text("permutation importance", x - 0.08, y, 1.12, 0.22, { align: "center", fontSize: 7.2 });
      c.box(x - 0.08, y + 0.30, 1.12, 0.22, { fill: WHITE, line: E_COL, lw: 0.6 });
      c.text("Cox vs PFI, FDR", x - 0.08, y + 0.30, 1.12, 0.22, { align: "center", fontSize: 7.2 }); },
  pairs: (c, x, y) => { for (let k = 0; k < 10; k++) c.circle(x + (k % 5) * 0.18, y + Math.floor(k / 5) * 0.2, 0.12, (k === 2 || k === 7) ? R_COL : "B8BFCC"); },
  label: (c, x, y) => { for (let k = 0; k < 10; k++) c.box(x + k * 0.09, y + 0.05, 0.07, 0.16, { fill: k > 4 ? R_COL : "B8BFCC", round: false });
      c.text("median cut", x + 0.2, y + 0.24, 0.6, 0.12, { fontSize: 7.2, color: GREY, align: "center" }); },
  rule: (c, x, y) => { c.box(x, y, 0.70, 0.2, { fill: R_COL }); c.text("rule on a feature", x, y, 0.70, 0.2, { align: "center", fontSize: 7.2, color: WHITE, bold: true });
      c.text("60% discovery", x, y + 0.26, 0.9, 0.12, { fontSize: 7.2, color: GREY }); },
  heldout: (c, x, y) => { c.text("40% held out:", x, y, 1.0, 0.13, { fontSize: 7.2, bold: true, color: R_COL });
      ["AUROC vs 0.5", "C-index in group", "90% bound coverage"].forEach((t, i) => c.text("• " + t, x, y + 0.15 + i * 0.14, 1.05, 0.13, { fontSize: 7.2 })); },
};

const jobs = [
  make("fig2_interp_zoom.pptx", 4.60, I_COL, "I", "How the weight check works", [
    { t: "The model reports one learned weight per compartment.", h: 0.42, g: G.weights },
    { t: "Hide compartments in every combination: 2⁸ = 256 sets in BLCA, 2⁹ = 512 in BRCA, each scored exactly.", h: 0.55, g: G.coalitions },
    { t: "Score every set twice: by each patient's risk score (what the model uses, no outcome) and by the out-of-bag C-index (what carries outcome signal). Shapley values share the credit.", h: 0.86, g: G.payoffs },
    { t: "Compare the weights with what the model uses, and what the model uses with outcome.", h: 0.62, g: G.compare },
  ], { label: "STRESS TESTS", h: 0.78, t: "Repeat on hospitals the model never saw and with masked removal; plant a known signal in one compartment and ask which measure ranks it first." },
  "fig2b.png", 3.3 / 4.25),
  make("fig3_explain_zoom.pptx", 4.25, E_COL, "E", "How the surrogate check works", [
    { t: "Start from each patient's out-of-bag risk score.", h: 0.40, g: G.score },
    { t: "A random forest learns that score from clinical (21 BLCA, 25 BRCA) and first-order pathomic features (504, 514).", h: 0.66, g: G.forest },
    { t: "Rank features and compartments by importance (impurity).", h: 0.48, g: G.ranked },
    { t: "Two checks: does the order survive a second estimator, and does each feature predict observed PFI after false-discovery-rate control?", h: 0.76, g: G.twoChecks },
  ], { label: "WHY IT MATTERS", h: 0.62, t: "The surrogate explains the model's score. Only features that pass the outcome test say something about the disease." },
  "fig3b.png", 3.3 / 3.95),
  make("fig4_reliab_zoom.pptx", 4.70, R_COL, "R", "How the reliability check works", [
    { t: "Error per patient: the share of that patient's comparable pairs the model orders wrongly (censoring-aware).", h: 0.62, g: G.pairs },
    { t: "Label high error at a threshold (median, 70th, 80th or 90th percentile).", h: 0.48, g: G.label },
    { t: "MED3pa learns rules on discovery patients from clinical or pathomic features; each rule defines a profile.", h: 0.62, g: G.rule },
    { t: "On held-out patients: is error predictable, does the model's C-index differ inside each profile, and do per-patient bounds hold by subgroup at three time horizons?", h: 0.78, g: G.heldout },
  ], { label: "TWO DIFFERENT CLAIMS", h: 0.62, t: "“Error is predictable here” is not “the model performs differently here”; the audit tests each separately." },
  "fig4b.png", 3.3 / 4.35),
];
Promise.all(jobs);
