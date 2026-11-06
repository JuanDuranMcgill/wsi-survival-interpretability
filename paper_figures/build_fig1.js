// Fig. 1 — the IER audit at a glance. One editable slide at final print size (7.0 x 7.45 in).
// Build: node build_fig1.js  ->  fig1_ier_overview.pptx ; export with export.ps1
const path = require("path");
const pptxgen = require("pptxgenjs");

const A = (f) => path.join(__dirname, "assets", f);
const NAVY = "1F2A5C", CARD = "E8EDF6", PANEL = "F4F6FA", EDGE = "CDD4E2", GREY = "6B7280", INK = "222222";
const I_COL = "B9770E", E_COL = "6C3483", R_COL = "117A65", WHITE = "FFFFFF";
const FONT = "Calibri";

const pres = new pptxgen();
pres.defineLayout({ name: "FIG1", width: 7.0, height: 7.45 });
pres.layout = "FIG1";
pres.title = "IER audit at a glance";
const s = pres.addSlide();
s.background = { color: WHITE };

// ---------- helpers ----------
function text(t, x, y, w, h, o = {}) {
  s.addText(t, { x, y, w, h, fontFace: FONT, fontSize: 7, color: INK, margin: 0, valign: "middle",
    isTextBox: true, ...o });
}
function box(x, y, w, h, o = {}) {
  s.addShape(o.round === false ? pres.ShapeType.rect : pres.ShapeType.roundRect,
    { x, y, w, h, rectRadius: o.radius ?? 0.06, fill: { color: o.fill ?? CARD },
      line: { color: o.line ?? (o.fill ?? CARD), width: o.lw ?? 0.5, dashType: o.dash ?? "solid" } });
}
function circle(x, y, d, fill, label, o = {}) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: o.line ?? fill, width: 0.5 } });
  if (label !== undefined) text(label, x, y, d, d, { align: "center", color: o.color ?? WHITE, bold: o.bold ?? true,
    fontSize: o.size ?? 7 });
}
function hline(x1, x2, y, color, o = {}) {
  s.addShape(pres.ShapeType.line, { x: x1, y, w: x2 - x1, h: 0,
    line: { color, width: o.lw ?? 1.0, endArrowType: o.arrow ? "triangle" : undefined } });
}
function vline(x, y1, y2, color, o = {}) {
  s.addShape(pres.ShapeType.line, { x, y: y1, w: 0, h: y2 - y1,
    line: { color, width: o.lw ?? 1.0, endArrowType: o.arrow ? "triangle" : undefined } });
}
function seg(x1, y1, x2, y2, color, o = {}) { // free segment, left-to-right or top-to-bottom only
  s.addShape(pres.ShapeType.line, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    flipV: (y2 < y1) !== (x2 < x1), line: { color, width: o.lw ?? 0.6 } });
}
function arrowRight(x1, x2, y) { hline(x1, x2, y, NAVY, { arrow: true, lw: 1.25 }); }

// ---------- A. the audited model ----------
text("The audited model", 0.15, 0.08, 3.0, 0.2, { fontSize: 9.5, bold: true, color: NAVY });

// Row 1: slide -> tiles -> compartments -> vectors
s.addImage({ path: A("he_slide_clean.jpg"), x: 0.15, y: 0.38, w: 1.40, h: 0.945 });
text("Whole-slide image (H&E)", 0.15, 1.36, 1.40, 0.16, { align: "center" });
arrowRight(1.60, 1.86, 0.85);
s.addImage({ path: A("he_zoom_clean.jpg"), x: 1.91, y: 0.38, w: 0.945, h: 0.945 });
for (let k = 1; k < 4; k++) {
  vline(1.91 + k * 0.236, 0.38, 1.325, WHITE, { lw: 0.75 });
  hline(1.91, 2.855, 0.38 + k * 0.236, WHITE, { lw: 0.75 });
}
text("Cut into tiles (256 × 256 px)", 1.80, 1.36, 1.17, 0.16, { align: "center" });
arrowRight(2.90, 3.16, 0.85);
s.addImage({ path: A("compartment_map.jpg"), x: 3.21, y: 0.38, w: 1.21, h: 0.945 });
text("CONCH labels each tile", 3.21, 1.36, 1.21, 0.16, { align: "center" });
box(3.21, 1.53, 1.21, 0.15, { fill: WHITE, line: GREY, dash: "dash", lw: 0.5 });
text("check: how decisively?", 3.21, 1.53, 1.21, 0.15, { align: "center", fontSize: 6.5, color: GREY, italic: true });
arrowRight(4.47, 4.73, 0.85);
const vec = ["9AA5BF", "5B6A8F", "C4CBDA", "2F3B63", "8794B3"];
for (let i = 0; i < 5; i++) {
  for (let j = 0; j < 7; j++) {
    const shade = vec[(i + j) % vec.length];
    box(4.80 + j * 0.075, 0.45 + i * 0.16, 0.07, 0.12, { fill: shade, round: false });
  }
}
text("UNI-2: one\nvector per tile", 4.58, 1.34, 0.98, 0.26, { align: "center" });
text("8 compartments in bladder (BLCA), 9 in breast (BRCA); each tile belongs to one",
  5.70, 0.55, 1.20, 0.62, { fontSize: 6.5, color: GREY, italic: true, valign: "top" });

// elbow to row 2
vline(5.065, 1.61, 1.74, NAVY, { lw: 1.25 });
hline(0.62, 5.065, 1.74, NAVY, { lw: 1.25 });
vline(0.62, 1.74, 1.97, NAVY, { lw: 1.25, arrow: true });

// Row 2: encoders -> attention -> weights -> sum -> risk
text("One encoder per compartment", 0.75, 1.80, 1.80, 0.15, { bold: true, color: NAVY });
const comps = ["Tumour tiles", "Necrosis tiles", "Adipose tiles", "… every compartment"];
const hs = [["h", "1"], ["h", "2"], ["h", "3"], ["h", "R"]];
const ws = [["w", "1"], ["w", "2"], ["w", "3"], ["w", "R"]];
const rowY = [2.00, 2.22, 2.44, 2.66];
rowY.forEach((y, i) => {
  box(0.15, y, 0.98, 0.18, { fill: CARD });
  text(comps[i], 0.15, y, 0.98, 0.18, { align: "center" });
  hline(1.14, 1.31, y + 0.09, NAVY, { arrow: true, lw: 0.75 });
  box(1.33, y, 0.55, 0.18, { fill: WHITE, line: NAVY, lw: 0.6 });
  text("encoder", 1.33, y, 0.55, 0.18, { align: "center", fontSize: 6.5 });
  hline(1.89, 2.05, y + 0.09, NAVY, { arrow: true, lw: 0.75 });
  s.addText([{ text: hs[i][0], options: {} }, { text: hs[i][1], options: { subscript: true } }],
    { x: 2.07, y, w: 0.22, h: 0.18, fontFace: FONT, fontSize: 7.5, italic: true, color: INK, margin: 0, valign: "middle",
      isTextBox: true });
  seg(2.30, y + 0.09, 2.55, 2.42, GREY);
});
box(2.55, 2.07, 0.86, 0.70, { fill: NAVY });
text("Cross-region\nattention", 2.55, 2.07, 0.86, 0.70, { align: "center", color: WHITE, bold: true, fontSize: 7.5 });
text("One learned weight\nper compartment", 3.35, 1.78, 1.10, 0.24, { align: "center", bold: true, color: I_COL, fontSize: 7 });
rowY.forEach((y, i) => {
  seg(3.41, 2.42, 3.72, y + 0.11, GREY);
  s.addShape(pres.ShapeType.ellipse, { x: 3.72, y: y + 0.01, w: 0.21, h: 0.21, fill: { color: I_COL }, line: { color: I_COL, width: 0.5 } });
  s.addText([{ text: ws[i][0], options: {} }, { text: ws[i][1], options: { subscript: true } }],
    { x: 3.72, y: y + 0.01, w: 0.21, h: 0.21, fontFace: FONT, fontSize: 6.5, bold: true, color: WHITE, align: "center",
      margin: 0, valign: "middle", isTextBox: true });
  seg(3.93, y + 0.11, 4.30, 2.42, GREY);
});
circle(4.30, 2.21, 0.42, NAVY, "Σ", { size: 11 });
arrowRight(4.73, 4.93, 2.42);
box(4.95, 2.21, 0.92, 0.42, { fill: CARD, line: NAVY, lw: 0.75 });
text("Risk score", 4.95, 2.21, 0.92, 0.42, { align: "center", bold: true, fontSize: 8, color: NAVY });
box(5.98, 2.21, 0.92, 0.42, { fill: WHITE, line: GREY, lw: 0.75 });
text("Observed outcome\n(progression-free\ninterval)", 5.98, 2.21, 0.92, 0.42, { align: "center", fontSize: 6.5 });
box(4.95, 1.80, 1.95, 0.30, { fill: WHITE, line: GREY, dash: "dash", lw: 0.5 });
text("check first: performance against clinical-only Cox, ABMIL and Kaplan–Meier",
  4.99, 1.80, 1.88, 0.30, { align: "center", fontSize: 6.5, color: GREY, italic: true });

// connectors to the three legs (each ends with an arrow into its panel)
const PY = 3.12; // panel top
vline(3.825, 2.88, 2.93, I_COL, { lw: 1.5 });
hline(1.20, 3.825, 2.93, I_COL, { lw: 1.5 });
vline(1.20, 2.93, PY, I_COL, { lw: 1.5, arrow: true });
vline(5.41, 2.63, 3.01, E_COL, { lw: 1.5 });
hline(3.50, 5.41, 3.01, E_COL, { lw: 1.5 });
vline(3.50, 3.01, PY, E_COL, { lw: 1.5, arrow: true });
vline(6.44, 2.63, 2.97, R_COL, { lw: 1.5 });
hline(5.80, 6.44, 2.97, R_COL, { lw: 1.5 });
vline(5.80, 2.97, PY, R_COL, { lw: 1.5, arrow: true });
text("weights", 2.40, 2.80, 0.6, 0.12, { fontSize: 6.5, color: I_COL, italic: true });
text("risk score", 4.55, 2.89, 0.8, 0.12, { fontSize: 6.5, color: E_COL, italic: true, align: "right" });
text("risk vs outcome", 5.40, 2.80, 0.98, 0.12, { fontSize: 6.5, color: R_COL, italic: true, align: "right" });

// ---------- B. the three legs ----------
function leg(x0, col, letter, title, sub, claim, steps, passes) {
  const W = 2.20, H = 3.76;
  box(x0, PY, W, H, { fill: PANEL, line: EDGE, lw: 0.75, radius: 0.08 });
  circle(x0 + 0.10, PY + 0.10, 0.40, col, letter, { size: 13 });
  text(title, x0 + 0.58, PY + 0.09, W - 0.65, 0.22, { fontSize: 10, bold: true, color: NAVY });
  text(sub, x0 + 0.58, PY + 0.31, W - 0.65, 0.18, { fontSize: 7, italic: true, color: GREY });
  text("WHAT IT SEEMS TO SAY", x0 + 0.12, PY + 0.60, W - 0.2, 0.13, { fontSize: 6.5, bold: true, color: col });
  text(claim, x0 + 0.12, PY + 0.74, W - 0.24, 0.30, { fontSize: 9, italic: true, bold: true, color: NAVY });
  text("HOW THE AUDIT CHECKS IT", x0 + 0.12, PY + 1.10, W - 0.2, 0.13, { fontSize: 6.5, bold: true, color: col });
  steps.forEach((st, i) => {
    const y = PY + 1.28 + i * 0.52;
    circle(x0 + 0.12, y + 0.02, 0.17, col, String(i + 1), { size: 6.5 });
    text(st.t, x0 + 0.34, y - 0.02, 1.30, 0.50, { fontSize: 7, valign: "top" });
    st.g(x0 + 1.68, y);
  });
  box(x0 + 0.10, PY + 2.90, W - 0.20, 0.76, { fill: WHITE, line: col, lw: 1.0 });
  text("PASSES IF", x0 + 0.18, PY + 2.95, W - 0.3, 0.13, { fontSize: 6.5, bold: true, color: col });
  text(passes, x0 + 0.18, PY + 3.09, W - 0.34, 0.54, { fontSize: 7.5, valign: "top" });
}
// tiny glyphs (each fits in about 0.45 x 0.45 in)
const g = {
  coalitions: (x, y) => { for (let r = 0; r < 4; r++) for (let c = 0; c < 6; c++) {
    const on = ((r * 7 + c * 3) % 5) > 1; box(x + c * 0.075, y + r * 0.10, 0.06, 0.07, { fill: on ? I_COL : "D9DCE3", round: false }); } },
  twoPayoffs: (x, y) => { box(x, y, 0.46, 0.17, { fill: WHITE, line: NAVY, lw: 0.5 }); text("risk", x, y, 0.46, 0.17, { align: "center", fontSize: 6.5 });
    box(x, y + 0.23, 0.46, 0.17, { fill: WHITE, line: NAVY, lw: 0.5 }); text("C-index", x, y + 0.23, 0.46, 0.17, { align: "center", fontSize: 6.5 }); },
  compare: (x, y) => { [0.30, 0.22, 0.16, 0.10].forEach((h, i) => box(x + i * 0.06, y + 0.40 - h, 0.045, h, { fill: I_COL, round: false }));
    [0.12, 0.30, 0.16, 0.22].forEach((h, i) => box(x + 0.27 + i * 0.06, y + 0.40 - h, 0.045, h, { fill: NAVY, round: false })); },
  forest: (x, y) => { circle(x + 0.17, y, 0.11, E_COL); circle(x + 0.04, y + 0.25, 0.11, E_COL); circle(x + 0.30, y + 0.25, 0.11, E_COL);
    seg(x + 0.20, y + 0.11, x + 0.10, y + 0.25, E_COL); seg(x + 0.22, y + 0.11, x + 0.34, y + 0.25, E_COL); },
  ranked: (x, y) => { [0.44, 0.34, 0.24, 0.14].forEach((w, i) => box(x, y + i * 0.10, w, 0.065, { fill: E_COL, round: false })); },
  fdr: (x, y) => { box(x, y + 0.08, 0.46, 0.22, { fill: WHITE, line: E_COL, lw: 0.5 }); text("q < 0.05?", x, y + 0.08, 0.46, 0.22, { align: "center", fontSize: 6.5 }); },
  errors: (x, y) => { for (let k = 0; k < 6; k++) circle(x + (k % 3) * 0.15, y + Math.floor(k / 3) * 0.16, 0.10, (k === 1 || k === 5) ? R_COL : "B8BFCC"); },
  rule: (x, y) => { box(x, y, 0.46, 0.18, { fill: R_COL }); text("stage = I ?", x, y, 0.46, 0.18, { align: "center", fontSize: 6.5, color: WHITE, bold: true });
    seg(x + 0.12, y + 0.18, x + 0.04, y + 0.36, R_COL); seg(x + 0.34, y + 0.18, x + 0.42, y + 0.36, R_COL); },
  ci: (x, y) => { vline(x + 0.25, y, y + 0.42, NAVY, { lw: 0.6 });
    [[0.05, 0.30], [0.14, 0.40], [0.22, 0.45]].forEach(([a, b], i) => { hline(x + a, x + b, y + 0.07 + i * 0.13, R_COL, { lw: 1.0 }); }); },
};
leg(0.10, I_COL, "I", "Interpretability", "Read the model's own weights",
  "“This tissue drives the risk.”",
  [{ t: "Hide compartments in every combination (2⁸ or 2⁹ sets).", g: g.coalitions },
   { t: "Score each set twice: each patient's risk score, and the C-index against outcome.", g: g.twoPayoffs },
   { t: "Compare both with the weights; repeat on unseen hospitals and on a planted signal.", g: g.compare }],
  "Weights order the tissues as the model uses them, and that use orders them as outcome does.");
leg(2.40, E_COL, "E", "Explainability", "Explain the score from outside",
  "“These features explain the risk.”",
  [{ t: "A random forest learns the risk score from clinical and pathomic features.", g: g.forest },
   { t: "Rank the features by importance.", g: g.ranked },
   { t: "Re-rank with a second estimator; test each feature against observed outcome.", g: g.fdr }],
  "The ranking survives a change of estimator, and features survive multiple-testing control against outcome.");
leg(4.70, R_COL, "R", "Reliability", "Find where the model errs",
  "“The model is more or less reliable for these patients.”",
  [{ t: "Error per patient over comparable pairs, aware of censoring.", g: g.errors },
   { t: "MED3pa learns rules that flag groups, on discovery patients.", g: g.rule },
   { t: "On held-out patients: C-index inside each group; 90% bounds per subgroup.", g: g.ci }],
  "A flagged group's C-index interval excludes the all-patient value (and bounds hold where promised).");

// ---------- tie line ----------
box(0.10, 6.98, 6.80, 0.38, { fill: NAVY, radius: 0.08 });
text("Each explanation is checked against the model and against what happened to patients.",
  0.10, 6.98, 6.80, 0.38, { align: "center", color: WHITE, bold: true, fontSize: 9 });

pres.writeFile({ fileName: path.join(__dirname, "fig1_ier_overview.pptx") }).then((f) => console.log("wrote", f));
