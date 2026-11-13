# Overleaf mechanics

Project `6a1dc8fa08f8c0d184f393aa`, manuscript `manuscript_aim.tex`, doc id
`6aa0dbb8881c2639adec82fa`. Work through the browser pane; if the page shows "Log in" or
"Restricted", ask Juan to log in and wait.

## Edit safely

1. Read the editor text: `document.querySelector('.cm-content').cmView.view.state.doc.toString()`.
2. Check it equals the saved file: `fetch('/project/<pid>/doc/<did>/download')`. Record a
   SHA-256 prefix of the text and check it again right before editing, so nobody else's edit
   is overwritten.
3. Build every change as an exact old string and its replacement; each old string must occur
   exactly once. Apply all changes for one item in a single
   `view.dispatch({changes: [...]})`, sorted by position.
4. Write LaTeX in plain double-quoted JavaScript strings with escaped backslashes, or in
   `String.raw` templates only when the text has no backticks and no `${` (LaTeX quotes ``
   break template literals).

## Compile and check

POST `/project/<pid>/compile?auto_compile=false` with header `X-Csrf-Token` set to
`document.querySelector('meta[name="ol-csrfToken"]').content` and body
`{rootDoc_id: null, draft: false, check: 'silent', incrementalCompilesEnabled: true, stopOnFirstError: false}`.
Read the log from the `output.log` entry: `f.url + '?compileGroup=' + j.compileGroup +
'&clsiserverid=' + j.clsiServerId` (the pdfDownloadDomain URL fails CORS). Report errors,
new overfull boxes and undefined references; the long-standing overfull boxes near the title
block are known.

## Comments

- List threads: GET `/project/<pid>/threads`; anchors: GET `/project/<pid>/ranges`.
- Resolve: POST `/project/<pid>/doc/<did>/thread/<tid>/resolve` (same CSRF header, body `{}`).
- Edit a message: POST `/project/<pid>/thread/<tid>/messages/<mid>/edit`.
- Default: resolve addressed comments with no reply.

## Figures

The Overleaf page cannot read files from the laptop. Build figures locally
(`visualization/make_audit_panels.py`, `paper_figures/build_zooms.js`, `paper_figures/export.ps1`)
and ask Juan to drag the PDF into the `Figures` folder, giving the full path.
