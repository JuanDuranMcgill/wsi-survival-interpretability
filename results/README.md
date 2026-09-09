# results/

The results contract: **every number that appears in the paper is written here
as a machine-readable file, committed alongside the code that produced it.** No
number gets typed into the LaTeX by hand.

Each file must record:
- the input run directory it read from
- the round count / N it used
- any split seed, threshold, or epoch-selection choice
- the endpoint (PFI)

See `docs/RUNBOOK.md` for what produces what, and `docs/RERUN_PLAN.md` for why.
