# Reviewer Decision: Local Excess-Confidence Directional Refinement

## Review scope

This is a method-design and implementation review, not a claim that the new
method has already produced publishable empirical gains. The reviewed artifact
is `REFINE_LOCAL_EXCESS_CONFIDENCE.md` together with the implementation, tests,
configuration, and one-command A/B runner.

## Field analysis and panel

- Primary field: uncertainty-aware 3D medical image segmentation.
- Secondary fields: calibration, spatial post-processing, domain
  generalization.
- Paradigm: quantitative machine-learning experiment.
- Maturity: implemented experiment protocol awaiting full-data execution.
- Target standard: a Q2 medical-imaging journal; a Q1 claim would require
  stronger external comparison and statistical reporting.

The panel perspectives were: an editor focused on contribution and scope; a
calibration/evaluation methodologist; a brain-tumor segmentation specialist; a
clinical deployment reviewer focused on harmful corrections; and a devil's
advocate testing the strongest rival explanation.

## Round 1 findings and author revision

1. **Methodology — major, resolved in code**: the first runner design selected
   a direction on source validation but evaluated only the winner externally.
   That could not establish that the new direction outperformed the previous
   best refine on the same checkpoint and centers. The revised runner now
   evaluates the best source-validation candidate from both fixed directions on
   all configured centers, writes `final_comparison.csv`, and never feeds those
   external results back into selection.
2. **Originality — minor, resolved in positioning**: neighbor-aware calibration
   and uncertainty-guided propagation are prior art. The method statement now
   claims only the parameter-free continuous direction interpolation based on
   local excess confidence. It does not claim local neighborhoods, uncertainty
   weighting, or iterative refinement as new.
3. **Reproducibility — minor, resolved in code**: the new mode is explicit in
   configuration and checkpoint metadata; `legacy_complement` remains available
   as a same-code-path ablation. The new mode adds no parameter or buffer, so old
   checkpoints retain strict state-dict compatibility.
4. **Safety — minor, resolved in protocol**: selection now uses ordinary ECE
   only after a `0.001` absolute Dice guardrail. Equal ECE retains the earlier
   legacy candidate. Three iterations and the established refine strength remain
   fixed.

## Round 2 panel decision

| Reviewer | Decision | Main reason |
|---|---|---|
| Editor | Minor Revision | Narrow but clearly stated incremental contribution; suitable if gains are consistent. |
| Methodology | Minor Revision | Leakage-resistant A/B protocol and strict Dice guardrail are sound; statistical evidence remains to be generated. |
| Domain | Minor Revision | The mechanism addresses the observed over-correction failure without replacing the geometric uncertainty core. |
| Clinical/practical | Minor Revision | Homogeneous supported predictions are protected continuously, while isolated high-confidence errors remain correctable. |
| Devil's advocate | No critical design flaw | The strongest counterargument remains that any gain may come from weaker effective correction rather than the proposed mechanism; the fixed-strength legacy endpoint ablation directly tests this. |

### Editorial decision: Minor Revision — approved for full experiment

The design passes conceptual feasibility and adequate incremental originality.
It is appropriate to run and report as a small refinement contribution. It is
not yet approved as a formal empirical result because the full-data A/B output
has not been supplied in this review round.

## Scores for the reviewed design

| Dimension | Score | Assessment |
|---|---:|---|
| Originality | 3/5 | Adequate, combination-level and explicitly incremental. |
| Methodological rigor | 4/5 | Same checkpoint, fixed endpoints, validation-only selection, held-out reporting, Dice guardrail. |
| Reproducibility | 4/5 | One-command run, resumable outputs, explicit metadata, strict checkpoint compatibility. |
| Feasibility | 4/5 | Small inference-only change with unit and regression coverage. |
| Evidence sufficiency | 2/5 | Implementation evidence exists; full BraTS empirical and uncertainty-interval evidence is pending. |

## Required evidence before manuscript submission

- Run `bash ours/run_refine_tuning.sh` on the fixed checkpoint and retain
  `selection.json`, `candidates.csv`, and `final_comparison.csv`.
- If source validation selects `legacy_complement`, report the new proposal as a
  negative ablation; do not claim improvement from held-out results alone.
- Report paired case-level bootstrap confidence intervals for ECE and Dice,
  ordinary ECE sensitivity to bin count, Brier score, per-class Dice/recall, and
  worst-decile high-confidence errors.
- Accept the new direction only if it passes the source Dice guardrail, improves
  held-out mean ECE over the legacy direction, and removes or materially reduces
  the prior TMC ECE regression without a clinically meaningful Dice loss.
- State explicitly that spatially coherent, low-uncertainty shared errors are
  outside the method's correction mechanism; do not imply that refine alone
  solves uncertainty-map error coverage.

## Prior-art boundary

The originality judgment was calibrated against established spatial label
smoothing, neighbor-aware calibration, uncertainty-guided graph refinement, and
post-hoc medical segmentation calibration:

- <https://arxiv.org/abs/2104.05788>
- <https://arxiv.org/abs/2401.14487>
- <https://proceedings.mlr.press/v121/soberanis-mukul20a.html>
- <https://arxiv.org/abs/2010.14290>

No exact prior formulation of the normalized local excess-confidence direction
coefficient was identified in the targeted search. This supports an incremental
method claim, not a claim of exhaustive novelty proof.
