# SimulacraBench submission: stack_trees_v6

Code for the SimulacraBench competition on Codabench: probabilistic completion
of survey responses for three instruments (UNICEF, World Bank, UNHCR). For each
held-out respondent only the GIVEN block is visible, and every blank PREDICT
cell needs a probability vector over that item's options, with a final slot
for NA_GATED on gated items. Scoring is the log score, rescaled so that a
uniform guess is 0 and a perfect one is 1, averaged over the instruments.

## Approach

`main.py` fits one small count model per item inside `predict()`, on the
visible rows of the frame:

- Strata are nested subsets of the GIVEN columns. Level d conditions on the
  first d features of the item's feature list and is smoothed toward level
  d - 1 with a Dirichlet prior of strength a_d. Level 0 is the item marginal
  with half a count per option, so no option ever gets zero probability.
- The feature list (up to four columns) and each a_d are chosen greedily per
  item by the exact leave-one-out log score on the visible rows, including the
  1e-3 mixing the grader applies.
- NA_GATED is treated as an ordinary class. Skip routing, rows where the
  routing was not followed, and coding differences between survey rounds are
  learned from the data instead of being hard-coded from the gate rules.
- A multinomial logistic regression (L2, C = 1) uses the additive one-hot GIVEN
  block and joint-category columns for country with every other GIVEN item.
  UNHCR also includes wave with every other GIVEN item. Pairs come from the
  declared schema supports, with duplicate pairs removed, at most 256 columns
  per pair and 1,024 interaction columns in total. The design uses float64 CSR
  storage, and regression fits use one CPU thread.
- Both members use the same three respondent folds. Backoff feature selection
  and smoothing are refitted on each training complement. An 11-point grid
  selects the mixture weight by held-out log score, including the scorer's
  1e-3 mixing. Returned vectors receive that mixing in the grader. The tree
  member uses the same folds and scorer objective, with its own simplex search.
- Complete count-model predictions are computed first. Optional item fitting
  runs serially in separate Linux processes. The matched-fold LR stage has a
  20 s item limit and a deadline capped at 180 s after entry into `predict()`.
  It caches only complete OOF true-class probabilities for both backoff and LR.
  The categorical HistGradientBoosting stage follows through the schema-based
  instrument allocation after entry, also with a 20 s item limit, for items with at least 600 known labels, two
  observed classes and a complete LR OOF cache. Its three fold models use only
  each fold's complement. Declared GIVEN category order is retained. Unknown
  and missing values become missing categories. Each tree distribution is mixed
  with its fold-local Jeffreys marginal, then calibrated and blended on the
  constrained simplex using the grader's epsilon-mixed log score. A successful
  tree replaces an item only after full refits complete. Late or failed work
  preserves the most recently committed blend. Each child is reaped before
  the next worker starts, and process creation failure falls back safely.

Nothing is trained offline and no weights are bundled.

Local qualification completed on 2026-10-12. The optional fitting deadlines are
measured from `predict()` entry and use schema-declared GIVEN keys: UNICEF
(`Country`) receives 90 s, World Bank (`group`) 180 s, and UNHCR (`wave`) 450 s.
Unknown schemas receive 240 s. The LR deadline is the earlier of 180 s and the
instrument allocation. Tree work uses the full allocation. The 20 s per-item
limit, model design, folds, eligibility, queue and fallback remain unchanged.

The official ZIP checker and text check passed. All three full-size seed-0
phase-1 synthetic scorers passed sequentially on one CPU core. Whole-scorer
wall time, eligible/completed tree items, positive-weight commits, and peak RSS
were:

| Instrument | Wall time | Eligible / completed | Positive-weight commits | Peak RSS |
| --- | ---: | ---: | ---: | ---: |
| UNICEF | 33.86 s | 12 / 12 | 12 | 230.70 MiB |
| World Bank | 133.57 s | 76 / 76 | 73 | 234.11 MiB |
| UNHCR | 315.87 s | 168 / 164 | 162 | 443.62 MiB |

Total scorer wall time was 483.30 s. Maximum RSS was 0.433 GiB. A frozen
late-queue control used 3,072 training rows and a separate 1,024-row outer
holdout with all 64 six-binary input combinations balanced and a constant wave.
The control executed both versions of `predict()`. An earlier gated tree
worker advanced the virtual clock by 250 s, so v5's 240 s schedule skipped
the later target and v6's 450 s schedule completed it. Admission and worker
deadlines used virtual time. OOF selection and full fits used real models
and training rows. The selected tree weight was 0.50 and outer epsilon-mixed
log loss improved by 0.1418954348493729 nats, exceeding the frozen 0.01
threshold. Whole-scorer timings used real clocks.

The prior v5 submission had a displayed score of 0.63. That single score does
not establish a causal model gain. The v6 synthetic scores (UNICEF 0.50, World
Bank 0.44, UNHCR 0.55) qualify behavior and runtime only. They do not estimate
utility on unreleased survey data. This candidate passed local qualification
and has not been submitted.

## Reproduce

```bash
git clone https://github.com/SituatedEvals/public
cd public
pip install -r requirements.txt
python make_sandbox.py --schema data/unhcr.json --out _sandbox/unhcr
python score.py --submission /path/to/this/repo --data _sandbox/unhcr \
    --schema _sandbox/unhcr/schema.json --phase 1
cd /path/to/this/repo && zip -r ../submission.zip main.py requirements.txt
```

The sandbox data is synthetic, so its scores check the output contract and
the run time, not the real-data gain.
