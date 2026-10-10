# SimulacraBench submission

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
  1e-3 mixing. Returned vectors receive that mixing in the grader. The selected
  blend score requires an independent evaluation before it can estimate gain.
- Complete count-model predictions are computed first. Optional item fitting
  runs serially in separate Linux processes, with a 20 s item limit and a
  deadline 180 s after entry into `predict()`. Late or failed results keep the
  complete count-model prediction. Each child is joined or terminated before
  the call returns. If process creation is unavailable, the count model handles
  the item.

Nothing is trained offline and no weights are bundled.

The packaged candidate passed all three full-size, seed-0 synthetic phase-1
checks on Python 3.13.15, with each scorer process restricted to one CPU core.
Total scorer wall time was 257.9788586390205 s: UNICEF 18.960066691972315 s,
World Bank 45.34330749907531 s and UNHCR 193.67548444797285 s. Qualification
also covered the interaction structure, fold isolation, missing classes,
output alignment, and process timeout and failure cleanup. These measurements
qualify behavior and runtime. The candidate still needs a scored competition
submission to measure its performance on the unreleased surveys.

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
