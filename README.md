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
- A multinomial logistic regression (L2, C = 1) on a sparse one-hot GIVEN block is
  mixed into each item's count model. The weight comes from an 11-point grid
  and maximises the log score of the mixture, using the count model's
  leave-one-out probabilities and 3-fold out-of-fold probabilities of the
  regression. The design matrix uses float64 CSR storage, and regression fits
  use one CPU thread. A soft deadline 120 s after entry into `predict()` is
  checked before each fold and final refit. If the deadline passes, that item
  keeps its count-model prediction. An individual fit can run past the
  deadline, so the full submission still needs a timing rehearsal against the
  900 s development limit.

Nothing is trained offline and no weights are bundled.

The packaged candidate passed all three full-size, seed-0 synthetic phase-1
checks on Python 3.13.15, with each scorer process restricted to one CPU core.
The total wall time was 121.919 s: UNICEF 17.626 s, World Bank 28.868 s and
UNHCR 75.425 s. These measurements qualify the pipeline and runtime, not the
score on the unreleased surveys.

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
