# SimulacraBench submission: stack_trees_v6

This submission predicts missing survey answers from the visible TRAIN rows and
GIVEN answers. It implements `predict(frame, schema)` for the organisers' runner.
It uses no external data or pretrained models.

## Approach

Each target starts with Dirichlet-smoothed counts over nested groups of GIVEN
answers. Features and smoothing strengths are selected by leave-one-out log
score. The gated sentinel is a separate response class.

A multinomial logistic regression can join this prediction, followed by a
shallow categorical histogram gradient boosting model. Three matched folds
select the blend using the evaluator's epsilon-mixed log score. The tree weight
is capped at 0.5. Each selected model is refitted on the visible training rows.

Tree outputs use a single temperature selected from 0.75, 1, 1.5 and 2.

Complete count predictions are computed first. Optional model workers run
serially, with a 20-second limit per item. The fitting budgets are 90 seconds
for UNICEF, 180 for World Bank and 450 for UNHCR. An incomplete worker leaves
the item's last complete prediction intact.

## Validation

An independent synthetic comparison fitted v5, v6 and v7 on three fixed seeds,
with 6,000 training respondents and 20,000 outer respondents in each of two
distributions per seed. Each call covered six gated three-class targets and
two ungated targets. All 144 optional workers completed, including 72 tree
selections. Outputs were legal and child processes were reaped.

V7 added separate gate and answer temperatures. Its mean gain over v6 on the
two declared calibration targets was 0.00001146 nats per cell, below the fixed
0.005 requirement. The paired 95% interval was -0.00011579 to 0.00013727.
The separate calibration was removed from the active source. These synthetic
controls do not estimate performance on the private competition data.

The best Development entry remains v5, submission 979285, with mean skill
0.63. The v6 entry, submission 981529, scored 0.6266666667. V7 was not submitted.

## Reproduce

Use Python 3.13 and install this directory's `requirements.txt` in a virtual
environment. Fetch the organisers' public runner and generate a sandbox:

```bash
git clone https://github.com/SituatedEvals/public
cd public
python -m pip install -r requirements.txt
python make_sandbox.py --schema data/unhcr.json --out _sandbox/unhcr
python score.py --submission /path/to/this/repo --data _sandbox/unhcr \
    --schema _sandbox/unhcr/schema.json --phase 1
```

Repeat with `unicef` and `world_bank` to check the other instruments. Use the
same interpreter and install the submission requirements before scoring.
The submission archive contains only the entry point and its dependencies:

```bash
cd /path/to/this/repo
zip ../submission.zip main.py requirements.txt
```
