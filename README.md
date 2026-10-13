# SimulacraBench submission: stack_trees_v8

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
ranges from 0 to 1 in steps of 0.1. Each selected model is refitted on the
visible training rows.

Tree outputs use a single temperature selected from 0.75, 1, 1.5 and 2.

Complete count predictions are computed first. Optional model workers run
serially, with a 20-second limit per item. The fitting budgets are 90 seconds
for UNICEF, 180 for World Bank and 450 for UNHCR. An incomplete worker leaves
the item's last complete prediction intact.

## Validation

The wider tree-weight grid was compared with frozen v6 on three fixed synthetic
fixtures. Each used 6,000 training respondents, 20,000 base outer respondents and
20,000 outer respondents with a different GIVEN distribution. Predictions
covered six gated three-class targets and two ungated targets.

On the two predeclared nonlinear targets, v8 reduced mean epsilon-mixed NLL by
0.10333637 nats per cell. The paired 95% bootstrap interval was 0.10167641 to
0.10501405. All 96 optional calls completed. Six nonlinear v8 selections used
pure trees, while v6 selected its maximum tree weight of 0.5. The fixed null
and distribution-shift regression checks passed.

The official runner passed full-size public sandboxes for all three instruments.
The three scorer runs took 435.86 seconds in total, with peak RSS of 458,870,784
bytes. Of 501 optional calls, 496 completed. Incomplete calls retained the last
complete prediction, and no child processes remained after prediction.

These are synthetic engineering checks using previously evaluated fixtures.
The bootstrap interval is conditional on three fitted seeds and does not
estimate performance on private competition data. The best Development entry
remains v5, submission 979285, with mean skill 0.63. V8 has not been submitted.

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
