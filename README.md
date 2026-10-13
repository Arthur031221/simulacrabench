# SimulacraBench submission: stack_trees_v7

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

For gated UNHCR targets, v7 also considers separate temperatures for the
probability of being asked and the answer distribution conditional on being
asked. Both temperatures use the fixed grid 0.75, 1, 1.5 and 2. This family must
beat the best joint-temperature blend by more than 0.0001 nats per cell on the
same folds. Otherwise the joint-temperature blend remains selected. Other
targets use joint temperature calibration.

Complete count predictions are computed first. Optional model workers run
serially, with a 20-second limit per item. The fitting budgets are 90 seconds
for UNICEF, 180 for World Bank and 450 for UNHCR. An incomplete worker leaves
the item's last complete prediction intact.

## Validation

The full-size public synthetic sandboxes passed the official phase-1 scorer
for all three instruments. Sequential CPU execution took 460.04 seconds in
total and peaked at 603,200 KiB RSS. UNHCR completed 167 of 171 eligible tree
workers. These sandboxes check the runner contract and resource limits.
They do not estimate performance on the competition's private data.

The best Development entry remains v5, submission 979285, with mean skill
0.63. The v6 entry, submission 981529, scored 0.6266666667. v7 has not been
submitted.

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
