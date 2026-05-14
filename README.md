# CMPE188 HW2

This repo contains the four new PyTorch tasks for HW2.

## Files
- `tasks/linreg_lvl5_multifeature_minibatch/task.py`
- `tasks/linreg_lvl6_huber_outlier_robust/task.py`
- `tasks/logreg_lvl5_l2_regularized_binary/task.py`
- `tasks/logreg_lvl6_polynomial_features_boundary/task.py`
- `ml_tasks.json` (grader-friendly filename with just these four task entries + protocol block)
- `ml_tasks_four_new_tasks.json` (same content as `ml_tasks.json`)

## Run
```bash
python3 -m pip install -r requirements.txt
python3 tasks/<task_id>/task.py
```

Each script trains, evaluates, prints metrics, and exits non-zero if checks fail.
