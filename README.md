# CMPE188 HW2

This repo is my HW2 submission with four new PyTorch tasks.

## Files
- `tasks/linreg_lvl5_multifeature_minibatch/task.py`
- `tasks/linreg_lvl6_huber_outlier_robust/task.py`
- `tasks/logreg_lvl5_l2_regularized_binary/task.py`
- `tasks/logreg_lvl6_polynomial_features_boundary/task.py`
- `ml_tasks.json` (contains the four task entries and protocol block)
- `requirements.txt`

## Run
```bash
python3 -m pip install -r requirements.txt
python3 tasks/<task_id>/task.py
```

Each script trains, evaluates, prints metrics, and exits non-zero if checks fail.
