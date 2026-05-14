"""Linear Regression (Mini-batch + Scheduler, Real Data).

Prediction equation:
    y_hat = Xw + b
MSE objective:
    J(w, b) = (1/N) * sum_i (y_i - y_hat_i)^2
"""

import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, load_diabetes
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "linreg_lvl5_multifeature_minibatch",
        "series": "Linear Regression",
        "level": 5,
        "description": "Mini-batch linear regression on real built-in datasets.",
        "quality_thresholds": {
            "california_housing": {"val_r2_min": 0.52, "val_rmse_max": 0.95},
            "diabetes": {"val_r2_min": 0.38, "val_rmse_max": 68.0},
        },
    }


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_real_regression_data() -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        data = fetch_california_housing()
        return data.data, data.target, "california_housing"
    except Exception:
        data = load_diabetes()
        return data.data, data.target, "diabetes"


def make_dataloaders(
    batch_size: int = 128,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, Any]:
    x_np, y_np, dataset_name = _load_real_regression_data()
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=generator)
    x = x[perm]
    y = y[perm]

    split_idx = int(train_ratio * x.size(0))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train_raw, y_val_raw = y[:split_idx], y[split_idx:]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std().clamp_min(1e-6)
    y_train_norm = (y_train_raw - y_mean) / y_std
    y_val_norm = (y_val_raw - y_mean) / y_std

    train_ds = TensorDataset(x_train, y_train_norm, y_train_raw)
    val_ds = TensorDataset(x_val, y_val_norm, y_val_raw)

    return {
        "dataset_name": dataset_name,
        "y_mean": y_mean,
        "y_std": y_std,
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "train_eval_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        "input_dim": int(x_train.shape[1]),
    }


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    return nn.Linear(input_dim, 1).to(device)


def predict(
    model: nn.Module,
    x: torch.Tensor,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        pred_norm = model(x).squeeze(1)
        pred_raw = pred_norm * y_std.to(device) + y_mean.to(device)
    return pred_raw.detach().cpu()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    preds_all: List[torch.Tensor] = []
    y_true_all: List[torch.Tensor] = []

    for xb, _yb_norm, yb_raw in data_loader:
        preds = predict(model, xb, y_mean, y_std, device)
        preds_all.append(preds)
        y_true_all.append(yb_raw)

    y_pred = torch.cat(preds_all)
    y_true = torch.cat(y_true_all)

    mse = torch.mean((y_pred - y_true) ** 2)
    rmse = torch.sqrt(mse)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "mse": float(mse.item()),
        "rmse": float(rmse.item()),
        "r2": float(r2.item()),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
    epochs: int = 220,
    lr: float = 0.01,
) -> Dict[str, List[float]]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=80, gamma=0.5)

    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    for _ in range(epochs):
        model.train()
        running = 0.0
        n_items = 0

        for xb, yb_norm, _yb_raw in train_loader:
            xb = xb.to(device)
            yb_norm = yb_norm.to(device)

            optimizer.zero_grad()
            pred_norm = model(xb).squeeze(1)
            loss = criterion(pred_norm, yb_norm)
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n_items += xb.size(0)

        train_loss_history.append(float(running / max(n_items, 1)))
        val_metrics = evaluate(model, val_loader, y_mean, y_std, device)
        val_loss_history.append(val_metrics["mse"])
        scheduler.step()

    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }


def save_artifacts(task_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(task_dir, exist_ok=True)
    out_path = os.path.join(task_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> int:
    metadata = get_task_metadata()
    set_seed(42)
    device = get_device()

    dl = make_dataloaders(batch_size=128, train_ratio=0.8, seed=42)
    model = build_model(dl["input_dim"], device)

    history = train(
        model=model,
        train_loader=dl["train_loader"],
        val_loader=dl["val_loader"],
        y_mean=dl["y_mean"],
        y_std=dl["y_std"],
        device=device,
        epochs=220,
        lr=0.01,
    )

    train_metrics = evaluate(model, dl["train_eval_loader"], dl["y_mean"], dl["y_std"], device)
    val_metrics = evaluate(model, dl["val_loader"], dl["y_mean"], dl["y_std"], device)

    summary = {
        "task_id": metadata["task_id"],
        "device": str(device),
        "dataset": dl["dataset_name"],
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "loss_history": history["train_loss_history"],
        "val_loss_history": history["val_loss_history"],
        "final_metrics": val_metrics,
    }

    artifact_path = save_artifacts(os.path.dirname(__file__), summary)
    summary["artifact"] = artifact_path
    print(json.dumps(summary, indent=2))

    thresholds = metadata["quality_thresholds"][dl["dataset_name"]]

    try:
        assert val_metrics["r2"] > thresholds["val_r2_min"]
        assert val_metrics["rmse"] < thresholds["val_rmse_max"]
        return 0
    except AssertionError:
        print("Check failed: validation R2/RMSE did not meet the threshold.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
