"""Linear Regression (Robust Huber vs MSE, Real Data).

Huber loss for residual r and threshold delta:
    L(r) = 0.5 * r^2                         if |r| <= delta
           delta * (|r| - 0.5 * delta)       otherwise

Compared with MSE, Huber is less sensitive to large outliers.
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
        "task_id": "linreg_lvl6_huber_outlier_robust",
        "series": "Linear Regression",
        "level": 6,
        "description": "Compare MSE vs Huber under training-target outlier corruption.",
        "quality_thresholds": {
            "val_r2_min": 0.35,
            "mae_improvement_min": -0.01,
        },
    }


def set_seed(seed: int = 123) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_real_regression_data() -> Tuple[np.ndarray, np.ndarray, str, float]:
    try:
        data = fetch_california_housing()
        return data.data, data.target, "california_housing", 8.0
    except Exception:
        data = load_diabetes()
        return data.data, data.target, "diabetes", 180.0


def make_dataloaders(
    batch_size: int = 128,
    train_ratio: float = 0.8,
    seed: int = 123,
    outlier_fraction: float = 0.20,
) -> Dict[str, Any]:
    x_np, y_np, dataset_name, outlier_scale = _load_real_regression_data()

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=generator)
    x = x[perm]
    y = y[perm]

    split_idx = int(train_ratio * x.size(0))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train_clean = y[:split_idx]
    y_val = y[split_idx:]

    y_train_corrupted = y_train_clean.clone()
    n_outliers = int(split_idx * outlier_fraction)
    outlier_idx = torch.randperm(split_idx, generator=torch.Generator().manual_seed(seed + 1))[:n_outliers]
    noise = outlier_scale * torch.randn(n_outliers, generator=torch.Generator().manual_seed(seed + 2))
    y_train_corrupted[outlier_idx] += noise

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    y_mean = y_train_corrupted.mean()
    y_std = y_train_corrupted.std().clamp_min(1e-6)
    y_train_norm = (y_train_corrupted - y_mean) / y_std
    y_val_norm = (y_val - y_mean) / y_std

    train_ds = TensorDataset(x_train, y_train_norm, y_train_clean)
    val_ds = TensorDataset(x_val, y_val_norm, y_val)

    return {
        "dataset_name": dataset_name,
        "outlier_fraction": outlier_fraction,
        "outlier_scale": outlier_scale,
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
        pred_norm = model(x.to(device)).squeeze(1)
        pred_raw = pred_norm * y_std.to(device) + y_mean.to(device)
    return pred_raw.detach().cpu()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
    device: torch.device,
) -> Dict[str, float]:
    y_pred_all: List[torch.Tensor] = []
    y_true_all: List[torch.Tensor] = []

    for xb, _yb_norm, yb_raw in data_loader:
        y_pred_all.append(predict(model, xb, y_mean, y_std, device))
        y_true_all.append(yb_raw)

    y_pred = torch.cat(y_pred_all)
    y_true = torch.cat(y_true_all)

    mae = torch.mean(torch.abs(y_pred - y_true))
    mse = torch.mean((y_pred - y_true) ** 2)
    rmse = torch.sqrt(mse)
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "mae": float(mae.item()),
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
    loss_name: str,
    epochs: int = 240,
    lr: float = 0.02,
) -> Dict[str, List[float]]:
    if loss_name == "mse":
        criterion: nn.Module = nn.MSELoss()
    elif loss_name == "huber":
        criterion = nn.SmoothL1Loss(beta=1.0)
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loss_history: List[float] = []
    val_mse_history: List[float] = []

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
        val_mse_history.append(val_metrics["mse"])

    return {
        "train_loss_history": train_loss_history,
        "val_loss_history": val_mse_history,
    }


def save_artifacts(task_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(task_dir, exist_ok=True)
    out_path = os.path.join(task_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> int:
    metadata = get_task_metadata()
    set_seed(123)
    device = get_device()
    dl = make_dataloaders(batch_size=128, train_ratio=0.8, seed=123, outlier_fraction=0.2)

    mse_model = build_model(dl["input_dim"], device)
    huber_model = build_model(dl["input_dim"], device)

    mse_history = train(
        model=mse_model,
        train_loader=dl["train_loader"],
        val_loader=dl["val_loader"],
        y_mean=dl["y_mean"],
        y_std=dl["y_std"],
        device=device,
        loss_name="mse",
        epochs=240,
        lr=0.02,
    )
    huber_history = train(
        model=huber_model,
        train_loader=dl["train_loader"],
        val_loader=dl["val_loader"],
        y_mean=dl["y_mean"],
        y_std=dl["y_std"],
        device=device,
        loss_name="huber",
        epochs=240,
        lr=0.02,
    )

    mse_train_metrics = evaluate(mse_model, dl["train_eval_loader"], dl["y_mean"], dl["y_std"], device)
    mse_val_metrics = evaluate(mse_model, dl["val_loader"], dl["y_mean"], dl["y_std"], device)
    huber_train_metrics = evaluate(huber_model, dl["train_eval_loader"], dl["y_mean"], dl["y_std"], device)
    huber_val_metrics = evaluate(huber_model, dl["val_loader"], dl["y_mean"], dl["y_std"], device)

    mae_improvement = mse_val_metrics["mae"] - huber_val_metrics["mae"]

    summary = {
        "task_id": metadata["task_id"],
        "device": str(device),
        "dataset": dl["dataset_name"],
        "outlier_fraction": dl["outlier_fraction"],
        "outlier_scale": dl["outlier_scale"],
        "mse_model": {
            "train_metrics": mse_train_metrics,
            "val_metrics": mse_val_metrics,
            "loss_history": mse_history["train_loss_history"],
            "val_loss_history": mse_history["val_loss_history"],
        },
        "huber_model": {
            "train_metrics": huber_train_metrics,
            "val_metrics": huber_val_metrics,
            "loss_history": huber_history["train_loss_history"],
            "val_loss_history": huber_history["val_loss_history"],
        },
        "mae_improvement": float(mae_improvement),
        "final_metrics": huber_val_metrics,
    }

    artifact_path = save_artifacts(os.path.dirname(__file__), summary)
    summary["artifact"] = artifact_path
    print(json.dumps(summary, indent=2))

    try:
        assert mae_improvement >= metadata["quality_thresholds"]["mae_improvement_min"]
        assert huber_val_metrics["r2"] >= metadata["quality_thresholds"]["val_r2_min"]
        return 0
    except AssertionError:
        print("Check failed: Huber model did not meet the expected validation metrics.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
