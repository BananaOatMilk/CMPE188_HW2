"""Logistic Regression (Binary + L2, Real Data).

Probability:
    p(y=1|x) = sigma(w^T x + b)
Objective with explicit L2 penalty:
    J(w, b) = BCEWithLogits(y, w^T x + b) + lambda * ||w||_2^2
"""

import json
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "logreg_lvl5_l2_regularized_binary",
        "series": "Logistic Regression",
        "level": 5,
        "description": "Binary logistic regression with explicit L2 regularization.",
        "quality_thresholds": {
            "val_f1_min": 0.95,
            "f1_drop_tolerance": 0.02,
        },
    }


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_dataloaders(
    batch_size: int = 64,
    train_ratio: float = 0.7,
    seed: int = 7,
) -> Dict[str, Any]:
    x_np, y_np = load_breast_cancer(return_X_y=True)

    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=generator)
    x = x[perm]
    y = y[perm]

    split_idx = int(train_ratio * x.size(0))
    x_train, x_val = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    x_mean = x_train.mean(dim=0, keepdim=True)
    x_std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - x_mean) / x_std
    x_val = (x_val - x_mean) / x_std

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    return {
        "dataset_name": "breast_cancer",
        "train_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        "val_loader": DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        "train_eval_loader": DataLoader(train_ds, batch_size=batch_size, shuffle=False),
        "input_dim": int(x_train.shape[1]),
    }


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    return nn.Linear(input_dim, 1).to(device)


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device)).squeeze(1)
        probs = torch.sigmoid(logits)
    return probs.detach().cpu()


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_prob_list: List[torch.Tensor] = []
    y_true_list: List[torch.Tensor] = []

    for xb, yb in data_loader:
        y_prob_list.append(predict(model, xb, device))
        y_true_list.append(yb)

    y_prob = torch.cat(y_prob_list)
    y_true = torch.cat(y_true_list)
    y_pred = (y_prob >= threshold).float()

    tp = torch.sum((y_pred == 1) & (y_true == 1)).item()
    tn = torch.sum((y_pred == 0) & (y_true == 0)).item()
    fp = torch.sum((y_pred == 1) & (y_true == 0)).item()
    fn = torch.sum((y_pred == 0) & (y_true == 1)).item()

    eps = 1e-8
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2.0 * precision * recall / max(precision + recall, eps)

    mse = torch.mean((y_prob - y_true) ** 2)
    ss_res = torch.sum((y_true - y_prob) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2).clamp_min(1e-12)
    r2 = 1.0 - ss_res / ss_tot

    return {
        "mse": float(mse.item()),
        "r2": float(r2.item()),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    l2_lambda: float,
    epochs: int = 260,
    lr: float = 0.02,
) -> Dict[str, Any]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state: Dict[str, torch.Tensor] = {}
    best_f1 = -1.0
    train_loss_history: List[float] = []
    val_loss_history: List[float] = []

    for _ in range(epochs):
        model.train()
        running = 0.0
        n_items = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb).squeeze(1)
            loss = criterion(logits, yb)
            if l2_lambda > 0.0:
                loss = loss + l2_lambda * torch.sum(model.weight.pow(2))
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n_items += xb.size(0)

        train_loss_history.append(float(running / max(n_items, 1)))

        with torch.no_grad():
            val_running = 0.0
            val_items = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb).squeeze(1)
                vloss = criterion(logits, yb)
                val_running += vloss.item() * xb.size(0)
                val_items += xb.size(0)
            val_loss_history.append(float(val_running / max(val_items, 1)))

        val_metrics = evaluate(model, val_loader, device)
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "weight": model.weight.detach().clone(),
                "bias": model.bias.detach().clone(),
            }

    if best_state:
        with torch.no_grad():
            model.weight.copy_(best_state["weight"])
            model.bias.copy_(best_state["bias"])

    return {
        "loss_history": train_loss_history,
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
    set_seed(7)
    device = get_device()
    dl = make_dataloaders(batch_size=64, train_ratio=0.7, seed=7)

    baseline_model = build_model(dl["input_dim"], device)
    reg_model = build_model(dl["input_dim"], device)

    baseline_history = train(
        model=baseline_model,
        train_loader=dl["train_loader"],
        val_loader=dl["val_loader"],
        device=device,
        l2_lambda=0.0,
        epochs=260,
        lr=0.02,
    )
    reg_history = train(
        model=reg_model,
        train_loader=dl["train_loader"],
        val_loader=dl["val_loader"],
        device=device,
        l2_lambda=0.01,
        epochs=260,
        lr=0.02,
    )

    baseline_train_metrics = evaluate(baseline_model, dl["train_eval_loader"], device)
    baseline_val_metrics = evaluate(baseline_model, dl["val_loader"], device)
    reg_train_metrics = evaluate(reg_model, dl["train_eval_loader"], device)
    reg_val_metrics = evaluate(reg_model, dl["val_loader"], device)

    baseline_norm = float(torch.norm(baseline_model.weight.detach()).item())
    reg_norm = float(torch.norm(reg_model.weight.detach()).item())

    summary = {
        "task_id": metadata["task_id"],
        "device": str(device),
        "dataset": dl["dataset_name"],
        "baseline": {
            "train_metrics": baseline_train_metrics,
            "val_metrics": baseline_val_metrics,
            "weight_norm": baseline_norm,
            "loss_history": baseline_history["loss_history"],
            "val_loss_history": baseline_history["val_loss_history"],
        },
        "regularized": {
            "train_metrics": reg_train_metrics,
            "val_metrics": reg_val_metrics,
            "weight_norm": reg_norm,
            "loss_history": reg_history["loss_history"],
            "val_loss_history": reg_history["val_loss_history"],
        },
        "final_metrics": reg_val_metrics,
    }

    artifact_path = save_artifacts(os.path.dirname(__file__), summary)
    summary["artifact"] = artifact_path
    print(json.dumps(summary, indent=2))

    try:
        assert reg_val_metrics["f1"] > metadata["quality_thresholds"]["val_f1_min"]
        assert reg_val_metrics["f1"] >= baseline_val_metrics["f1"] - metadata["quality_thresholds"]["f1_drop_tolerance"]
        assert reg_norm < baseline_norm
        return 0
    except AssertionError:
        print("Regularized logistic regression validation assertions failed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
