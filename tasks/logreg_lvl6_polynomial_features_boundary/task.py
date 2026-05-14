"""Logistic Regression (Polynomial Expansion on Real Data).

Logistic probability:
    p(y=1|x) = sigma(w^T phi(x) + b)
Polynomial feature map (degree 2 subset):
    phi(x1, x2) = [x1, x2, x1^2, x2^2, x1*x2]
"""

import json
import os
import random
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from sklearn.datasets import load_wine
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "logreg_lvl6_polynomial_features_boundary",
        "series": "Logistic Regression",
        "level": 6,
        "description": "Compare linear logistic regression to polynomial expansion on Wine data.",
        "quality_thresholds": {
            "poly_val_accuracy_min": 0.82,
            "poly_val_f1_min": 0.78,
            "accuracy_gain_min": 0.03,
        },
    }


def set_seed(seed: int = 99) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _poly_features(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, 0:1]
    x2 = x[:, 1:2]
    return torch.cat([x1, x2, x1.pow(2), x2.pow(2), x1 * x2], dim=1)


def make_dataloaders(
    batch_size: int = 64,
    train_ratio: float = 0.7,
    seed: int = 99,
) -> Dict[str, Any]:
    data = load_wine()
    x = torch.tensor(data.data[:, [9, 10]], dtype=torch.float32)
    y = torch.tensor((data.target == 0).astype(np.float32), dtype=torch.float32)

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.size(0), generator=generator)
    x = x[perm]
    y = y[perm]

    split_idx = int(train_ratio * x.size(0))
    x_train_raw, x_val_raw = x[:split_idx], x[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    mean = x_train_raw.mean(dim=0, keepdim=True)
    std = x_train_raw.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train_raw = (x_train_raw - mean) / std
    x_val_raw = (x_val_raw - mean) / std

    x_train_poly = _poly_features(x_train_raw)
    x_val_poly = _poly_features(x_val_raw)

    linear_train_ds = TensorDataset(x_train_raw, y_train)
    linear_val_ds = TensorDataset(x_val_raw, y_val)
    poly_train_ds = TensorDataset(x_train_poly, y_train)
    poly_val_ds = TensorDataset(x_val_poly, y_val)

    return {
        "dataset_name": "wine_class0_vs_rest",
        "x_val_raw": x_val_raw,
        "y_val": y_val,
        "linear_train_loader": DataLoader(linear_train_ds, batch_size=batch_size, shuffle=True),
        "linear_val_loader": DataLoader(linear_val_ds, batch_size=batch_size, shuffle=False),
        "linear_train_eval_loader": DataLoader(linear_train_ds, batch_size=batch_size, shuffle=False),
        "poly_train_loader": DataLoader(poly_train_ds, batch_size=batch_size, shuffle=True),
        "poly_val_loader": DataLoader(poly_val_ds, batch_size=batch_size, shuffle=False),
        "poly_train_eval_loader": DataLoader(poly_train_ds, batch_size=batch_size, shuffle=False),
        "linear_input_dim": int(x_train_raw.shape[1]),
        "poly_input_dim": int(x_train_poly.shape[1]),
    }


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    return nn.Linear(input_dim, 1).to(device)


def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x.to(device)).squeeze(1))
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
    epochs: int = 420,
    lr: float = 0.03,
) -> Dict[str, List[float]]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

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
            loss.backward()
            optimizer.step()

            running += loss.item() * xb.size(0)
            n_items += xb.size(0)

        train_loss_history.append(float(running / max(n_items, 1)))

        model.eval()
        with torch.no_grad():
            v_running = 0.0
            v_items = 0
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                vloss = criterion(model(xb).squeeze(1), yb)
                v_running += vloss.item() * xb.size(0)
                v_items += xb.size(0)
        val_loss_history.append(float(v_running / max(v_items, 1)))

    return {
        "loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
    }


def _save_boundary_plot(
    linear_model: nn.Module,
    poly_model: nn.Module,
    x_val_raw: torch.Tensor,
    y_val: torch.Tensor,
    out_png: str,
    device: torch.device,
) -> str:
    if plt is None:
        out_txt = out_png.replace(".png", ".txt")
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("matplotlib unavailable; boundary plot skipped\n")
        return out_txt

    mins = x_val_raw.min(dim=0).values - 0.5
    maxs = x_val_raw.max(dim=0).values + 0.5

    grid_x1 = torch.linspace(mins[0].item(), maxs[0].item(), 220)
    grid_x2 = torch.linspace(mins[1].item(), maxs[1].item(), 220)
    xx, yy = torch.meshgrid(grid_x1, grid_x2, indexing="xy")
    grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    with torch.no_grad():
        lin_probs = torch.sigmoid(linear_model(grid.to(device)).squeeze(1)).reshape(xx.shape).cpu()
        poly_probs = torch.sigmoid(poly_model(_poly_features(grid).to(device)).squeeze(1)).reshape(xx.shape).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, probs, title in [
        (axes[0], lin_probs, "Linear Features"),
        (axes[1], poly_probs, "Polynomial Features"),
    ]:
        contour = ax.contourf(xx.numpy(), yy.numpy(), probs.numpy(), levels=30, cmap="coolwarm", alpha=0.75)
        ax.contour(xx.numpy(), yy.numpy(), probs.numpy(), levels=[0.5], colors="black", linewidths=1.25)
        ax.scatter(
            x_val_raw[:, 0].numpy(),
            x_val_raw[:, 1].numpy(),
            c=y_val.numpy(),
            cmap="coolwarm",
            edgecolors="k",
            s=24,
        )
        ax.set_title(title)
        ax.set_xlabel("feature_9")
        ax.set_ylabel("feature_10")

    fig.colorbar(contour, ax=axes.ravel().tolist(), shrink=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    return out_png


def save_artifacts(task_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(task_dir, exist_ok=True)
    out_path = os.path.join(task_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


def main() -> int:
    metadata = get_task_metadata()
    set_seed(99)
    device = get_device()
    dl = make_dataloaders(batch_size=64, train_ratio=0.7, seed=99)

    linear_model = build_model(dl["linear_input_dim"], device)
    poly_model = build_model(dl["poly_input_dim"], device)

    linear_history = train(
        model=linear_model,
        train_loader=dl["linear_train_loader"],
        val_loader=dl["linear_val_loader"],
        device=device,
        epochs=420,
        lr=0.03,
    )
    poly_history = train(
        model=poly_model,
        train_loader=dl["poly_train_loader"],
        val_loader=dl["poly_val_loader"],
        device=device,
        epochs=420,
        lr=0.03,
    )

    linear_train_metrics = evaluate(linear_model, dl["linear_train_eval_loader"], device)
    linear_val_metrics = evaluate(linear_model, dl["linear_val_loader"], device)
    poly_train_metrics = evaluate(poly_model, dl["poly_train_eval_loader"], device)
    poly_val_metrics = evaluate(poly_model, dl["poly_val_loader"], device)

    boundary_path = _save_boundary_plot(
        linear_model=linear_model,
        poly_model=poly_model,
        x_val_raw=dl["x_val_raw"],
        y_val=dl["y_val"],
        out_png=os.path.join(os.path.dirname(__file__), "logreg_lvl6_linear_vs_poly_boundary.png"),
        device=device,
    )

    accuracy_gain = poly_val_metrics["accuracy"] - linear_val_metrics["accuracy"]

    summary = {
        "task_id": metadata["task_id"],
        "device": str(device),
        "dataset": dl["dataset_name"],
        "linear": {
            "train_metrics": linear_train_metrics,
            "val_metrics": linear_val_metrics,
            "loss_history": linear_history["loss_history"],
            "val_loss_history": linear_history["val_loss_history"],
        },
        "polynomial": {
            "train_metrics": poly_train_metrics,
            "val_metrics": poly_val_metrics,
            "loss_history": poly_history["loss_history"],
            "val_loss_history": poly_history["val_loss_history"],
        },
        "accuracy_gain": float(accuracy_gain),
        "boundary_artifact": boundary_path,
        "final_metrics": poly_val_metrics,
    }

    artifact_path = save_artifacts(os.path.dirname(__file__), summary)
    summary["artifact"] = artifact_path
    print(json.dumps(summary, indent=2))

    th = metadata["quality_thresholds"]
    try:
        assert poly_val_metrics["accuracy"] > th["poly_val_accuracy_min"]
        assert poly_val_metrics["f1"] > th["poly_val_f1_min"]
        assert accuracy_gain > th["accuracy_gain_min"]
        return 0
    except AssertionError:
        print("Check failed: polynomial model did not beat thresholds/baseline enough.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
