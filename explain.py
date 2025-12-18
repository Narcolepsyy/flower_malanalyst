"""
Compute lightweight, human-readable feature attributions for the latest global model.

This script reads the aggregated model saved by the Flower server (npz), rebuilds the
model, and scores feature importance using:
- Logistic regression: absolute coefficient magnitude.
- MLP: input-gradient magnitude averaged over a background sample.

Outputs a JSON file (default: state/explanations.json) that the dashboard can consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from federated_malware.dataset_utils import load_feature_names, load_malmem
from federated_malware.model_utils import NumpyLogisticModel, TorchMLPModel


def _load_parameters(npz_path: Path) -> Tuple[List[np.ndarray], int | None]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Model weights not found at {npz_path}")
    data = np.load(npz_path, allow_pickle=False)
    round_num: int | None = None
    if "round" in data:
        try:
            round_num = int(data["round"].reshape(-1)[0])
        except Exception:
            round_num = None

    param_keys = [k for k in data.files if k.startswith("p")]
    if not param_keys:
        param_keys = [k for k in data.files if k != "round"]

    def _key_sort(key: str) -> int:
        suffix = key[1:] if key.startswith("p") else key
        return int(suffix) if suffix.isdigit() else 0

    params = [data[k] for k in sorted(param_keys, key=_key_sort)]
    return params, round_num


def _infer_model_type(params: List[np.ndarray], override: str | None) -> str:
    if override:
        return override
    # CatBoost models are serialized as single byte arrays
    if len(params) == 1 and params[0].dtype == np.uint8:
        return "catboost"
    if len(params) == 2 and params[1].shape[-1] == 1:
        return "logreg"
    if len(params) == 6 and params[-1].shape[-1] == 1:
        return "mlp"
    # Hybrid quantum has 8 parameters (6 classical + 2 quantum layer params)
    if len(params) == 8:
        return "hybrid-quantum"
    raise ValueError("Unable to infer model type from parameter shapes; pass --model explicitly")


def _build_model(params: List[np.ndarray], model_type: str, n_features: int):
    if model_type == "logreg":
        model = NumpyLogisticModel(n_features=n_features)
        model.set_parameters(params)
        return model
    elif model_type == "catboost":
        import pickle
        if len(params[0]) == 0:
            raise ValueError("CatBoost model not yet trained (empty parameters)")
        model = pickle.loads(params[0].tobytes())
        return model
    else:
        # torch layer shapes: [out_features, in_features]
        hidden1 = params[0].shape[0]
        hidden2 = params[2].shape[0]
        model = TorchMLPModel(n_features=n_features, hidden1=hidden1, hidden2=hidden2)
        model.set_parameters(params)
        return model


def _logreg_importance(model: NumpyLogisticModel) -> np.ndarray:
    return np.abs(model.weights)


def _mlp_gradient_importance(model: TorchMLPModel, x: np.ndarray) -> np.ndarray:
    model.model.eval()
    xb = torch.tensor(x, dtype=torch.float32, device=model.device, requires_grad=True)
    out = torch.sigmoid(model.model(xb)).mean()
    out.backward()
    grads = xb.grad.detach().abs().mean(dim=0).cpu().numpy()
    return grads


def _mlp_weight_importance(model: TorchMLPModel) -> np.ndarray:
    """Proxy importance from first-layer weights (robust even if gradients vanish)."""
    first_layer = next(model.model.parameters())  # weight matrix of shape [hidden1, n_features]
    return first_layer.detach().abs().mean(dim=0).cpu().numpy()


def _catboost_importance(model, feature_names: List[str]) -> np.ndarray:
    """Get CatBoost built-in feature importance."""
    try:
        importance = model.get_feature_importance()
        return np.array(importance)
    except Exception:
        # Fallback: return uniform importance
        return np.ones(len(feature_names))


def _hybrid_quantum_importance(params: List[np.ndarray]) -> np.ndarray:
    """Get hybrid quantum model importance from first classical layer weights."""
    # First layer weights shape: [64, n_features]
    first_layer = params[0]
    return np.abs(first_layer).mean(axis=0)


def _format_top_features(scores: np.ndarray, feature_names: List[str], k: int) -> List[dict]:
    k = min(k, len(scores))
    idxs = np.argsort(-scores)[:k]
    formatted = []
    for idx in idxs:
        name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
        formatted.append({"feature": name, "score": float(scores[idx])})
    return formatted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate XAI feature attributions for the global FL model")
    parser.add_argument(
        "--data-path",
        type=str,
        default="Obfuscated-MalMem2022.csv",
        help="CSV path to the MalMem dataset (used to infer feature names and scaling)",
    )
    parser.add_argument(
        "--model-weights",
        type=str,
        default="state/latest_model.npz",
        help="Aggregated model weights saved by the Flower server",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["logreg", "mlp", "catboost", "hybrid-quantum"],
        default=None,
        help="Force model type if auto-detection from weights fails",
    )
    parser.add_argument("--top-k", type=int, default=10, help="How many features to keep in the output")
    parser.add_argument(
        "--background-size",
        type=int,
        default=256,
        help="Rows to sample for gradient-based importance (MLP only)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["auto", "grad", "weights"],
        default="auto",
        help="Attribution method for MLP: gradient saliency, first-layer weights, or auto-fallback",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="state/explanations.json",
        help="Where to write the explanation JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model_weights)
    params, round_num = _load_parameters(model_path)

    feature_names = load_feature_names(args.data_path)
    x, _, _ = load_malmem(args.data_path)
    model_type = _infer_model_type(params, args.model)
    model = _build_model(params, model_type, n_features=x.shape[1])

    if model_type == "logreg":
        scores = _logreg_importance(model)
    elif model_type == "catboost":
        scores = _catboost_importance(model, feature_names)
    elif model_type == "hybrid-quantum":
        scores = _hybrid_quantum_importance(params)
    else:
        if len(x) == 0:
            raise ValueError("Dataset is empty; cannot compute gradient-based importances")
        bg_size = max(1, min(args.background_size, len(x)))
        if args.method in ("grad", "auto"):
            scores = _mlp_gradient_importance(model, x[:bg_size])
        else:
            scores = _mlp_weight_importance(model)

        # If gradients vanish (all ~0), fall back to weight-based importance in auto mode
        if args.method == "auto" and np.allclose(scores, 0):
            scores = _mlp_weight_importance(model)
            print("[explain] Gradient attributions were near-zero; fell back to weight magnitudes.")

    top_features = _format_top_features(scores, feature_names, args.top_k)
    output = {
        "round": round_num,
        "model_type": model_type,
        "model_weights": str(model_path),
        "method": args.method,
        "background_size": None if model_type == "logreg" else min(args.background_size, len(x)),
        "top_features": top_features,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))

    print(f"Saved explanations to {out_path}")
    print("Top features:")
    for item in top_features:
        print(f"  {item['feature']}: {item['score']:.4f}")


if __name__ == "__main__":
    main()
