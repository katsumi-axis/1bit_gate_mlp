from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from onebit_gate_mlp.data import batch_iter, load_mnist1d
from onebit_gate_mlp.models import build_model
from onebit_gate_mlp.optim import Bop, split_tree


def accuracy(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    *,
    limit: int | None = None,
) -> float:
    if limit is not None:
        x = x[:limit]
        y = y[:limit]

    correct = 0
    total = 0
    rng = np.random.default_rng(0)
    for xb, yb in batch_iter(x, y, batch_size, shuffle=False, rng=rng):
        logits = model(mx.array(xb))
        preds = mx.argmax(logits, axis=1)
        correct += int(mx.sum(preds == mx.array(yb)).item())
        total += yb.shape[0]
    return correct / total


def train(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)

    (train_x, train_y), (test_x, test_y) = load_mnist1d(Path(args.data_dir))
    if args.train_limit is not None:
        train_x = train_x[: args.train_limit]
        train_y = train_y[: args.train_limit]

    model = build_model(
        args.model,
        input_dim=train_x.shape[1],
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        patch_size=args.patch_size,
        swiglu_ratio=args.swiglu_ratio,
        binary_bases=args.binary_bases,
    )
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)
    binary_optimizer = Bop(threshold=args.bop_threshold, gamma=args.bop_gamma)
    is_binary_parameter = getattr(model, "is_binary_parameter", lambda _path: False)

    def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
        logits = model(xb)
        return nn.losses.cross_entropy(logits, yb, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    metrics_file = None
    metrics_writer = None
    if args.metrics_csv is not None:
        metrics_path = Path(args.metrics_csv)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_file = metrics_path.open("w", newline="")
        metrics_writer = csv.DictWriter(
            metrics_file,
            fieldnames=["epoch", "train_loss", "train_acc", "test_acc", "flip_rate", "time"],
        )
        metrics_writer.writeheader()

    try:
        for epoch in range(1, args.epochs + 1):
            start = time.time()
            running_loss = 0.0
            seen = 0
            pbar = tqdm(
                batch_iter(train_x, train_y, args.batch_size, shuffle=True, rng=rng),
                total=(train_x.shape[0] + args.batch_size - 1) // args.batch_size,
                desc=f"epoch {epoch}/{args.epochs}",
                disable=args.no_progress,
            )
            for step, (xb, yb) in enumerate(pbar, start=1):
                xb_mx = mx.array(xb)
                yb_mx = mx.array(yb)
                loss, grads = loss_and_grad(model, xb_mx, yb_mx)
                binary_grads, float_grads = split_tree(grads, is_binary_parameter)
                binary_params, float_params = split_tree(
                    model.trainable_parameters(), is_binary_parameter
                )

                if float_grads:
                    model.update(optimizer.apply_gradients(float_grads, float_params), strict=False)
                if binary_grads:
                    model.update(
                        binary_optimizer.update(binary_params, binary_grads), strict=False
                    )
                mx.eval(model.parameters(), optimizer.state, binary_optimizer.state)

                batch_count = yb.shape[0]
                running_loss += float(loss.item()) * batch_count
                seen += batch_count
                pbar.set_postfix(loss=running_loss / seen)
                if args.max_steps is not None and step >= args.max_steps:
                    break

            train_acc = accuracy(model, train_x, train_y, args.batch_size, limit=args.train_eval_limit)
            test_acc = accuracy(model, test_x, test_y, args.batch_size, limit=args.eval_limit)
            elapsed = time.time() - start
            train_loss = running_loss / seen
            print(
                f"epoch={epoch} "
                f"train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.4f} "
                f"test_acc={test_acc:.4f} "
                f"flip_rate={binary_optimizer.last_flip_rate:.6f} "
                f"time={elapsed:.1f}s"
            )
            if metrics_writer is not None:
                metrics_writer.writerow(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "flip_rate": binary_optimizer.last_flip_rate,
                        "time": elapsed,
                    }
                )
                metrics_file.flush()
    finally:
        if metrics_file is not None:
            metrics_file.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLX models on 1D MNIST.")
    parser.add_argument(
        "--model",
        choices=["mlp", "attention_mlp", "onebit_swiglu_mlp", "attention_onebit_mlp"],
        default="mlp",
    )
    parser.add_argument("--data-dir", default="data/mnist1d")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--swiglu-ratio", type=float, default=2.0)
    parser.add_argument("--binary-bases", type=int, default=1)
    parser.add_argument("--bop-threshold", type=float, default=1e-4)
    parser.add_argument("--bop-gamma", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--train-eval-limit", type=int, default=None)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--metrics-csv", default=None)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
