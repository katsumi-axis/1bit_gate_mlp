# 1bit + gate MLP

Experiments on [greydanus/mnist1d](https://github.com/greydanus/mnist1d) with four
small sequence classifiers:

- `mlp`: a plain MLP over the 40-dimensional 1D input
- `attention_mlp`: patch embedding, attention blocks, and an MLP head
- `onebit_swiglu_mlp`: `-1/+1` binary linear layers followed by float SwiGLU gates
- `attention_onebit_mlp`: attention followed by the 1bit SwiGLU MLP

The 1bit models do not keep shadow or latent float weights for binary parameters.
Binary weights are updated directly with Bop flips over `-1/+1`. Float parameters
such as SwiGLU gates, LayerNorm, heads, and attention layers are updated normally
with AdamW.

## Results

These charts show 100-epoch train accuracy. `h128` and `h256` refer to
`--hidden-dim`. The plain attention MLP uses `depth=2` for h128 and `depth=1`
for h256 because the h256 depth-1 setting overfits cleanly. For
attention+1bit MLP, `t1/t2/t4/t8` correspond to `--binary-bases 1/2/4/8`.

### MLP

Plain MLP vs 1bit SwiGLU MLP.

![MLP vs 1bit MLP accuracy](docs/assets/mlp_vs_onebit_accuracy.png)

### Attention

Attention MLP vs attention+1bit SwiGLU MLP.

![Attention MLP vs attention 1bit MLP accuracy](docs/assets/attention_vs_onebit_accuracy.png)

## Setup

```bash
nix develop
uv sync
```

## Train

```bash
uv run train-1d-mnist --model mlp --epochs 5
uv run train-1d-mnist --model attention_mlp --epochs 5
uv run train-1d-mnist --model onebit_swiglu_mlp --epochs 5 --weight-decay 0
uv run train-1d-mnist --model attention_onebit_mlp --binary-bases 8 --epochs 5 --weight-decay 0
```

The MNIST-1D frozen dataset is downloaded on first run to
`data/mnist1d/raw/mnist1d_data.pkl`. The default split is train 4000 / test 1000
with input length 40.

## Useful Options

```bash
uv run train-1d-mnist --help
uv run train-1d-mnist --model attention_mlp --patch-size 5 --depth 2 --hidden-dim 128
uv run train-1d-mnist --model mlp --batch-size 256 --lr 1e-3
uv run train-1d-mnist --model onebit_swiglu_mlp --bop-threshold 1e-4 --bop-gamma 1e-4
uv run train-1d-mnist --model attention_onebit_mlp --binary-bases 8 --patch-size 5
```
