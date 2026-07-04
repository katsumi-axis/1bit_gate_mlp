from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class MLPOnly(nn.Module):
    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, depth: int = 3, num_classes: int = 10):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        layers: list[nn.Module] = []
        dim = input_dim
        for _ in range(depth):
            layers.append(nn.Linear(dim, hidden_dim))
            layers.append(nn.GELU())
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.net(x)


def _binary_weight(shape: tuple[int, ...]) -> mx.array:
    return mx.where(mx.random.uniform(shape=shape) < 0.5, -1.0, 1.0)


class BinaryLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, *, binarize_input: bool = True):
        super().__init__()
        self.weight = _binary_weight((output_dim, input_dim))
        self.binarize_input = binarize_input
        self.scale = 1.0 / math.sqrt(input_dim)

    def __call__(self, x: mx.array) -> mx.array:
        if self.binarize_input:
            x = mx.where(x >= 0, 1.0, -1.0)
        return (x @ mx.transpose(self.weight)) * self.scale


class MultiBasisBinaryLinear(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bases: int = 4,
    ):
        super().__init__()
        if bases < 1:
            raise ValueError("bases must be >= 1")
        self.weights = [_binary_weight((output_dim, input_dim)) for _ in range(bases)]
        self.thresholds = _make_thresholds(bases)
        self.scale = 1.0 / math.sqrt(input_dim * bases)

    def __call__(self, x: mx.array) -> mx.array:
        outputs = []
        for weight, threshold in zip(self.weights, self.thresholds, strict=True):
            binary_input = mx.where(x >= threshold, 1.0, -1.0)
            outputs.append(binary_input @ mx.transpose(weight))
        out = outputs[0]
        for value in outputs[1:]:
            out = out + value
        return out * self.scale


class OneBitPostGateBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        intermediate_dim: int,
        binary_bases: int,
    ):
        super().__init__()
        if binary_bases == 1:
            self.up_proj = BinaryLinear(input_dim, intermediate_dim)
            self.down_proj = BinaryLinear(intermediate_dim, hidden_dim)
        else:
            self.up_proj = MultiBasisBinaryLinear(input_dim, intermediate_dim, binary_bases)
            self.down_proj = MultiBasisBinaryLinear(intermediate_dim, hidden_dim, binary_bases)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)

    def __call__(self, x: mx.array) -> mx.array:
        down = self.down_proj(self.up_proj(x))
        gate = self.gate_proj(down)
        return down * nn.silu(gate)


def _make_thresholds(bases: int) -> mx.array:
    if bases == 1:
        return mx.zeros((1,))
    return mx.linspace(-1.0, 1.0, bases)


class OneBitGateMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 256,
        depth: int = 3,
        gate_ratio: float = 2.0,
        binary_bases: int = 1,
        num_classes: int = 10,
    ):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.layers = []
        dim = input_dim
        gate_hidden = int(hidden_dim * gate_ratio)
        for _ in range(depth):
            self.layers.append(
                OneBitPostGateBlock(
                    input_dim=dim,
                    hidden_dim=hidden_dim,
                    intermediate_dim=gate_hidden,
                    binary_bases=binary_bases,
                )
            )
            dim = hidden_dim
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def is_binary_parameter(path: str) -> bool:
        return path.startswith("layers.") and (
            ".up_proj." in path or ".down_proj." in path
        ) and (path.endswith(".weight") or ".weights." in path)

    def __call__(self, x: mx.array) -> mx.array:
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x))


class AttentionOneBitMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        patch_size: int = 28,
        hidden_dim: int = 128,
        attention_depth: int = 2,
        mlp_depth: int = 3,
        num_heads: int = 4,
        gate_ratio: float = 2.0,
        binary_bases: int = 4,
        num_classes: int = 10,
    ):
        super().__init__()
        if input_dim % patch_size != 0:
            raise ValueError("input_dim must be divisible by patch_size")

        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, hidden_dim)
        self.pos_embed = mx.zeros((1, self.num_patches, hidden_dim))
        self.blocks = [AttentionBlock(hidden_dim, num_heads) for _ in range(attention_depth)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.onebit_mlp = OneBitGateMLP(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            depth=mlp_depth,
            gate_ratio=gate_ratio,
            binary_bases=binary_bases,
            num_classes=num_classes,
        )

    @staticmethod
    def is_binary_parameter(path: str) -> bool:
        return path.startswith("onebit_mlp.") and OneBitGateMLP.is_binary_parameter(
            path.removeprefix("onebit_mlp.")
        )

    def __call__(self, x: mx.array) -> mx.array:
        batch = x.shape[0]
        x = x.reshape(batch, self.num_patches, self.patch_size)
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = mx.mean(self.norm(x), axis=1)
        return self.onebit_mlp(x)


class AttentionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        x_norm = self.norm1(x)
        batch, tokens, channels = x_norm.shape
        qkv = self.qkv(x_norm).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = mx.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ mx.transpose(k, (0, 1, 3, 2))) / math.sqrt(self.head_dim)
        attn = mx.softmax(attn, axis=-1)
        y = attn @ v
        y = mx.transpose(y, (0, 2, 1, 3)).reshape(batch, tokens, channels)
        x = residual + self.proj(y)
        return x + self.mlp(self.norm2(x))


class AttentionMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 784,
        patch_size: int = 7,
        hidden_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        num_classes: int = 10,
    ):
        super().__init__()
        if input_dim % patch_size != 0:
            raise ValueError("input_dim must be divisible by patch_size")

        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        self.patch_embed = nn.Linear(patch_size, hidden_dim)
        self.pos_embed = mx.zeros((1, self.num_patches, hidden_dim))
        self.blocks = [AttentionBlock(hidden_dim, num_heads) for _ in range(depth)]
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        batch = x.shape[0]
        x = x.reshape(batch, self.num_patches, self.patch_size)
        x = self.patch_embed(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = mx.mean(x, axis=1)
        return self.head(x)


def build_model(
    name: str,
    *,
    input_dim: int,
    hidden_dim: int,
    depth: int,
    num_heads: int,
    patch_size: int,
    gate_ratio: float = 2.0,
    binary_bases: int = 1,
) -> nn.Module:
    if name == "mlp":
        return MLPOnly(input_dim=input_dim, hidden_dim=hidden_dim, depth=depth)
    if name == "onebit_gate_mlp":
        return OneBitGateMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            gate_ratio=gate_ratio,
            binary_bases=binary_bases,
        )
    if name == "attention_onebit_mlp":
        return AttentionOneBitMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            attention_depth=depth,
            mlp_depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            gate_ratio=gate_ratio,
            binary_bases=binary_bases,
        )
    if name == "attention_mlp":
        return AttentionMLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
        )
    raise ValueError(f"unknown model: {name}")
