"""
MLX-native PocketLLM-style state_dict compression utilities.

This mirrors the public interface from `compress_gpt.py`, but operates on MLX
arrays directly so `train_gpt_mlx.py` can compress/decompress without routing
through Torch tensors.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights",
    ).split(",")
    if pattern
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    pattern
    for pattern in os.environ.get(
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS",
        ",".join(CONTROL_TENSOR_NAME_PATTERNS),
    ).split(",")
    if pattern
)

POCKET_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("POCKET_KEEP_FLOAT_MAX_NUMEL", 65_536))
POCKET_KEEP_FLOAT_STORE_DTYPE = mx.float16
POCKET_SUBVECTOR_DIM = int(os.environ.get("POCKET_SUBVECTOR_DIM", 8))
POCKET_CODEBOOK_SIZE = int(os.environ.get("POCKET_CODEBOOK_SIZE", 512))
POCKET_META_HIDDEN_DIM = int(os.environ.get("POCKET_META_HIDDEN_DIM", 32))
POCKET_META_NUM_LAYERS = int(os.environ.get("POCKET_META_NUM_LAYERS", 3))
POCKET_TRAIN_STEPS = int(os.environ.get("POCKET_TRAIN_STEPS", 80))
POCKET_BATCH_ROWS = int(os.environ.get("POCKET_BATCH_ROWS", 1024))
POCKET_CHUNK_ROWS = int(os.environ.get("POCKET_CHUNK_ROWS", 1024))
POCKET_LR = float(os.environ.get("POCKET_LR", 3e-3))
POCKET_VQ_LAMBDA = float(os.environ.get("POCKET_VQ_LAMBDA", 0.25))
POCKET_RLN_EPS = float(os.environ.get("POCKET_RLN_EPS", 1e-6))
POCKET_COMPRESS_MIN_2D_NUMEL = int(os.environ.get("POCKET_COMPRESS_MIN_2D_NUMEL", 131_072))


def _numpy_dtype_name(arr: mx.array) -> str:
    if isinstance(arr, mx.array):
        dtype_names = {
            mx.bool_: "bool",
            mx.uint8: "uint8",
            mx.int8: "int8",
            mx.uint16: "uint16",
            mx.int16: "int16",
            mx.float16: "float16",
            mx.bfloat16: "bfloat16",
            mx.uint32: "uint32",
            mx.int32: "int32",
            mx.float32: "float32",
            mx.uint64: "uint64",
            mx.int64: "int64",
            mx.float64: "float64",
        }
        name = dtype_names.get(arr.dtype)
        if name is not None:
            return name
        return str(arr.dtype).removeprefix("mlx.core.")
    return np.asarray(arr).dtype.name


def _dtype_from_name(name: str):
    try:
        return getattr(mx, name)
    except AttributeError:
        return np.dtype(name)


def _as_numpy(arr: mx.array, dtype=None) -> np.ndarray:
    if isinstance(arr, mx.array):
        # MLX bfloat16 does not always expose a NumPy-compatible buffer cleanly.
        # Materialize through float32 unless the caller requested another dtype.
        if arr.dtype == mx.bfloat16:
            target_dtype = np.float32 if dtype is None else dtype
            np_arr = np.array(arr.astype(mx.float32), dtype=target_dtype, copy=False)
        else:
            np_arr = np.array(arr, copy=False)
    else:
        np_arr = np.asarray(arr)
    if dtype is not None:
        np_arr = np_arr.astype(dtype, copy=False)
    if not np_arr.flags.c_contiguous:
        np_arr = np.ascontiguousarray(np_arr)
    return np_arr


def tensor_nbytes(arr: mx.array | np.ndarray) -> int:
    if isinstance(arr, mx.array):
        dtype_sizes = {
            mx.bool_: 1,
            mx.uint8: 1,
            mx.int8: 1,
            mx.uint16: 2,
            mx.int16: 2,
            mx.float16: 2,
            mx.bfloat16: 2,
            mx.uint32: 4,
            mx.int32: 4,
            mx.float32: 4,
            mx.uint64: 8,
            mx.int64: 8,
            mx.float64: 8,
        }
        itemsize = dtype_sizes.get(arr.dtype)
        if itemsize is None:
            return int(_as_numpy(arr).nbytes)
        return int(arr.size) * itemsize
    return int(np.asarray(arr).nbytes)


def keep_float_tensor(name: str, arr: mx.array, passthrough_orig_dtypes: dict[str, str]) -> mx.array:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return arr.astype(mx.float32)
    if arr.dtype in {mx.float32, mx.bfloat16}:
        passthrough_orig_dtypes[name] = _numpy_dtype_name(arr)
        return arr.astype(POCKET_KEEP_FLOAT_STORE_DTYPE)
    return arr


@dataclass(frozen=True)
class SplitSpec:
    n_rows: int
    row_width: int
    subv_dim: int
    subv_per_row: int
    pad_cols: int


def _split_weight_rows(w: mx.array, subv_dim: int) -> tuple[mx.array, SplitSpec]:
    n_rows, n_cols = w.shape
    pad_cols = (subv_dim - (n_cols % subv_dim)) % subv_dim
    w_np = _as_numpy(w, np.float32)
    if pad_cols:
        w_np = np.pad(w_np, ((0, 0), (0, pad_cols)))
    row_width = int(w_np.shape[1])
    subv_per_row = row_width // subv_dim
    s = mx.array(w_np.reshape(n_rows, subv_per_row, subv_dim))
    return s, SplitSpec(n_rows, row_width, subv_dim, subv_per_row, pad_cols)


def _merge_weight_rows(s_hat: mx.array, spec: SplitSpec, dtype) -> mx.array:
    out = mx.reshape(s_hat, (spec.n_rows, spec.row_width))
    if spec.pad_cols:
        out = out[:, : spec.row_width - spec.pad_cols]
    return out.astype(dtype)


class ReshapedLayerNorm(nn.Module):
    def __init__(self, eps: float = POCKET_RLN_EPS):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rows, subv_per_row, subv_dim = x.shape
        x2 = mx.reshape(x, (rows, subv_per_row * subv_dim))
        rms = mx.sqrt(mx.mean(x2 * x2, axis=1, keepdims=True) + self.eps)
        return mx.reshape(x2 / rms, x.shape)


class MetaMLP(nn.Module):
    def __init__(self, d: int, hidden: int, num_layers: int):
        super().__init__()
        if num_layers < 2:
            raise ValueError("POCKET_META_NUM_LAYERS must be >= 2")
        self.rln = ReshapedLayerNorm()
        self.in_proj = nn.Linear(d, hidden)
        self.blocks = [nn.Linear(hidden, hidden) for _ in range(max(0, num_layers - 2))]
        self.out_proj = nn.Linear(hidden, d)

    def __call__(self, x: mx.array) -> mx.array:
        h = nn.gelu(self.in_proj(self.rln(x)))
        for block in self.blocks:
            y = nn.gelu(block(self.rln(h)))
            h = h + y
        return self.out_proj(self.rln(h))


class PocketCodec(nn.Module):
    def __init__(self, d: int, hidden: int, num_layers: int, codebook_size: int, mean: float, std: float):
        super().__init__()
        self.encoder = MetaMLP(d=d, hidden=hidden, num_layers=num_layers)
        self.decoder = MetaMLP(d=d, hidden=hidden, num_layers=num_layers)
        self.codebook = mx.random.normal((codebook_size, d), dtype=mx.float32) * std + mean


def _nearest_codebook_indices(z_flat: mx.array, codebook: mx.array, chunk: int = 16384) -> mx.array:
    idx_parts: list[mx.array] = []
    c = codebook.astype(mx.float32)
    c_norm = mx.sum(c * c, axis=1)[None, :]
    for start in range(0, z_flat.shape[0], chunk):
        z = z_flat[start : start + chunk].astype(mx.float32)
        z_norm = mx.sum(z * z, axis=1, keepdims=True)
        dist = z_norm + c_norm - 2.0 * (z @ c.T)
        idx_parts.append(mx.argmin(dist, axis=1))
    return mx.concatenate(idx_parts, axis=0)


def _build_indices_array(indices: mx.array, codebook_size: int) -> np.ndarray:
    indices_np = _as_numpy(indices)
    if codebook_size <= 256:
        return indices_np.astype(np.uint8, copy=False)
    if codebook_size <= 65_536:
        return indices_np.astype(np.uint16, copy=False)
    return indices_np.astype(np.int32, copy=False)


def _decoder_state_numpy(decoder: MetaMLP) -> dict[str, np.ndarray]:
    return {name: _as_numpy(value, np.float16) for name, value in tree_flatten(decoder.parameters())}


def _load_decoder_state(decoder: MetaMLP, state: dict[str, np.ndarray]) -> None:
    flat = [(name, mx.array(value.astype(np.float32, copy=False))) for name, value in state.items()]
    decoder.update(tree_unflatten(flat))


def _compress_tensor_pocketllm(name: str, arr: mx.array) -> tuple[dict[str, object], int]:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor for PocketLLM compression, got shape={tuple(arr.shape)} at {name}")

    s, spec = _split_weight_rows(arr.astype(mx.float32), POCKET_SUBVECTOR_DIM)
    n_rows = int(s.shape[0])
    subv_per_row = int(s.shape[1])
    flat_count = n_rows * subv_per_row
    d = int(s.shape[2])
    k = min(POCKET_CODEBOOK_SIZE, max(16, flat_count))

    probe = s[: min(n_rows, 256)]
    probe_encoder = MetaMLP(d=d, hidden=POCKET_META_HIDDEN_DIM, num_layers=POCKET_META_NUM_LAYERS)
    z0 = probe_encoder(probe)
    mx.eval(z0)
    z0_np = _as_numpy(z0, np.float32)
    mean = float(z0_np.mean()) if z0_np.size else 0.0
    std = float(z0_np.std()) if z0_np.size else 1.0
    std = max(std, 1e-3)

    codec = PocketCodec(d, POCKET_META_HIDDEN_DIM, POCKET_META_NUM_LAYERS, k, mean, std)
    optimizer = optim.Adam(learning_rate=POCKET_LR)

    def loss_fn(model: PocketCodec, s_batch: mx.array) -> mx.array:
        z = model.encoder(s_batch)
        z_flat = mx.reshape(z, (-1, d))
        idx = _nearest_codebook_indices(z_flat, model.codebook)
        z_q = model.codebook[idx].reshape(z.shape)
        z_ste = z + mx.stop_gradient(z_q - z)
        s_hat = model.decoder(z_ste)
        rmse = mx.sqrt(mx.mean((s_batch - s_hat) * (s_batch - s_hat)) + 1e-12)
        vq = mx.mean((z - mx.stop_gradient(z_q)) * (z - mx.stop_gradient(z_q)))
        return rmse + POCKET_VQ_LAMBDA * vq

    loss_and_grad = nn.value_and_grad(codec, loss_fn)
    batch_rows = min(POCKET_BATCH_ROWS, n_rows)
    steps = max(1, POCKET_TRAIN_STEPS)
    for _ in range(steps):
        if batch_rows >= n_rows:
            s_batch = s
        else:
            row_idx = mx.random.randint(0, n_rows, (batch_rows,))
            s_batch = s[row_idx]
        loss, grads = loss_and_grad(codec, s_batch)
        params = dict(tree_flatten(codec.parameters()))
        grad_flat = dict(tree_flatten(grads))
        updated = optimizer.apply_gradients(grad_flat, params)
        codec.update(tree_unflatten(list(updated.items())))
        mx.eval(loss, codec.state)

    idx_chunks: list[mx.array] = []
    for start in range(0, n_rows, POCKET_CHUNK_ROWS):
        z_chunk = codec.encoder(s[start : start + POCKET_CHUNK_ROWS])
        z_chunk = mx.reshape(z_chunk, (-1, d))
        idx_chunks.append(_nearest_codebook_indices(z_chunk, codec.codebook))
    indices = mx.concatenate(idx_chunks, axis=0)
    mx.eval(indices, codec.state)

    stored_indices = _build_indices_array(indices, k)
    stored_codebook = _as_numpy(codec.codebook, np.float16)
    decoder_state = _decoder_state_numpy(codec.decoder)

    compressed_bytes = tensor_nbytes(stored_codebook) + tensor_nbytes(stored_indices)
    for value in decoder_state.values():
        compressed_bytes += tensor_nbytes(value)

    payload: dict[str, object] = {
        "codebook": stored_codebook,
        "indices": stored_indices,
        "decoder_state": decoder_state,
        "n_rows": spec.n_rows,
        "subv_per_row": spec.subv_per_row,
        "subv_dim": spec.subv_dim,
        "row_width": spec.row_width,
        "pad_cols": spec.pad_cols,
        "meta_hidden_dim": POCKET_META_HIDDEN_DIM,
        "meta_num_layers": POCKET_META_NUM_LAYERS,
    }
    return payload, compressed_bytes


def compress_state_dict_pocketllm(state_dict: dict[str, mx.array]):
    compressed: dict[str, dict[str, object]] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, np.ndarray] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    stats = dict.fromkeys(
        (
            "param_count",
            "num_tensors",
            "num_compressed_tensors",
            "num_nonfloat_tensors",
            "baseline_tensor_bytes",
            "int8_payload_bytes",
        ),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor
        stats["param_count"] += int(t.size)
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        t_np = _as_numpy(t)
        should_passthrough = (
            (not np.issubdtype(t_np.dtype, np.floating))
            or (int(t.size) <= POCKET_KEEP_FLOAT_MAX_NUMEL)
            or (t.ndim != 2)
            or (int(t.size) < POCKET_COMPRESS_MIN_2D_NUMEL)
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        )

        if should_passthrough:
            if not np.issubdtype(t_np.dtype, np.floating):
                stats["num_nonfloat_tensors"] += 1
                kept = t
            else:
                kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            kept_np = _as_numpy(kept)
            passthrough[name] = kept_np
            stats["int8_payload_bytes"] += tensor_nbytes(kept_np)
            continue

        payload, compressed_bytes = _compress_tensor_pocketllm(name, t)
        compressed[name] = payload
        dtypes[name] = _numpy_dtype_name(t)
        stats["num_compressed_tensors"] += 1
        stats["int8_payload_bytes"] += compressed_bytes

    obj: dict[str, object] = {
        "__quant_format__": "pocketllm_meta_mlx_v1",
        "quantized": compressed,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def decompress_state_dict_pocketllm(obj: dict[str, object]) -> dict[str, mx.array]:
    out: dict[str, mx.array] = {}
    passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
    quantized = obj.get("quantized", {})
    dtypes = obj.get("dtypes", {})

    for name, payload in quantized.items():
        n_rows = int(payload["n_rows"])
        subv_per_row = int(payload["subv_per_row"])
        subv_dim = int(payload["subv_dim"])
        row_width = int(payload["row_width"])
        pad_cols = int(payload["pad_cols"])
        hidden = int(payload["meta_hidden_dim"])
        layers = int(payload["meta_num_layers"])

        codebook = mx.array(payload["codebook"].astype(np.float32, copy=False))
        indices = mx.array(payload["indices"].astype(np.int32, copy=False)).reshape((-1,))
        z_q = codebook[indices].reshape((n_rows, subv_per_row, subv_dim))

        decoder = MetaMLP(d=subv_dim, hidden=hidden, num_layers=layers)
        _load_decoder_state(decoder, payload["decoder_state"])
        s_hat = decoder(z_q)
        mx.eval(s_hat)
        spec = SplitSpec(
            n_rows=n_rows,
            row_width=row_width,
            subv_dim=subv_dim,
            subv_per_row=subv_per_row,
            pad_cols=pad_cols,
        )
        out[name] = _merge_weight_rows(s_hat, spec, dtype=_dtype_from_name(dtypes[name]))

    for name, value in obj["passthrough"].items():
        out_arr = mx.array(value)
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_arr = out_arr.astype(_dtype_from_name(orig_dtype))
        out[name] = out_arr
    return out
