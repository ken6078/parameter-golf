"""
MLX-native PocketLLM-style state_dict compression utilities.

This mirrors the high-level abstractions in `compress_gpt.py` and keeps the
public interface used by `train_gpt_mlx.py`:
- compress_state_dict_pocketllm(state_dict)
- decompress_state_dict_pocketllm(obj)
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass

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


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _numpy_dtype_name(arr: mx.array) -> str:
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


def _dtype_from_name(name: str):
    try:
        return getattr(mx, name)
    except AttributeError:
        return np.dtype(name)


def _as_numpy(arr: mx.array, dtype=None) -> np.ndarray:
    if arr.dtype == mx.bfloat16:
        np_arr = np.array(arr.astype(mx.float32), copy=False)
    else:
        np_arr = np.array(arr, copy=False)
    if dtype is not None:
        np_arr = np_arr.astype(dtype, copy=False)
    if not np_arr.flags.c_contiguous:
        np_arr = np.ascontiguousarray(np_arr)
    return np_arr


def tensor_nbytes(arr: mx.array | np.ndarray) -> int:
    if isinstance(arr, mx.array):
        return int(_as_numpy(arr).nbytes)
    return int(np.asarray(arr).nbytes)


@dataclass(frozen=True)
class WeightSplitMeta:
    n_rows: int
    row_width: int
    subv_dim: int
    subv_per_row: int


@dataclass
class PocketCompressorConfig:
    mode: str = os.environ.get("POCKET_MODE", "post_train_compress")
    subvector_dim: int = int(os.environ.get("POCKET_SUBVECTOR_DIM", 8))
    latent_dim: int = int(os.environ.get("POCKET_LATENT_DIM", 0))
    codebook_size: int = int(os.environ.get("POCKET_CODEBOOK_SIZE", 512))
    codebook_init: str = os.environ.get("POCKET_CODEBOOK_INIT", "normal")
    codebook_init_std: float = float(os.environ.get("POCKET_CODEBOOK_INIT_STD", 0.02))
    encoder_layers: int = int(os.environ.get("POCKET_ENCODER_LAYERS", 3))
    decoder_layers: int = int(os.environ.get("POCKET_DECODER_LAYERS", 3))
    meta_hidden_dim: int = int(os.environ.get("POCKET_META_HIDDEN_DIM", 64))
    activation: str = os.environ.get("POCKET_ACTIVATION", "gelu")
    residual_mode: str = os.environ.get("POCKET_RESIDUAL_MODE", "after_first")
    norm_type: str = os.environ.get("POCKET_NORM_TYPE", "rln")
    rln_eps: float = float(os.environ.get("POCKET_RLN_EPS", 1e-6))
    train_steps: int = int(os.environ.get("POCKET_TRAIN_STEPS", 80))
    batch_rows: int = int(os.environ.get("POCKET_BATCH_ROWS", 1024))
    chunk_rows: int = int(os.environ.get("POCKET_CHUNK_ROWS", 1024))
    lr: float = float(os.environ.get("POCKET_LR", 3e-3))
    lambda_vq: float = float(os.environ.get("POCKET_VQ_LAMBDA", 0.25))
    loss_reduction: str = os.environ.get("POCKET_LOSS_REDUCTION", "mean_rmse")
    rmse_eps: float = float(os.environ.get("POCKET_RMSE_EPS", 1e-12))
    keep_float_max_numel: int = int(os.environ.get("POCKET_KEEP_FLOAT_MAX_NUMEL", 65_536))
    compress_min_2d_numel: int = int(os.environ.get("POCKET_COMPRESS_MIN_2D_NUMEL", 131_072))
    keep_float_store_dtype = mx.float16
    compress_attention_only: bool = _env_flag("POCKET_COMPRESS_ATTENTION_ONLY", False)
    compress_ffn_only: bool = _env_flag("POCKET_COMPRESS_FFN_ONLY", False)
    compress_all_linear: bool = _env_flag("POCKET_COMPRESS_ALL_LINEAR", False)
    target_regex: str = os.environ.get("POCKET_TARGET_REGEX", "")
    periodic_refresh_every: int = int(os.environ.get("POCKET_PERIODIC_REFRESH_EVERY", 100))
    enable_lora_finetune: bool = _env_flag("POCKET_ENABLE_LORA_FINETUNE", False)
    lora_rank: int = int(os.environ.get("POCKET_LORA_RANK", 32))
    lora_alpha: int = int(os.environ.get("POCKET_LORA_ALPHA", 64))
    lora_batch_size: int = int(os.environ.get("POCKET_LORA_BATCH_SIZE", 16))
    lora_epochs: int = int(os.environ.get("POCKET_LORA_EPOCHS", 3))
    lora_lr: float = float(os.environ.get("POCKET_LORA_LR", 1e-4))
    preset: str = os.environ.get("POCKET_PRESET", "")
    run_self_tests: bool = _env_flag("POCKET_RUN_SELF_TESTS", False)

    @classmethod
    def from_env(cls) -> "PocketCompressorConfig":
        cfg = cls()
        cfg.apply_preset()
        if cfg.latent_dim <= 0:
            cfg.latent_dim = cfg.subvector_dim
        return cfg

    def apply_preset(self) -> None:
        presets = {
            "8x": (4, 2**15),
            "10x": (4, 2**12),
            "16x": (8, 2**15),
            "20x": (8, 2**12),
        }
        if self.preset in presets:
            d, k = presets[self.preset]
            if "POCKET_SUBVECTOR_DIM" not in os.environ:
                self.subvector_dim = d
            if "POCKET_CODEBOOK_SIZE" not in os.environ:
                self.codebook_size = k


@dataclass
class PocketCompressedLayerResult:
    name: str
    payload: dict[str, object]
    reconstructed: mx.array
    compressed_bytes: int
    rmse: float
    vq_mse_sum: float
    mse_top100: float


def split_weight_matrix(w: mx.array, subvector_dim: int) -> tuple[mx.array, WeightSplitMeta]:
    n_rows, row_width = w.shape
    if row_width % subvector_dim != 0:
        raise ValueError(
            f"Row width {row_width} is not divisible by subvector_dim {subvector_dim}; "
            "set POCKET_SUBVECTOR_DIM accordingly."
        )
    w_np = _as_numpy(w, np.float32)
    subv_per_row = row_width // subvector_dim
    s = w_np.reshape(-1, subvector_dim)
    meta = WeightSplitMeta(n_rows=n_rows, row_width=row_width, subv_dim=subvector_dim, subv_per_row=subv_per_row)
    return mx.array(s), meta


def merge_weight_matrix(s_hat: mx.array, meta: WeightSplitMeta, dtype) -> mx.array:
    s_np = _as_numpy(s_hat, np.float32)
    out = s_np.reshape(meta.n_rows, meta.row_width)
    return mx.array(out).astype(dtype)


class ReshapedLayerNorm(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def __call__(self, x: mx.array, meta: WeightSplitMeta) -> mx.array:
        rows = mx.reshape(x, (meta.n_rows, meta.row_width))
        mean = mx.mean(rows, axis=1, keepdims=True)
        var = mx.mean((rows - mean) * (rows - mean), axis=1, keepdims=True)
        out = (rows - mean) / mx.sqrt(var + self.eps)
        return mx.reshape(out, x.shape)


class PocketMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str,
        residual_mode: str,
        norm_type: str,
        rln_eps: float,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("encoder/decoder layers must be >= 2")
        self.activation = activation.lower()
        self.residual_mode = residual_mode
        self.norm_type = norm_type
        self.rln = ReshapedLayerNorm(rln_eps)
        dims = [in_dim]
        dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(out_dim)
        self.layers = [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]

    def _act(self, x: mx.array) -> mx.array:
        if self.activation == "gelu":
            return nn.gelu(x)
        if self.activation == "relu":
            return nn.relu(x)
        if self.activation == "silu":
            return nn.silu(x)
        raise ValueError(f"Unsupported activation={self.activation}")

    def _norm(self, x: mx.array, meta: WeightSplitMeta) -> mx.array:
        if self.norm_type == "none":
            return x
        if self.norm_type == "ln_subvector":
            mean = mx.mean(x, axis=1, keepdims=True)
            var = mx.mean((x - mean) * (x - mean), axis=1, keepdims=True)
            return (x - mean) / mx.sqrt(var + 1e-6)
        if self.norm_type == "rln":
            return self.rln(x, meta)
        raise ValueError(f"Unsupported norm_type={self.norm_type}")

    def __call__(self, x: mx.array, meta: WeightSplitMeta) -> mx.array:
        h = x
        last_idx = len(self.layers) - 1
        for idx, layer in enumerate(self.layers):
            h_norm = self._norm(h, meta)
            y = layer(h_norm)
            if idx != last_idx:
                y = self._act(y)
            use_residual = False
            if idx != last_idx and y.shape == h.shape:
                if self.residual_mode == "after_first":
                    use_residual = idx >= 1
                elif self.residual_mode == "except_last":
                    use_residual = True
                else:
                    raise ValueError(f"Unsupported residual_mode={self.residual_mode}")
            h = h + y if use_residual else y
        return h

class VectorQuantizerSTE(nn.Module):
    def __init__(self, latent_dim: int, codebook_size: int, init_mode: str, init_std: float):
        super().__init__()
        if init_mode == "normal":
            self.codebook = mx.random.normal((codebook_size, latent_dim), dtype=mx.float32) * init_std
        elif init_mode == "uniform":
            bound = max(init_std, 1e-6)
            self.codebook = (mx.random.uniform((codebook_size, latent_dim), dtype=mx.float32) * 2.0 - 1.0) * bound
        else:
            raise ValueError(f"Unsupported codebook_init={init_mode}")

    @staticmethod
    def nearest_indices(z_flat: mx.array, codebook: mx.array, chunk: int = 16384) -> mx.array:
        idx_parts: list[mx.array] = []
        c = codebook.astype(mx.float32)
        c_norm = mx.sum(c * c, axis=1)[None, :]
        for start in range(0, z_flat.shape[0], chunk):
            z = z_flat[start : start + chunk].astype(mx.float32)
            z_norm = mx.sum(z * z, axis=1, keepdims=True)
            dist = z_norm + c_norm - 2.0 * (z @ c.T)
            idx_parts.append(mx.argmin(dist, axis=1))
        return mx.concatenate(idx_parts, axis=0)

    def __call__(self, z: mx.array, chunk: int = 16384) -> tuple[mx.array, mx.array, mx.array]:
        idx = self.nearest_indices(z, self.codebook, chunk=chunk)
        z_q = self.codebook[idx]
        z_q_st = z + mx.stop_gradient(z_q - z)
        vq_mse_sum = mx.sum((z - mx.stop_gradient(z_q)) * (z - mx.stop_gradient(z_q)))
        return z_q_st, idx, vq_mse_sum


class PocketCodec(nn.Module):
    def __init__(self, cfg: PocketCompressorConfig):
        super().__init__()
        self.encoder = PocketMLP(
            in_dim=cfg.subvector_dim,
            out_dim=cfg.latent_dim,
            hidden_dim=cfg.meta_hidden_dim,
            num_layers=cfg.encoder_layers,
            activation=cfg.activation,
            residual_mode=cfg.residual_mode,
            norm_type=cfg.norm_type,
            rln_eps=cfg.rln_eps,
        )
        self.decoder = PocketMLP(
            in_dim=cfg.latent_dim,
            out_dim=cfg.subvector_dim,
            hidden_dim=cfg.meta_hidden_dim,
            num_layers=cfg.decoder_layers,
            activation=cfg.activation,
            residual_mode=cfg.residual_mode,
            norm_type=cfg.norm_type,
            rln_eps=cfg.rln_eps,
        )
        self.quant = VectorQuantizerSTE(
            latent_dim=cfg.latent_dim,
            codebook_size=cfg.codebook_size,
            init_mode=cfg.codebook_init,
            init_std=cfg.codebook_init_std,
        )


class PocketLayerCompressor:
    def __init__(self, cfg: PocketCompressorConfig):
        self.cfg = cfg

    def _decoder_state_numpy(self, decoder: PocketMLP) -> dict[str, np.ndarray]:
        return {name: _as_numpy(value, np.float16) for name, value in tree_flatten(decoder.parameters())}

    def _load_decoder_state(self, decoder: PocketMLP, state: dict[str, np.ndarray]) -> None:
        flat = [(name, mx.array(value.astype(np.float32, copy=False))) for name, value in state.items()]
        decoder.update(tree_unflatten(flat))

    def compress(self, name: str, arr: mx.array) -> PocketCompressedLayerResult:
        s, meta = split_weight_matrix(arr.astype(mx.float32), self.cfg.subvector_dim)
        n_rows = int(meta.n_rows)
        s_rows = mx.reshape(s, (n_rows, meta.subv_per_row, meta.subv_dim))

        codec = PocketCodec(self.cfg)
        optimizer = optim.Adam(learning_rate=self.cfg.lr)

        def loss_fn(model: PocketCodec, batch_rows_arr: mx.array, batch_meta: WeightSplitMeta) -> mx.array:
            s_batch = mx.reshape(batch_rows_arr, (-1, batch_meta.subv_dim))
            z = model.encoder(s_batch, batch_meta)
            z_q_st, _, vq_mse_sum = model.quant(z, chunk=max(1, self.cfg.chunk_rows * batch_meta.subv_per_row))
            s_hat = model.decoder(z_q_st, batch_meta)
            sq = (s_batch - s_hat) * (s_batch - s_hat)
            if self.cfg.loss_reduction == "paper_sum":
                rmse = mx.sqrt(mx.sum(sq) + self.cfg.rmse_eps)
            else:
                rmse = mx.sqrt(mx.mean(sq) + self.cfg.rmse_eps)
            return rmse + self.cfg.lambda_vq * vq_mse_sum

        loss_and_grad = nn.value_and_grad(codec, loss_fn)
        for _ in range(max(1, self.cfg.train_steps)):
            batch_rows = min(max(1, self.cfg.batch_rows), n_rows)
            if batch_rows >= n_rows:
                s_batch_rows = s_rows
            else:
                row_idx = mx.random.randint(0, n_rows, (batch_rows,))
                s_batch_rows = s_rows[row_idx]
            batch_meta = WeightSplitMeta(
                n_rows=batch_rows,
                row_width=meta.row_width,
                subv_dim=meta.subv_dim,
                subv_per_row=meta.subv_per_row,
            )
            loss, grads = loss_and_grad(codec, s_batch_rows, batch_meta)
            params = dict(tree_flatten(codec.parameters()))
            grad_flat = dict(tree_flatten(grads))
            updated = optimizer.apply_gradients(grad_flat, params)
            codec.update(tree_unflatten(list(updated.items())))
            mx.eval(loss, codec.state)

        z_all = codec.encoder(s, meta)
        idx = VectorQuantizerSTE.nearest_indices(z_all, codec.quant.codebook, chunk=max(1, self.cfg.chunk_rows * meta.subv_per_row))
        z_q = codec.quant.codebook[idx]
        s_hat = codec.decoder(z_q, meta)
        w_hat = merge_weight_matrix(s_hat, meta, arr.dtype)

        sqerr = mx.reshape((s - s_hat) * (s - s_hat), (-1,))
        sqerr_np = _as_numpy(sqerr, np.float32)
        topk = min(100, sqerr_np.size)
        mse_top100 = float(np.partition(sqerr_np, -topk)[-topk:].mean()) if topk > 0 else 0.0
        rmse = float(np.sqrt(float(sqerr_np.mean()) + self.cfg.rmse_eps)) if sqerr_np.size else 0.0
        vq_mse_sum = float(_as_numpy(mx.sum((z_all - z_q) * (z_all - z_q))).item())

        codebook_np = _as_numpy(codec.quant.codebook, np.float16)
        idx_np = _as_numpy(idx)
        if codebook_np.shape[0] <= 256:
            idx_store = idx_np.astype(np.uint8, copy=False)
        elif codebook_np.shape[0] <= 65_536:
            idx_store = idx_np.astype(np.uint16, copy=False)
        else:
            idx_store = idx_np.astype(np.int32, copy=False)
        decoder_state = self._decoder_state_numpy(codec.decoder)

        payload = {
            "codebook": codebook_np,
            "indices": idx_store,
            "decoder_state": decoder_state,
            "meta": asdict(meta),
            "latent_dim": self.cfg.latent_dim,
            "meta_hidden_dim": self.cfg.meta_hidden_dim,
            "decoder_layers": self.cfg.decoder_layers,
            "activation": self.cfg.activation,
            "residual_mode": self.cfg.residual_mode,
            "norm_type": self.cfg.norm_type,
            "rln_eps": self.cfg.rln_eps,
            "compressed_dtype": _numpy_dtype_name(arr),
        }

        compressed_bytes = tensor_nbytes(codebook_np) + tensor_nbytes(idx_store)
        for v in decoder_state.values():
            compressed_bytes += tensor_nbytes(v)

        return PocketCompressedLayerResult(
            name=name,
            payload=payload,
            reconstructed=w_hat,
            compressed_bytes=compressed_bytes,
            rmse=rmse,
            vq_mse_sum=vq_mse_sum,
            mse_top100=mse_top100,
        )

    def reconstruct(self, payload: dict[str, object], dtype=None) -> mx.array:
        meta = WeightSplitMeta(**payload["meta"])
        codebook = mx.array(payload["codebook"].astype(np.float32, copy=False))
        indices = mx.array(payload["indices"].astype(np.int32, copy=False)).reshape((-1,))
        z_q = codebook[indices]

        decoder = PocketMLP(
            in_dim=int(payload["latent_dim"]),
            out_dim=meta.subv_dim,
            hidden_dim=int(payload["meta_hidden_dim"]),
            num_layers=int(payload["decoder_layers"]),
            activation=str(payload.get("activation", "gelu")),
            residual_mode=str(payload.get("residual_mode", "after_first")),
            norm_type=str(payload.get("norm_type", "rln")),
            rln_eps=float(payload.get("rln_eps", 1e-6)),
        )
        self._load_decoder_state(decoder, payload["decoder_state"])
        s_hat = decoder(z_q, meta)
        mx.eval(s_hat)
        out_dtype = dtype if dtype is not None else _dtype_from_name(str(payload.get("compressed_dtype", "float32")))
        return merge_weight_matrix(s_hat, meta, out_dtype)

class PocketModelCompressor:
    ATTN_NAME_HINTS = (
        "q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo", "qkv", "attn.q", "attn.k", "attn.v", "attn.o"
    )
    FFN_NAME_HINTS = (
        "up_proj", "gate_proj", "down_proj", "mlp.up", "mlp.gate", "mlp.down", "fc", "proj", "ffn"
    )

    def __init__(self, cfg: PocketCompressorConfig):
        self.cfg = cfg
        self.layer = PocketLayerCompressor(cfg)

    def _matches_target(self, name: str) -> bool:
        lname = name.lower()
        if self.cfg.target_regex:
            return re.search(self.cfg.target_regex, name) is not None
        if self.cfg.compress_all_linear:
            return True
        attn_hit = any(tok in lname for tok in self.ATTN_NAME_HINTS)
        ffn_hit = any(tok in lname for tok in self.FFN_NAME_HINTS)
        if self.cfg.compress_attention_only and self.cfg.compress_ffn_only:
            return attn_hit or ffn_hit
        if self.cfg.compress_attention_only:
            return attn_hit
        if self.cfg.compress_ffn_only:
            return ffn_hit
        return attn_hit or ffn_hit

    def _should_passthrough(self, name: str, t: mx.array, t_np: np.ndarray) -> bool:
        return (
            (not np.issubdtype(t_np.dtype, np.floating))
            or (int(t.size) <= self.cfg.keep_float_max_numel)
            or (t.ndim != 2)
            or (int(t.size) < self.cfg.compress_min_2d_numel)
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
            or (not self._matches_target(name))
        )

    def _can_refresh_now(self, step: int | None = None) -> bool:
        if self.cfg.mode != "periodic_refresh":
            return True
        if step is None:
            return False
        every = max(1, self.cfg.periodic_refresh_every)
        return step % every == 0

    def compress_state_dict(self, state_dict: dict[str, mx.array], step: int | None = None):
        compressed: dict[str, dict[str, object]] = {}
        dtypes: dict[str, str] = {}
        passthrough: dict[str, np.ndarray] = {}
        passthrough_orig_dtypes: dict[str, str] = {}
        layer_logs: dict[str, dict[str, float]] = {}

        stats = dict.fromkeys(
            (
                "param_count",
                "num_tensors",
                "num_compressed_tensors",
                "num_nonfloat_tensors",
                "baseline_tensor_bytes",
                "int8_payload_bytes",
                "codebook_bytes",
                "indices_bytes",
                "decoder_bytes",
            ),
            0,
        )

        can_refresh = self._can_refresh_now(step)
        for name, t in state_dict.items():
            t_np = _as_numpy(t)
            stats["param_count"] += int(t.size)
            stats["num_tensors"] += 1
            stats["baseline_tensor_bytes"] += tensor_nbytes(t)

            should_passthrough = self._should_passthrough(name, t, t_np) or (self.cfg.mode == "periodic_refresh" and not can_refresh)
            if should_passthrough:
                if not np.issubdtype(t_np.dtype, np.floating):
                    stats["num_nonfloat_tensors"] += 1
                    kept = t_np
                elif any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
                    kept = _as_numpy(t.astype(mx.float32))
                else:
                    if t.dtype in {mx.float32, mx.bfloat16}:
                        passthrough_orig_dtypes[name] = _numpy_dtype_name(t)
                        kept = _as_numpy(t.astype(self.cfg.keep_float_store_dtype))
                    else:
                        kept = t_np
                passthrough[name] = kept
                stats["int8_payload_bytes"] += tensor_nbytes(kept)
                continue

            try:
                res = self.layer.compress(name, t)
            except Exception:
                kept = _as_numpy(t.astype(self.cfg.keep_float_store_dtype))
                passthrough[name] = kept
                stats["int8_payload_bytes"] += tensor_nbytes(kept)
                continue

            compressed[name] = res.payload
            dtypes[name] = _numpy_dtype_name(t)
            stats["num_compressed_tensors"] += 1
            stats["int8_payload_bytes"] += res.compressed_bytes
            stats["codebook_bytes"] += tensor_nbytes(res.payload["codebook"])
            stats["indices_bytes"] += tensor_nbytes(res.payload["indices"])
            dec_bytes = sum(tensor_nbytes(v) for v in res.payload["decoder_state"].values())
            stats["decoder_bytes"] += dec_bytes
            layer_logs[name] = {
                "rmse": res.rmse,
                "vq_mse_sum": res.vq_mse_sum,
                "mse_top100": res.mse_top100,
                "subvector_dim": float(self.cfg.subvector_dim),
                "codebook_size": float(res.payload["codebook"].shape[0]),
                "encoder_layers": float(self.cfg.encoder_layers),
                "decoder_layers": float(self.cfg.decoder_layers),
            }

        obj: dict[str, object] = {
            "__quant_format__": "pocketllm_meta_mlx_v2",
            "quantized": compressed,
            "dtypes": dtypes,
            "passthrough": passthrough,
            "config": asdict(self.cfg),
            "layer_logs": layer_logs,
            "mode": self.cfg.mode,
            "lora_finetune": {
                "enabled": self.cfg.enable_lora_finetune,
                "rank": self.cfg.lora_rank,
                "alpha": self.cfg.lora_alpha,
                "batch_size": self.cfg.lora_batch_size,
                "epochs": self.cfg.lora_epochs,
                "lr": self.cfg.lora_lr,
            },
        }
        if passthrough_orig_dtypes:
            obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
        return obj, stats

    def decompress_state_dict(self, obj: dict[str, object]) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        quantized = obj.get("quantized", {})
        dtypes = obj.get("dtypes", {})
        passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})

        for name, payload in quantized.items():
            dtype_name = dtypes.get(name)
            dtype = _dtype_from_name(dtype_name) if isinstance(dtype_name, str) else None
            out[name] = self.layer.reconstruct(payload, dtype=dtype)

        for name, value in obj.get("passthrough", {}).items():
            arr = mx.array(value)
            orig_dtype = passthrough_orig_dtypes.get(name)
            if isinstance(orig_dtype, str):
                arr = arr.astype(_dtype_from_name(orig_dtype))
            out[name] = arr
        return out


def pocket_self_check_split_merge() -> None:
    x = mx.array(np.random.randn(4, 16).astype(np.float32))
    s, meta = split_weight_matrix(x, 4)
    y = merge_weight_matrix(s, meta, x.dtype)
    mx.eval(y)
    assert np.array_equal(_as_numpy(x), _as_numpy(y))


def pocket_self_check_vq_shapes() -> None:
    z = mx.array(np.random.randn(128, 4).astype(np.float32))
    vq = VectorQuantizerSTE(latent_dim=4, codebook_size=64, init_mode="normal", init_std=0.02)
    zq, idx, loss = vq(z)
    mx.eval(zq, idx, loss)
    assert zq.shape == z.shape
    assert idx.shape == (128,)


def pocket_self_check_reconstruction_shapes() -> None:
    cfg = PocketCompressorConfig.from_env()
    cfg.subvector_dim = 4
    cfg.latent_dim = 4
    cfg.codebook_size = 64
    cfg.train_steps = 2
    cfg.batch_rows = 8
    cfg.compress_min_2d_numel = 0
    comp = PocketLayerCompressor(cfg)
    w = mx.array(np.random.randn(16, 16).astype(np.float32))
    res = comp.compress("self_check.weight", w)
    rec = comp.reconstruct(res.payload, dtype=mx.float32)
    mx.eval(rec)
    assert rec.shape == w.shape


def pocket_self_check_artifact_roundtrip() -> None:
    cfg = PocketCompressorConfig.from_env()
    cfg.subvector_dim = 4
    cfg.latent_dim = 4
    cfg.codebook_size = 64
    cfg.train_steps = 2
    cfg.batch_rows = 8
    cfg.compress_min_2d_numel = 0
    model_comp = PocketModelCompressor(cfg)
    state = {"blk.q_proj.weight": mx.array(np.random.randn(16, 16).astype(np.float32))}
    obj, _ = model_comp.compress_state_dict(state)
    rec = model_comp.decompress_state_dict(obj)
    assert rec["blk.q_proj.weight"].shape == state["blk.q_proj.weight"].shape


def compress_state_dict_pocketllm(state_dict: dict[str, mx.array], step: int | None = None):
    cfg = PocketCompressorConfig.from_env()
    if cfg.run_self_tests:
        pocket_self_check_split_merge()
        pocket_self_check_vq_shapes()
        pocket_self_check_reconstruction_shapes()
        pocket_self_check_artifact_roundtrip()

    comp = PocketModelCompressor(cfg)
    obj, stats = comp.compress_state_dict(state_dict, step=step)

    print(
        "[PocketLLM-MLX] mode={mode} target_regex={regex} attn_only={attn} ffn_only={ffn} all_linear={all_linear}".format(
            mode=cfg.mode,
            regex=cfg.target_regex or "<none>",
            attn=cfg.compress_attention_only,
            ffn=cfg.compress_ffn_only,
            all_linear=cfg.compress_all_linear,
        )
    )
    print(
        "[PocketLLM-MLX] d={d} k={k} enc_layers={enc} dec_layers={dec} residual_mode={res} norm_type={norm}".format(
            d=cfg.subvector_dim,
            k=cfg.codebook_size,
            enc=cfg.encoder_layers,
            dec=cfg.decoder_layers,
            res=cfg.residual_mode,
            norm=cfg.norm_type,
        )
    )
    logs = obj.get("layer_logs", {})
    for name, m in logs.items():
        print(
            f"[PocketLLM-MLX][layer] {name} rmse={m['rmse']:.6e} vq_mse_sum={m['vq_mse_sum']:.6e} "
            f"mse_top100={m['mse_top100']:.6e}"
        )
    print(
        "[PocketLLM-MLX][bytes] codebook={cb} indices={idx} decoder={dec} total={tot}".format(
            cb=stats["codebook_bytes"],
            idx=stats["indices_bytes"],
            dec=stats["decoder_bytes"],
            tot=stats["int8_payload_bytes"],
        )
    )
    return obj, stats


def decompress_state_dict_pocketllm(obj: dict[str, object]) -> dict[str, mx.array]:
    cfg = PocketCompressorConfig.from_env()
    comp = PocketModelCompressor(cfg)
    return comp.decompress_state_dict(obj)
