from __future__ import annotations

"""
PocketLLM-style compression utilities for Parameter Golf (PyTorch).

Usage notes (README-style quick commands):
- Baseline compression pipeline:
  `python compress_gpt.py --model final_model.pt --out final_model.compress.ptz`
- Post-train compression with paper presets:
  `POCKET_PRESET=8x python compress_gpt.py`
  `POCKET_PRESET=10x python compress_gpt.py`
  `POCKET_PRESET=16x python compress_gpt.py`
  `POCKET_PRESET=20x python compress_gpt.py`
- Compress only attention linear weights:
  `POCKET_COMPRESS_ATTENTION_ONLY=1 python compress_gpt.py`
- Compress only FFN linear weights:
  `POCKET_COMPRESS_FFN_ONLY=1 python compress_gpt.py`
- Optional post-compression LoRA finetune knobs (consumed by external train flow):
  `POCKET_ENABLE_LORA_FINETUNE=1 POCKET_LORA_RANK=32 POCKET_LORA_ALPHA=64`

Paper ambiguity notes:
- Residual placement wording differs across paper sections; choose via `POCKET_RESIDUAL_MODE`.
- Pseudocode mentions k-means; implementation uses learnable codebook + NN assignment + STE.
- Merge step uses reconstructed subvectors `S_hat` (not original `S`).
- Target compression modules are configurable by regex/name policy.
"""

import argparse
import io
import os
import re
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

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


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


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
    rln_affine: bool = _env_flag("POCKET_RLN_AFFINE", False)
    train_steps: int = int(os.environ.get("POCKET_TRAIN_STEPS", 80))
    batch_rows: int = int(os.environ.get("POCKET_BATCH_ROWS", 1024))
    chunk_rows: int = int(os.environ.get("POCKET_CHUNK_ROWS", 1024))
    lr: float = float(os.environ.get("POCKET_LR", 3e-3))
    lambda_vq: float = float(os.environ.get("POCKET_VQ_LAMBDA", 0.25))
    loss_reduction: str = os.environ.get("POCKET_LOSS_REDUCTION", "mean_rmse")
    rmse_eps: float = float(os.environ.get("POCKET_RMSE_EPS", 1e-12))
    keep_float_max_numel: int = int(os.environ.get("POCKET_KEEP_FLOAT_MAX_NUMEL", 65_536))
    compress_min_2d_numel: int = int(os.environ.get("POCKET_COMPRESS_MIN_2D_NUMEL", 131_072))
    keep_float_store_dtype: torch.dtype = torch.float16
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
    reconstructed: Tensor
    compressed_bytes: int
    rmse: float
    vq_mse_sum: float
    mse_top100: float


def split_weight_matrix(w: Tensor, subvector_dim: int) -> tuple[Tensor, WeightSplitMeta]:
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape={tuple(w.shape)}")
    n_rows, row_width = w.shape
    if row_width % subvector_dim != 0:
        raise ValueError(
            f"Row width {row_width} is not divisible by subvector_dim {subvector_dim}; "
            "set POCKET_SUBVECTOR_DIM accordingly."
        )
    subv_per_row = row_width // subvector_dim
    s_rows = w.reshape(n_rows, subv_per_row, subvector_dim).contiguous()
    s = s_rows.reshape(-1, subvector_dim).contiguous()
    meta = WeightSplitMeta(
        n_rows=n_rows,
        row_width=row_width,
        subv_dim=subvector_dim,
        subv_per_row=subv_per_row,
    )
    return s, meta


def merge_weight_matrix(s_hat: Tensor, meta_info: WeightSplitMeta) -> Tensor:
    if s_hat.ndim != 2 or s_hat.shape[1] != meta_info.subv_dim:
        raise ValueError(
            f"Unexpected S_hat shape {tuple(s_hat.shape)}, expected [N, {meta_info.subv_dim}]"
        )
    expected_n = meta_info.n_rows * meta_info.subv_per_row
    if s_hat.shape[0] != expected_n:
        raise ValueError(f"Unexpected N={s_hat.shape[0]} expected {expected_n}")
    return s_hat.reshape(meta_info.n_rows, meta_info.row_width).contiguous()


class ReshapedLayerNorm(nn.Module):
    """RLN: normalize at full-row scope, then reshape back to subvector view."""

    def __init__(self, row_width: int, eps: float = 1e-6, affine: bool = False):
        super().__init__()
        self.row_width = row_width
        self.eps = eps
        if affine:
            self.weight = nn.Parameter(torch.ones(row_width))
            self.bias = nn.Parameter(torch.zeros(row_width))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: Tensor, meta: WeightSplitMeta) -> Tensor:
        if x.ndim != 2:
            raise ValueError(f"RLN expects [N, d], got {tuple(x.shape)}")
        rows = x.reshape(meta.n_rows, meta.subv_per_row * meta.subv_dim)
        mean = rows.mean(dim=1, keepdim=True)
        var = (rows - mean).square().mean(dim=1, keepdim=True)
        normed = (rows - mean) / torch.sqrt(var + self.eps)
        if self.weight is not None and self.bias is not None:
            normed = normed * self.weight + self.bias
        return normed.reshape(-1, meta.subv_dim).contiguous()


class PocketMLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        num_layers: int,
        activation: str = "gelu",
        residual_mode: str = "after_first",
        norm_type: str = "rln",
        rln_eps: float = 1e-6,
        rln_affine: bool = False,
        row_width: int | None = None,
    ):
        super().__init__()
        if num_layers < 2:
            raise ValueError("encoder/decoder layers must be >= 2")
        self.activation = activation.lower()
        self.residual_mode = residual_mode
        self.norm_type = norm_type

        dims = [in_dim]
        dims.extend([hidden_dim] * (num_layers - 1))
        dims.append(out_dim)
        self.layers = nn.ModuleList(nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1))

        if self.norm_type == "rln":
            if row_width is None:
                raise ValueError("row_width is required for RLN")
            self.rln = ReshapedLayerNorm(row_width=row_width, eps=rln_eps, affine=rln_affine)
        else:
            self.rln = None

    def _act(self, x: Tensor) -> Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)
        raise ValueError(f"Unsupported activation={self.activation}")

    def _norm(self, x: Tensor, meta: WeightSplitMeta) -> Tensor:
        if self.norm_type == "none":
            return x
        if self.norm_type == "ln_subvector":
            return F.layer_norm(x, (x.shape[-1],))
        if self.norm_type == "rln":
            assert self.rln is not None
            return self.rln(x, meta)
        raise ValueError(f"Unsupported norm_type={self.norm_type}")

    def forward(self, x: Tensor, meta: WeightSplitMeta) -> Tensor:
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
        return h.contiguous()


class VectorQuantizerSTE(nn.Module):
    def __init__(self, latent_dim: int, codebook_size: int, init_mode: str = "normal", init_std: float = 0.02):
        super().__init__()
        if init_mode == "normal":
            codebook = torch.randn(codebook_size, latent_dim) * init_std
        elif init_mode == "uniform":
            bound = max(init_std, 1e-6)
            codebook = (torch.rand(codebook_size, latent_dim) * 2.0 - 1.0) * bound
        else:
            raise ValueError(f"Unsupported codebook_init={init_mode}")
        self.codebook = nn.Parameter(codebook)

    @staticmethod
    def nearest_indices(z_flat: Tensor, codebook: Tensor, chunk: int = 16384) -> Tensor:
        idx_parts: list[Tensor] = []
        c = codebook.float()
        c_norm = c.square().sum(dim=1).view(1, -1)
        for start in range(0, z_flat.shape[0], chunk):
            z = z_flat[start : start + chunk].float()
            z_norm = z.square().sum(dim=1, keepdim=True)
            dist = z_norm + c_norm - 2.0 * (z @ c.t())
            idx_parts.append(torch.argmin(dist, dim=1))
        return torch.cat(idx_parts, dim=0).contiguous()

    def forward(self, z: Tensor, chunk: int = 16384) -> tuple[Tensor, Tensor, Tensor]:
        indices = self.nearest_indices(z_flat=z, codebook=self.codebook, chunk=chunk)
        z_q = self.codebook[indices]
        z_q_st = z + (z_q - z).detach()
        vq_mse_sum = (z - z_q.detach()).square().sum()
        return z_q_st, indices, vq_mse_sum

class PocketLayerCompressor:
    def __init__(self, cfg: PocketCompressorConfig, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or torch.device("cpu")

    def compress(self, name: str, weight_2d: Tensor, keep_original_dtype: torch.dtype | None = None) -> PocketCompressedLayerResult:
        w = weight_2d.detach().to(self.device, dtype=torch.float32).contiguous()
        s, meta = split_weight_matrix(w, self.cfg.subvector_dim)
        latent_dim = self.cfg.latent_dim

        encoder = PocketMLP(
            in_dim=self.cfg.subvector_dim,
            out_dim=latent_dim,
            hidden_dim=self.cfg.meta_hidden_dim,
            num_layers=self.cfg.encoder_layers,
            activation=self.cfg.activation,
            residual_mode=self.cfg.residual_mode,
            norm_type=self.cfg.norm_type,
            rln_eps=self.cfg.rln_eps,
            rln_affine=self.cfg.rln_affine,
            row_width=meta.row_width,
        ).to(self.device)
        decoder = PocketMLP(
            in_dim=latent_dim,
            out_dim=self.cfg.subvector_dim,
            hidden_dim=self.cfg.meta_hidden_dim,
            num_layers=self.cfg.decoder_layers,
            activation=self.cfg.activation,
            residual_mode=self.cfg.residual_mode,
            norm_type=self.cfg.norm_type,
            rln_eps=self.cfg.rln_eps,
            rln_affine=self.cfg.rln_affine,
            row_width=meta.row_width,
        ).to(self.device)
        quant = VectorQuantizerSTE(
            latent_dim=latent_dim,
            codebook_size=self.cfg.codebook_size,
            init_mode=self.cfg.codebook_init,
            init_std=self.cfg.codebook_init_std,
        ).to(self.device)

        params = list(encoder.parameters()) + list(decoder.parameters()) + list(quant.parameters())
        optimizer = torch.optim.Adam(params, lr=self.cfg.lr)

        s_rows = s.reshape(meta.n_rows, meta.subv_per_row, meta.subv_dim).contiguous()

        for _ in range(max(1, self.cfg.train_steps)):
            batch_rows = min(max(1, self.cfg.batch_rows), meta.n_rows)
            if batch_rows >= meta.n_rows:
                s_batch_rows = s_rows
            else:
                row_idx = torch.randint(0, meta.n_rows, (batch_rows,), device=self.device)
                s_batch_rows = s_rows[row_idx]
            s_batch = s_batch_rows.reshape(-1, meta.subv_dim)
            batch_meta = WeightSplitMeta(
                n_rows=s_batch_rows.shape[0],
                row_width=meta.row_width,
                subv_dim=meta.subv_dim,
                subv_per_row=meta.subv_per_row,
            )

            z = encoder(s_batch, batch_meta)
            z_q_st, _, vq_mse_sum = quant(z, chunk=max(1, self.cfg.chunk_rows * meta.subv_per_row))
            s_hat = decoder(z_q_st, batch_meta)

            sq = (s_batch - s_hat).square()
            if self.cfg.loss_reduction == "paper_sum":
                rmse = torch.sqrt(sq.sum() + self.cfg.rmse_eps)
            elif self.cfg.loss_reduction == "mean_rmse":
                rmse = torch.sqrt(sq.mean() + self.cfg.rmse_eps)
            else:
                raise ValueError(f"Unsupported loss_reduction={self.cfg.loss_reduction}")
            loss = rmse + self.cfg.lambda_vq * vq_mse_sum

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            z_all = encoder(s, meta)
            idx = VectorQuantizerSTE.nearest_indices(z_all, quant.codebook, chunk=max(1, self.cfg.chunk_rows * meta.subv_per_row))
            z_q = quant.codebook[idx]
            s_hat = decoder(z_q, meta)
            w_hat_fp32 = merge_weight_matrix(s_hat, meta)
            w_hat = w_hat_fp32.to(dtype=keep_original_dtype or weight_2d.dtype).contiguous()

            sqerr = (s - s_hat).square().reshape(-1)
            topk = min(100, sqerr.numel())
            mse_top100 = float(sqerr.topk(topk).values.mean().item()) if topk > 0 else 0.0
            rmse = float(torch.sqrt(sqerr.mean() + self.cfg.rmse_eps).item()) if sqerr.numel() else 0.0
            vq_mse_sum = float((z_all - z_q).square().sum().item())

            stored_codebook = quant.codebook.detach().to(dtype=torch.float16).contiguous().cpu()
            if quant.codebook.shape[0] <= 256:
                stored_indices = idx.to(dtype=torch.uint8).contiguous().cpu()
            elif quant.codebook.shape[0] <= 65_536:
                stored_indices = idx.to(dtype=torch.uint16).contiguous().cpu()
            else:
                stored_indices = idx.to(dtype=torch.int32).contiguous().cpu()

            decoder_state = {
                k: v.detach().to(dtype=torch.float16).cpu().contiguous()
                for k, v in decoder.state_dict().items()
            }
            payload: dict[str, object] = {
                "codebook": stored_codebook,
                "indices": stored_indices,
                "decoder_state": decoder_state,
                "meta": asdict(meta),
                "latent_dim": latent_dim,
                "meta_hidden_dim": self.cfg.meta_hidden_dim,
                "decoder_layers": self.cfg.decoder_layers,
                "activation": self.cfg.activation,
                "residual_mode": self.cfg.residual_mode,
                "norm_type": self.cfg.norm_type,
                "rln_eps": self.cfg.rln_eps,
                "rln_affine": self.cfg.rln_affine,
                "compressed_dtype": str((keep_original_dtype or weight_2d.dtype)).removeprefix("torch."),
            }
            compressed_bytes = tensor_nbytes(stored_codebook) + tensor_nbytes(stored_indices)
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

    @staticmethod
    def reconstruct(payload: dict[str, object], dtype: torch.dtype | None = None) -> Tensor:
        meta_dict = payload["meta"]
        meta = WeightSplitMeta(**meta_dict)
        latent_dim = int(payload["latent_dim"])

        codebook = payload["codebook"].to(dtype=torch.float32)
        idx = payload["indices"].to(dtype=torch.long).reshape(-1)
        z_q = codebook[idx]

        decoder = PocketMLP(
            in_dim=latent_dim,
            out_dim=meta.subv_dim,
            hidden_dim=int(payload["meta_hidden_dim"]),
            num_layers=int(payload["decoder_layers"]),
            activation=str(payload.get("activation", "gelu")),
            residual_mode=str(payload.get("residual_mode", "after_first")),
            norm_type=str(payload.get("norm_type", "rln")),
            rln_eps=float(payload.get("rln_eps", 1e-6)),
            rln_affine=bool(payload.get("rln_affine", False)),
            row_width=meta.row_width,
        )
        decoder.load_state_dict({k: v.to(dtype=torch.float32) for k, v in payload["decoder_state"].items()}, strict=True)
        decoder.eval()
        with torch.no_grad():
            s_hat = decoder(z_q, meta)
            out = merge_weight_matrix(s_hat, meta)
        final_dtype = dtype
        if final_dtype is None:
            final_dtype_name = str(payload.get("compressed_dtype", "float32"))
            final_dtype = getattr(torch, final_dtype_name, torch.float32)
        return out.to(dtype=final_dtype).contiguous()


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

    def _should_passthrough(self, name: str, t: Tensor) -> bool:
        return (
            (not t.is_floating_point())
            or (t.numel() <= self.cfg.keep_float_max_numel)
            or (t.ndim != 2)
            or (t.numel() < self.cfg.compress_min_2d_numel)
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

    def compress_state_dict(self, state_dict: dict[str, Tensor], step: int | None = None):
        compressed: dict[str, dict[str, object]] = {}
        dtypes: dict[str, str] = {}
        passthrough: dict[str, Tensor] = {}
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
        for name, tensor in state_dict.items():
            t = tensor.detach().to("cpu").contiguous()
            stats["param_count"] += int(t.numel())
            stats["num_tensors"] += 1
            stats["baseline_tensor_bytes"] += tensor_nbytes(t)

            should_passthrough = self._should_passthrough(name, t) or (self.cfg.mode == "periodic_refresh" and not can_refresh)
            if should_passthrough:
                if not t.is_floating_point():
                    stats["num_nonfloat_tensors"] += 1
                    kept = t
                elif any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
                    kept = t.float().contiguous()
                else:
                    if t.dtype in {torch.float32, torch.bfloat16}:
                        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
                        kept = t.to(dtype=self.cfg.keep_float_store_dtype).contiguous()
                    else:
                        kept = t
                passthrough[name] = kept
                stats["int8_payload_bytes"] += tensor_nbytes(kept)
                continue

            try:
                result = self.layer.compress(name=name, weight_2d=t, keep_original_dtype=t.dtype)
            except Exception:
                kept = t.to(dtype=self.cfg.keep_float_store_dtype).contiguous()
                passthrough[name] = kept
                stats["int8_payload_bytes"] += tensor_nbytes(kept)
                continue

            compressed[name] = result.payload
            dtypes[name] = str(t.dtype).removeprefix("torch.")
            stats["num_compressed_tensors"] += 1
            stats["int8_payload_bytes"] += result.compressed_bytes
            stats["codebook_bytes"] += tensor_nbytes(result.payload["codebook"])
            stats["indices_bytes"] += tensor_nbytes(result.payload["indices"])
            dec_bytes = sum(tensor_nbytes(v) for v in result.payload["decoder_state"].values())
            stats["decoder_bytes"] += dec_bytes
            layer_logs[name] = {
                "rmse": result.rmse,
                "vq_mse_sum": result.vq_mse_sum,
                "mse_top100": result.mse_top100,
                "subvector_dim": float(self.cfg.subvector_dim),
                "codebook_size": float(result.payload["codebook"].shape[0]),
                "encoder_layers": float(self.cfg.encoder_layers),
                "decoder_layers": float(self.cfg.decoder_layers),
            }

        obj: dict[str, object] = {
            "__quant_format__": "pocketllm_meta_v2",
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

    @staticmethod
    def decompress_state_dict(obj: dict[str, object]) -> dict[str, Tensor]:
        out: dict[str, Tensor] = {}
        passthrough_orig_dtypes = obj.get("passthrough_orig_dtypes", {})
        quantized = obj.get("quantized", {})
        dtypes = obj.get("dtypes", {})

        for name, payload in quantized.items():
            dtype_name = dtypes.get(name)
            dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else None
            out[name] = PocketLayerCompressor.reconstruct(payload, dtype=dtype).to("cpu").contiguous()

        for name, t in obj.get("passthrough", {}).items():
            out_t = t.detach().to("cpu").contiguous()
            orig_dtype = passthrough_orig_dtypes.get(name)
            if isinstance(orig_dtype, str) and hasattr(torch, orig_dtype):
                out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
            out[name] = out_t
        return out

def pocket_self_check_split_merge() -> None:
    x = torch.randn(4, 16)
    s, meta = split_weight_matrix(x, subvector_dim=4)
    y = merge_weight_matrix(s, meta)
    assert torch.equal(x, y), "split->merge roundtrip failed"


def pocket_self_check_vq_shapes() -> None:
    z = torch.randn(128, 4)
    vq = VectorQuantizerSTE(latent_dim=4, codebook_size=64)
    zq, idx, loss = vq(z)
    assert zq.shape == z.shape
    assert idx.shape == (128,)
    assert loss.ndim == 0


def pocket_self_check_reconstruction_shapes() -> None:
    cfg = PocketCompressorConfig.from_env()
    cfg.subvector_dim = 4
    cfg.latent_dim = 4
    cfg.codebook_size = 64
    cfg.train_steps = 2
    cfg.batch_rows = 8
    cfg.compress_min_2d_numel = 0
    comp = PocketLayerCompressor(cfg)
    w = torch.randn(16, 16)
    res = comp.compress("self_check.weight", w)
    rec = PocketLayerCompressor.reconstruct(res.payload)
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
    state = {"blk.q_proj.weight": torch.randn(16, 16)}
    obj, _ = model_comp.compress_state_dict(state)
    rec = PocketModelCompressor.decompress_state_dict(obj)
    assert rec["blk.q_proj.weight"].shape == state["blk.q_proj.weight"].shape


def compress_state_dict_pocketllm(state_dict: dict[str, Tensor], step: int | None = None):
    cfg = PocketCompressorConfig.from_env()
    if cfg.run_self_tests:
        pocket_self_check_split_merge()
        pocket_self_check_vq_shapes()
        pocket_self_check_reconstruction_shapes()
        pocket_self_check_artifact_roundtrip()
    model_comp = PocketModelCompressor(cfg)
    obj, stats = model_comp.compress_state_dict(state_dict, step=step)

    print(
        "[PocketLLM] mode={mode} target_regex={regex} attn_only={attn} ffn_only={ffn} all_linear={all_linear}".format(
            mode=cfg.mode,
            regex=cfg.target_regex or "<none>",
            attn=cfg.compress_attention_only,
            ffn=cfg.compress_ffn_only,
            all_linear=cfg.compress_all_linear,
        )
    )
    print(
        "[PocketLLM] d={d} k={k} enc_layers={enc} dec_layers={dec} residual_mode={res} norm_type={norm}".format(
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
            f"[PocketLLM][layer] {name} rmse={m['rmse']:.6e} vq_mse_sum={m['vq_mse_sum']:.6e} "
            f"mse_top100={m['mse_top100']:.6e}"
        )
    print(
        "[PocketLLM][bytes] codebook={cb} indices={idx} decoder={dec} total={tot}".format(
            cb=stats["codebook_bytes"],
            idx=stats["indices_bytes"],
            dec=stats["decoder_bytes"],
            tot=stats["int8_payload_bytes"],
        )
    )
    return obj, stats


def decompress_state_dict_pocketllm(obj: dict[str, object]) -> dict[str, Tensor]:
    return PocketModelCompressor.decompress_state_dict(obj)


def _load_state_dict(path: Path) -> dict[str, Tensor]:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict) and all(isinstance(v, Tensor) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            candidate = obj.get(key)
            if isinstance(candidate, dict) and all(isinstance(v, Tensor) for v in candidate.values()):
                return candidate
    raise TypeError(f"Unsupported model format in {path}. Expected state_dict-like content.")


def _compute_roundtrip_metrics(original: dict[str, Tensor], reconstructed: dict[str, Tensor]) -> tuple[float, float, int]:
    sqerr_sum = 0.0
    abs_sum = 0.0
    elem_count = 0
    for name, t in original.items():
        if name not in reconstructed:
            continue
        r = reconstructed[name]
        if not (t.is_floating_point() and r.is_floating_point()):
            continue
        tf = t.float().contiguous()
        rf = r.float().contiguous()
        if tf.shape != rf.shape:
            continue
        diff = tf - rf
        sqerr_sum += float(diff.square().sum().item())
        abs_sum += float(diff.abs().sum().item())
        elem_count += int(diff.numel())
    if elem_count == 0:
        return 0.0, 0.0, 0
    return sqerr_sum / elem_count, abs_sum / elem_count, elem_count


def run_compress_pipeline(
    model_path: str = "final_model.pt",
    output_path: str = "final_model.compress.ptz",
    code_path: str = "train_gpt.py",
    eval_bpb: bool = False,
) -> None:
    model_file = Path(model_path)
    out_file = Path(output_path)
    code_file = Path(code_path)

    state_dict = _load_state_dict(model_file)
    baseline_file_bytes = model_file.stat().st_size
    code_bytes = len(code_file.read_text(encoding="utf-8").encode("utf-8")) if code_file.exists() else 0

    quant_obj, quant_stats = compress_state_dict_pocketllm(state_dict)
    quant_buf = io.BytesIO()
    torch.save(quant_obj, quant_buf)
    quant_raw = quant_buf.getvalue()
    quant_blob = zlib.compress(quant_raw, level=9)
    out_file.write_bytes(quant_blob)

    quant_file_bytes = out_file.stat().st_size
    payload_ratio = quant_stats["baseline_tensor_bytes"] / max(quant_stats["int8_payload_bytes"], 1)
    file_ratio = baseline_file_bytes / max(quant_file_bytes, 1)

    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")
    reconstructed = decompress_state_dict_pocketllm(quant_state)
    mse, mae, n_float = _compute_roundtrip_metrics(state_dict, reconstructed)

    print(f"Serialized model: {baseline_file_bytes} bytes")
    print(f"Code size: {code_bytes} bytes")
    print(f"Total submission size: {baseline_file_bytes + code_bytes} bytes")
    print(
        f"Serialized model pocket+zlib: {quant_file_bytes} bytes "
        f"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{len(quant_raw)} "
        f"payload_ratio:{payload_ratio:.2f}x file_ratio:{file_ratio:.2f}x)"
    )
    print(f"Total submission size pocket+zlib: {quant_file_bytes + code_bytes} bytes")
    print(
        f"roundtrip_score mse:{mse:.8e} mae:{mae:.8e} compared_float_params:{n_float} "
        f"compressed_tensors:{quant_stats['num_compressed_tensors']} total_tensors:{quant_stats['num_tensors']}"
    )
    if eval_bpb:
        _run_val_bpb_eval(reconstructed)


def _run_val_bpb_eval(roundtrip_state_dict: dict[str, Tensor]) -> None:
    import train_gpt as tg

    if not torch.cuda.is_available():
        raise RuntimeError("--eval-bpb requires CUDA, same as train_gpt.py eval flow.")

    args = tg.Hyperparameters()
    rank = 0
    world_size = 1
    grad_accum_steps = 8
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    if not args.tokenizer_path.endswith(".model"):
        raise ValueError(f"Expected sentencepiece .model tokenizer, got {args.tokenizer_path}")
    sp = tg.spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )

    val_tokens = tg.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(sp, args.vocab_size, device)

    base_model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device).bfloat16()
    for module in base_model.modules():
        if isinstance(module, tg.CastedLinear):
            module.float()
    tg.restore_low_dim_params_to_fp32(base_model)
    base_model.load_state_dict(roundtrip_state_dict, strict=True)
    model: nn.Module = torch.compile(base_model, dynamic=False, fullgraph=True)

    torch.cuda.synchronize()
    q_val_loss, q_val_bpb = tg.eval_val(
        args=args,
        model=model,
        rank=rank,
        world_size=world_size,
        device=device,
        grad_accum_steps=grad_accum_steps,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    print(f"final_pocket_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f}")
    print(f"final_pocket_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress final_model.pt with PocketLLM and report size/roundtrip score.")
    parser.add_argument("--model", default="final_model.pt", help="Input .pt file")
    parser.add_argument("--out", default="final_model.compress.ptz", help="Output compressed artifact")
    parser.add_argument("--code", default="train_gpt.py", help="Code file for submission-size accounting")
    parser.add_argument("--eval-bpb", action="store_true", help="Run tokenizer-agnostic val_bpb eval using train_gpt.py helpers")
    args = parser.parse_args()
    run_compress_pipeline(model_path=args.model, output_path=args.out, code_path=args.code, eval_bpb=args.eval_bpb)


if __name__ == "__main__":
    main()
