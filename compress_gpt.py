'''
This script is refer to Proceedings of the AAAI Conference on Artificial Intelligence (2024) paper 
"PocketLLM: Efficient Large Language Model Compression via Meta-Encoder and Subvector Quantization" (https://arxiv.org/abs/2406.09463) 
for a simple example of compressing a GPT model's state_dict using the PocketLLM approach, 
which involves splitting weight tensors into subvectors, encoding them with a meta-encoder, 
quantizing to a codebook, and decoding with a meta-decoder. 
The script also includes utilities for keeping certain tensors in float format, computing roundtrip reconstruction metrics, 
and evaluating bits-per-byte on a validation set using the original training code's evaluation flow.
'''

from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path
import zlib

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

POCKET_KEEP_FLOAT_MAX_NUMEL = int(os.environ.get("POCKET_KEEP_FLOAT_MAX_NUMEL", 65_536))
POCKET_KEEP_FLOAT_STORE_DTYPE = torch.float16
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


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=POCKET_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


@dataclass(frozen=True)
class SplitSpec:
    n_rows: int
    row_width: int
    subv_dim: int
    subv_per_row: int
    pad_cols: int


def _split_weight_rows(w: Tensor, subv_dim: int) -> tuple[Tensor, SplitSpec]:
    n_rows, n_cols = w.shape
    pad_cols = (subv_dim - (n_cols % subv_dim)) % subv_dim
    if pad_cols:
        w = F.pad(w, (0, pad_cols))
    row_width = w.shape[1]
    subv_per_row = row_width // subv_dim
    s = w.view(n_rows, subv_per_row, subv_dim).contiguous()
    return s, SplitSpec(n_rows, row_width, subv_dim, subv_per_row, pad_cols)


def _merge_weight_rows(s_hat: Tensor, spec: SplitSpec, dtype: torch.dtype) -> Tensor:
    out = s_hat.view(spec.n_rows, spec.row_width)
    if spec.pad_cols:
        out = out[:, : spec.row_width - spec.pad_cols]
    return out.to(dtype=dtype).contiguous()


class ReshapedLayerNorm(nn.Module):
    def __init__(self, eps: float = POCKET_RLN_EPS):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [rows, subv_per_row, subv_dim]
        rows, subv_per_row, subv_dim = x.shape
        x2 = x.view(rows, subv_per_row * subv_dim)
        rms = torch.sqrt(x2.square().mean(dim=1, keepdim=True) + self.eps)
        x2 = x2 / rms
        return x2.view(rows, subv_per_row, subv_dim).contiguous()


class MetaMLP(nn.Module):
    def __init__(self, d: int, hidden: int, num_layers: int):
        super().__init__()
        if num_layers < 2:
            raise ValueError("POCKET_META_NUM_LAYERS must be >= 2")
        self.rln = ReshapedLayerNorm()
        self.in_proj = nn.Linear(d, hidden)
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(max(0, num_layers - 2)))
        self.out_proj = nn.Linear(hidden, d)

    def forward(self, x: Tensor) -> Tensor:
        # Residual links are used in hidden blocks, excluding the first/last projections.
        h = F.gelu(self.in_proj(self.rln(x)))
        for block in self.blocks:
            y = F.gelu(block(self.rln(h)))
            h = h + y
        return self.out_proj(self.rln(h))


def _nearest_codebook_indices(z_flat: Tensor, codebook: Tensor, chunk: int = 16384) -> Tensor:
    idx_parts: list[Tensor] = []
    c = codebook.float()
    c_norm = c.square().sum(dim=1).view(1, -1)
    for start in range(0, z_flat.shape[0], chunk):
        z = z_flat[start : start + chunk].float()
        z_norm = z.square().sum(dim=1, keepdim=True)
        dist = z_norm + c_norm - 2.0 * (z @ c.t())
        idx_parts.append(torch.argmin(dist, dim=1))
    return torch.cat(idx_parts, dim=0).contiguous()


def _build_indices_tensor(indices: Tensor, codebook_size: int) -> Tensor:
    if codebook_size <= 256:
        return indices.to(dtype=torch.uint8).contiguous()
    if codebook_size <= 65_536:
        return indices.to(dtype=torch.uint16).contiguous()
    return indices.to(dtype=torch.int32).contiguous()


def _compress_tensor_pocketllm(name: str, t: Tensor) -> tuple[dict[str, object], int]:
    if t.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor for PocketLLM compression, got shape={tuple(t.shape)} at {name}")

    w = t.float().contiguous()
    s, spec = _split_weight_rows(w, POCKET_SUBVECTOR_DIM)
    device = torch.device("cpu")
    s = s.to(device=device)
    n_rows = s.shape[0]
    subv_per_row = s.shape[1]
    flat_count = n_rows * subv_per_row
    d = s.shape[2]
    k = min(POCKET_CODEBOOK_SIZE, max(16, flat_count))

    encoder = MetaMLP(d=d, hidden=POCKET_META_HIDDEN_DIM, num_layers=POCKET_META_NUM_LAYERS).to(device)
    decoder = MetaMLP(d=d, hidden=POCKET_META_HIDDEN_DIM, num_layers=POCKET_META_NUM_LAYERS).to(device)

    with torch.no_grad():
        z0 = encoder(s[: min(n_rows, 256)])
        mean = float(z0.mean().item()) if z0.numel() else 0.0
        std = float(z0.std(unbiased=False).item()) if z0.numel() else 1.0
        std = max(std, 1e-3)
    codebook = nn.Parameter(torch.randn(k, d, device=device) * std + mean)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()) + [codebook],
        lr=POCKET_LR,
    )

    batch_rows = min(POCKET_BATCH_ROWS, n_rows)
    steps = max(1, POCKET_TRAIN_STEPS)
    for _ in range(steps):
        if batch_rows >= n_rows:
            s_batch = s
        else:
            row_idx = torch.randint(0, n_rows, (batch_rows,), device=device)
            s_batch = s[row_idx]

        z = encoder(s_batch)
        z_flat = z.view(-1, d)
        idx = _nearest_codebook_indices(z_flat, codebook)
        z_q = codebook[idx].view_as(z)
        z_ste = z + (z_q - z).detach()
        s_hat = decoder(z_ste)

        rmse = torch.sqrt(torch.mean((s_batch - s_hat) ** 2) + 1e-12)
        vq = torch.mean((z - z_q.detach()) ** 2)
        loss = rmse + POCKET_VQ_LAMBDA * vq

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        idx_chunks: list[Tensor] = []
        for start in range(0, n_rows, POCKET_CHUNK_ROWS):
            z_chunk = encoder(s[start : start + POCKET_CHUNK_ROWS]).view(-1, d)
            idx_chunks.append(_nearest_codebook_indices(z_chunk, codebook))
        indices = torch.cat(idx_chunks, dim=0).contiguous()
        stored_indices = _build_indices_tensor(indices, k)
        stored_codebook = codebook.detach().to(dtype=torch.float16).contiguous()

        decoder_state: dict[str, Tensor] = {
            key: value.detach().to(dtype=torch.float16).contiguous()
            for key, value in decoder.state_dict().items()
        }

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


def compress_state_dict_pocketllm(state_dict: dict[str, Tensor]):
    # Same interface as example.py, but compression method follows PocketLLM
    # (split -> meta-encoder -> codebook nearest-neighbor -> meta-decoder).
    compressed: dict[str, dict[str, object]] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
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
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        should_passthrough = (
            (not t.is_floating_point())
            or (t.numel() <= POCKET_KEEP_FLOAT_MAX_NUMEL)
            or (t.ndim != 2)
            or (t.numel() < POCKET_COMPRESS_MIN_2D_NUMEL)
            or any(pattern in name for pattern in CONTROL_TENSOR_NAME_PATTERNS)
        )

        if should_passthrough:
            if not t.is_floating_point():
                stats["num_nonfloat_tensors"] += 1
                kept = t
            else:
                kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        payload, compressed_bytes = _compress_tensor_pocketllm(name, t)
        compressed[name] = payload
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["num_compressed_tensors"] += 1
        stats["int8_payload_bytes"] += compressed_bytes

    obj: dict[str, object] = {
        "__quant_format__": "pocketllm_meta_v1",
        "quantized": compressed,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


def decompress_state_dict_pocketllm(obj: dict[str, object]) -> dict[str, Tensor]:
    out: dict[str, Tensor] = {}
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

        codebook = payload["codebook"].to(dtype=torch.float32)
        indices = payload["indices"].to(dtype=torch.long).view(-1)
        z_q = codebook[indices].view(n_rows, subv_per_row, subv_dim).contiguous()

        decoder = MetaMLP(d=subv_dim, hidden=hidden, num_layers=layers)
        decoder.load_state_dict(
            {key: value.to(dtype=torch.float32) for key, value in payload["decoder_state"].items()},
            strict=True,
        )
        decoder.eval()
        with torch.no_grad():
            s_hat = decoder(z_q)
            spec = SplitSpec(
                n_rows=n_rows,
                row_width=row_width,
                subv_dim=subv_dim,
                subv_per_row=subv_per_row,
                pad_cols=pad_cols,
            )
            dtype = getattr(torch, dtypes[name])
            out[name] = _merge_weight_rows(s_hat, spec, dtype=dtype)

    for name, t in obj["passthrough"].items():
        out_t = t.detach().to("cpu").contiguous()
        orig_dtype = passthrough_orig_dtypes.get(name)
        if isinstance(orig_dtype, str):
            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()
        out[name] = out_t
    return out


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


def _compute_roundtrip_metrics(
    original: dict[str, Tensor],
    reconstructed: dict[str, Tensor],
) -> tuple[float, float, int]:
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
