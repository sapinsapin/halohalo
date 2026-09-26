"""Hugging Face Whisper checkpoint -> OpenAI's original layout.

MLX (mlx-whisper), whisper.cpp (convert-pt-to-ggml.py) and whisper.cpp's Core
ML encoder (convert-whisper-to-coreml.py) all start from OpenAI's layout, so
one rename serves three targets. The mapping is the inverse of transformers'
convert_openai_to_hf.py; fine-tuning changes weights, not names or shapes, so
the published fine-tunes convert like the base model.

  python -m porting.whisper_openai sapinsapin/whisper-small-pld-ceb
  -> $FINETUNE_DIR/port/artefacts/<name>/openai/model.pt  {"dims", "model_state_dict"}
"""

import argparse
import json
import os
from pathlib import Path

FINETUNE_DIR = Path(os.environ.get("FINETUNE_DIR", "/mnt/d/halohalo/finetune_runs"))
ART = FINETUNE_DIR / "port" / "artefacts"

RENAMES = [                       # applied in order; order matters
    ("model.", ""),
    (".layers.", ".blocks."),
    (".self_attn_layer_norm.", ".attn_ln."),
    (".encoder_attn_layer_norm.", ".cross_attn_ln."),
    (".final_layer_norm.", ".mlp_ln."),
    (".self_attn.", ".attn."),
    (".encoder_attn.", ".cross_attn."),
    (".q_proj.", ".query."),
    (".k_proj.", ".key."),
    (".v_proj.", ".value."),
    (".out_proj.", ".out."),
    (".fc1.", ".mlp.0."),
    (".fc2.", ".mlp.2."),
    ("encoder.embed_positions.weight", "encoder.positional_embedding"),
    ("decoder.embed_positions.weight", "decoder.positional_embedding"),
    ("decoder.embed_tokens.", "decoder.token_embedding."),
    ("encoder.layer_norm.", "encoder.ln_post."),
    ("decoder.layer_norm.", "decoder.ln."),
]


def dims(cfg: dict) -> dict:
    return {"n_mels": cfg["num_mel_bins"], "n_audio_ctx": cfg["max_source_positions"],
            "n_audio_state": cfg["d_model"], "n_audio_head": cfg["encoder_attention_heads"],
            "n_audio_layer": cfg["encoder_layers"], "n_vocab": cfg["vocab_size"],
            "n_text_ctx": cfg["max_target_positions"], "n_text_state": cfg["d_model"],
            "n_text_head": cfg["decoder_attention_heads"], "n_text_layer": cfg["decoder_layers"]}


def rename(k: str) -> str:
    for a, b in RENAMES:
        k = k.replace(a, b)
    return k


def convert(repo: str, out_dir: Path | None = None) -> Path:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    tok = os.environ.get("HF_TOKEN")
    d = Path(snapshot_download(repo, token=tok, allow_patterns=["*.json", "*.safetensors"]))
    cfg = json.loads((d / "config.json").read_text())
    sd = {}
    for f in sorted(d.glob("*.safetensors")):
        sd.update(load_file(f))
    sd.pop("proj_out.weight", None)            # tied to the token embedding
    # fp16, as OpenAI ships them: whisper.cpp's converter marks 2-D tensors
    # f16 without converting them, so an fp32 checkpoint yields a corrupt file
    new = {rename(k): v.half() for k, v in sd.items()}
    left = [k for k in new if any(s in k for s in ("_proj", "layer_norm", "embed_", "layers"))]
    if left:
        raise ValueError(f"unmapped keys: {left[:5]}")
    gen = json.loads((d / "generation_config.json").read_text()) if (d / "generation_config.json").exists() else {}
    out_dir = out_dir or ART / repo.split("/")[-1] / "openai"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"dims": dims(cfg), "model_state_dict": new}, out_dir / "model.pt")
    (out_dir / "halohalo.json").write_text(json.dumps(
        {"source": repo, "language": (gen.get("language") or "tagalog").strip("<|>")}, indent=1))
    return out_dir / "model.pt"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo")
    args = ap.parse_args()
    print(convert(args.repo))


if __name__ == "__main__":
    main()
