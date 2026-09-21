"""
Multi-speaker SFT of Qwen3-TTS-12Hz-Base on a PLD export. Plan item P3.

The forward pass and loss are upstream's (QwenLM/Qwen3-TTS
finetuning/sft_12hz.py, Apache-2.0, vendored unmodified under
third_party/qwen3_tts_finetuning). Three things differ, all because upstream
wrote for one speaker and we have hundreds:

1. **Reference mels are made the same length.** Upstream's collate does
   torch.cat(ref_mels, dim=0), which only works when every row shares one
   reference clip. Ours differ per row, so each reference is cropped to
   REF_SECONDS — the model card's own "3-second rapid voice clone" — and
   shorter ones are tiled rather than zero-padded, since silence would drag a
   pooled speaker embedding toward nothing in particular.

2. **The checkpoint stays in `base` mode.** Upstream ends by writing the first
   batch's speaker embedding into codec_embedding.weight[3000], deleting the
   speaker encoder and flipping tts_model_type to custom_voice: one baked-in
   voice. We save the whole model, speaker encoder included, so a voice is
   still supplied at inference as a few seconds of reference audio. That is
   what a multi-speaker corpus needs, and it keeps 980 identifiable PLD
   speakers out of the weights.

3. **Sized for the card.** Upstream defaults to batch 2 x accum 4, the same
   accessible-hardware habit that had Whisper using 13 GB of 96. Qwen's codec
   runs at 12.5 Hz, so a 10 s clip is ~125 frames; batches can be large.

  venv_qwen/bin/python3 scripts/sft_qwen_tts.py --language ceb --max-steps 10   # smoke
  venv_qwen/bin/python3 scripts/sft_qwen_tts.py --language ceb
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party" / "qwen3_tts_finetuning"))

REF_SECONDS = 3.0
MEL_HOP, MEL_SR = 256, 24000            # as in upstream's extract_mels


def fix_ref_length(mel: torch.Tensor, frames: int) -> torch.Tensor:
    """(1, T, n_mels) -> (1, frames, n_mels): crop long, tile short."""
    t = mel.shape[1]
    if t >= frames:
        return mel[:, :frames]
    reps = -(-frames // t)
    return mel.repeat(1, reps, 1)[:, :frames]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--language", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--attn", default="sdpa",
                    help="upstream asks for flash_attention_2, which needs a "
                         "long compile; sdpa is PyTorch's fused path")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from dataset import TTSDataset
    from huggingface_hub import snapshot_download
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from torch.utils.data import DataLoader
    from transformers import AutoConfig

    work = Path(os.environ.get("PLD_WORK_DIR", "/mnt/data/pld_shards"))
    runs = Path(os.environ.get("FINETUNE_DIR", "/mnt/data/finetune_runs"))
    jsonl = work / "qwen_tts" / f"pld_{args.language}" / "train_coded.jsonl"
    out_dir = runs / f"qwen3tts_pld_{args.language}"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dir = snapshot_download(args.model, token=os.environ.get("HF_TOKEN"))
    wrapper = Qwen3TTSModel.from_pretrained(
        model_dir, torch_dtype=torch.bfloat16, attn_implementation=args.attn)
    config = AutoConfig.from_pretrained(model_dir)
    model = wrapper.model.to("cuda").train()

    rows = [json.loads(line) for line in open(jsonl, encoding="utf-8")]
    dataset = TTSDataset(rows, wrapper.processor, config)
    frames = int(REF_SECONDS * MEL_SR / MEL_HOP)

    def collate(batch):
        for b in batch:
            b["ref_mel"] = fix_ref_length(b["ref_mel"], frames)
        return dataset.collate_fn(batch)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate, num_workers=args.workers,
                        drop_last=True)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                              fused=True)
    print(f"{args.language}: {len(rows)} clips, batch {args.batch_size}x"
          f"{args.grad_accum}, {len(loader)} steps/epoch, ref {frames} frames")

    step, t0 = 0, time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(args.epochs):
        for batch in loader:
            batch = {k: v.to("cuda") for k, v in batch.items()}
            input_ids, codec_ids = batch["input_ids"], batch["codec_ids"]
            codec_mask = batch["codec_mask"]

            # ---- upstream's forward, unchanged in substance ----
            with torch.no_grad():
                spk = model.speaker_encoder(batch["ref_mels"].to(model.dtype))
            text_emb = model.talker.model.text_embedding(input_ids[:, :, 0]) \
                * batch["text_embedding_mask"]
            codec_emb = model.talker.model.codec_embedding(input_ids[:, :, 1]) \
                * batch["codec_embedding_mask"]
            codec_emb[:, 6, :] = spk                 # per row: that row's speaker
            embeds = text_emb + codec_emb
            for i in range(1, 16):
                e = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                embeds = embeds + e * codec_mask.unsqueeze(-1)

            out = model.talker(inputs_embeds=embeds[:, :-1, :],
                               attention_mask=batch["attention_mask"][:, :-1],
                               labels=batch["codec_0_labels"][:, 1:],
                               output_hidden_states=True)
            hidden = out.hidden_states[0][-1][codec_mask[:, :-1]]
            _, sub_loss = model.talker.forward_sub_talker_finetune(
                codec_ids[codec_mask], hidden)
            loss = (out.loss + 0.3 * sub_loss) / args.grad_accum
            # ----------------------------------------------------
            loss.backward()

            step += 1
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                optim.zero_grad(set_to_none=True)
            if step % 20 == 0 or step == 1:
                print(f"epoch {epoch} step {step} loss {loss.item() * args.grad_accum:.4f} "
                      f"({(time.perf_counter() - t0) / step:.2f} s/step, peak "
                      f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)", flush=True)
            if args.max_steps and step >= args.max_steps:
                print("smoke test: stopping before any save")
                return

    # base mode, speaker encoder kept: see the docstring
    final = out_dir / "final"
    shutil.copytree(model_dir, final, dirs_exist_ok=True, symlinks=False)
    from safetensors.torch import save_file
    for old in final.glob("model*.safetensors*"):
        old.unlink()
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
              str(final / "model.safetensors"))
    (out_dir / "result.json").write_text(json.dumps({
        "model": args.model, "language": args.language, "steps": step,
        "final_loss": loss.item() * args.grad_accum,
        "peak_gib": torch.cuda.max_memory_allocated() / 2**30}, indent=2))
    print(f"saved {final}")


if __name__ == "__main__":
    main()
