# Vendored: Qwen3-TTS finetuning scripts

`dataset.py`, `sft_12hz.py` and `prepare_data.py` are copied unmodified from
[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (`main`, fetched
2026-09-19), Apache-2.0, `LICENSE` alongside them.

They live in the repo rather than being cloned at run time so a training run is
reproducible: upstream `main` can change under us, and a preempted VM that
re-clones should get the same code the earlier checkpoints were trained with.
**Keep them unmodified.** Our changes belong in `scripts/export_qwen_tts.py`
(which writes their input) and `scripts/run_qwen_tts.sh` (which drives them).

## What they do, and the one thing to know

`sft_12hz.py` computes a speaker embedding per batch from each row's
`ref_audio` and writes it into the codec embedding, so **training is already
per-sample multi-speaker**. Only its *saving* step is single-speaker: it takes
the first batch's embedding, bakes it into `codec_embedding.weight[3000]`, and
flips `tts_model_type` to `custom_voice`.

We do not want that. Keeping the checkpoint in `base` mode means a voice is
supplied at inference as ~3 seconds of reference audio, which is both what a
multi-speaker corpus needs and what keeps 980 identifiable PLD speakers out of
the weights. See docs/tts_sota_plan.md P3.
