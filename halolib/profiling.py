"""torch.profiler wired into a HF Trainer, for answering "where do the steps go".

Utilisation percentages from nvidia-smi only say a kernel was resident, not
that it did useful work: the R1 bake-off sat at 99% "utilisation" while tensor
cores were busy 19% of the time. This profiles a window of real training steps
and prints where the time actually goes.

    trainer.add_callback(profiler_callback(out_dir))

Writes a Chrome trace next to the run (open at chrome://tracing or
https://ui.perfetto.dev) and prints the top kernels plus a GPU-busy fraction.
"""
from __future__ import annotations

from pathlib import Path

# a step is skipped, then one warm-up step (CUDA caching allocator and cuDNN
# autotuning settle), then the steps that are actually recorded
WAIT, WARMUP, ACTIVE = 2, 2, 6


def profiler_callback(out_dir: str | Path, active: int = ACTIVE):
    import torch
    from torch.profiler import ProfilerActivity, profile, schedule
    from transformers import TrainerCallback

    trace_dir = Path(out_dir) / "profile"
    trace_dir.mkdir(parents=True, exist_ok=True)

    class ProfileCallback(TrainerCallback):
        def __init__(self):
            self.prof = None

        def on_train_begin(self, args, state, control, model=None, **kw):
            if model is not None:
                # transformers picks SDPA on its own when available, so say
                # what is actually in use rather than what was asked for
                impl = getattr(model.config, "_attn_implementation", "unknown")
                print(f"attention implementation: {impl}")
            self.prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=WAIT, warmup=WARMUP, active=active, repeat=1),
                record_shapes=True, profile_memory=True, with_stack=False)
            self.prof.__enter__()

        def on_step_end(self, args, state, control, **kw):
            if self.prof is None:
                return
            self.prof.step()
            if state.global_step >= WAIT + WARMUP + active:
                self._report(args)
                control.should_training_stop = True

        def on_train_end(self, args, state, control, **kw):
            if self.prof is not None:
                self._report(args)

        def _report(self, args):
            prof, self.prof = self.prof, None
            prof.__exit__(None, None, None)
            trace = trace_dir / "trace.json.gz"
            prof.export_chrome_trace(str(trace))

            events = prof.key_averages()
            print("\n=== top kernels by GPU time ===")
            print(events.table(sort_by="self_device_time_total", row_limit=15,
                               max_name_column_width=55))

            gpu_us = sum(e.self_device_time_total for e in events)
            cpu_us = sum(e.self_cpu_time_total for e in events)
            steps = ACTIVE
            eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
            print(f"\n=== {steps} steps, effective batch {eff_batch} ===")
            print(f"GPU kernel time : {gpu_us / 1e6:.2f} s")
            print(f"CPU time        : {cpu_us / 1e6:.2f} s")
            if cpu_us:
                # well below 1.0 means the GPU waits on the host: data loading,
                # small kernels, or too many Python-side launches per step
                print(f"GPU/CPU ratio   : {gpu_us / cpu_us:.2f}")
            print(f"peak GPU memory : "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB of "
                  f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
            print(f"trace           : {trace}")

    return ProfileCallback()
