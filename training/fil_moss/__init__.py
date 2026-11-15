"""Filipino MOSS-TTS-Realtime: stage 1 (trustworthy SFT) and stage 2 (judge-robust GRPO).

Pure-Python modules (``scoring``, ``gates``, ``guard``) run anywhere and carry the decisions.
GPU modules (``runtime``, ``judges``, ``encode``, ``stage1``, ``stage2``) run on dwnd-01 inside
the pinned MOSS environment at ``$MOSS_PILOT_ROOT`` (default ``~/work/hil-moss-pilot``).
"""
