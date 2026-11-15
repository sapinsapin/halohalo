"""Filipino MOSS-TTS-Realtime: stage 1 (engineering smoke) and stage 2 (bounded baseline SFT + held-out
evaluation), following training/fil_moss/README.md and the Hiligaynon GB10 pilot.

Pure-Python modules (``scoring``, ``gates``, ``guard``, ``timing``) run anywhere and carry the decisions.
GPU modules (``runtime``, ``judges``, ``encode``, ``evaluation``, ``stage1``, ``stage2``, ``bench``) run on
dwnd-01 inside the pinned MOSS environment at ``$MOSS_PILOT_ROOT`` (default ``~/work/hil-moss-pilot``);
``judge_server`` runs on the RTX 3070.
"""
