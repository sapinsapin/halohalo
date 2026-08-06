from .cleaner import clean_text, is_usable
from .fineweb import add_fineweb_columns, dedup_against, append_to, push_with_retry, train_test_split

# Speech-side modules (audio, manifest, shard, sources.*) are imported as
# submodules — e.g. `from halolib import audio`, `from halolib.sources import
# livestream` — to keep this top-level namespace light. align/vad/qc require
# the GPU stack and are imported lazily by the pipeline drivers.
