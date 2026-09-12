"""
halolib — shared helpers for the halohalo corpus pipelines.

Speech-side modules (audio, manifest, raw, shard, sources.*) are imported as
submodules — e.g. `from halolib import audio`, `from halolib.sources import
livestream` — to keep this top-level namespace light. align/vad/qc require the
GPU stack and are imported lazily by the pipeline drivers.

The text-side re-exports below are resolved lazily too (PEP 562): `fineweb`
pulls in `datasets`, and an import that heavy should not be the price of
`from halolib import audio` or of running an upload script that never touches
a Dataset. `from halolib import clean_text` keeps working exactly as before.
"""

_LAZY = {
    "clean_text": ".cleaner",
    "is_usable": ".cleaner",
    "add_fineweb_columns": ".fineweb",
    "dedup_against": ".fineweb",
    "append_to": ".fineweb",
    "push_with_retry": ".fineweb",
    "train_test_split": ".fineweb",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value  # cache, so this indirection costs one lookup
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
