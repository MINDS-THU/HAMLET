"""Top-level hamlet package with lazy submodule access.

Enables usage like:
    import hamlet
    hamlet.train.get_model_and_tokenizer(...)

without eagerly importing heavy dependencies until the submodule is accessed.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Lazily import submodules on attribute access.

    This keeps `import hamlet` lightweight while still supporting
    attribute-style access to `hamlet.core`, `hamlet.serve`, and `hamlet.train`.
    """
    if name in {"core", "serve", "train"}:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def main() -> None:
    print("Hello from hamlet!")
