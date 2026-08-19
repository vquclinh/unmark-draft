"""UNMARK - tone-factored input adaptation for Vietnamese.

Only lightweight, dependency-free utilities live in this package. Anything that
needs torch or transformers belongs in `scripts/` and must import them lazily,
so that `pip install -r requirements/dev.txt` is enough to run the test suite.
"""

__all__ = ["orthography", "gates"]
