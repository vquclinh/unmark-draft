"""RESTORE Stage-2 baseline package.

RESTORE is intentionally separate from the UNMARK adapter path. This package
contains the frozen external diacritic-restoration baseline only:

    observed text -> frozen RESTORE seq2seq model -> frozen PhoBERT -> own head

It must not import UNMARK adapter or Stage-1 training machinery.
"""

from unmark.baselines.restore.config import (
    RESTORE_MODEL_ID,
    RESTORE_MODEL_REVISION,
    RESTORE_PROTOCOL_VERSION,
)

__all__ = [
    "RESTORE_MODEL_ID",
    "RESTORE_MODEL_REVISION",
    "RESTORE_PROTOCOL_VERSION",
]
