"""Public orthographic channel constants.

Full Vietnamese tokenization and syllable inventory handling are outside this
publication export. The adapter channel tables are exposed here so public tests
can verify their exact sizes and labels.
"""

TONE_LABELS = ("SAC", "HUYEN", "HOI", "NGA", "NANG", "UNMARKED", "UNUSED")
LETTER_LABELS = ("NONE", "BREVE", "CIRCUMFLEX", "HORN", "STROKE")
TONE_NA_SENTINEL = -1
LETTER_NA_SENTINEL = -1
