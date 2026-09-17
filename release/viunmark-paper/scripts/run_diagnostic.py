#!/usr/bin/env python
from __future__ import annotations

from viunmark.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["diagnostic", *(__import__("sys").argv[1:])]))
