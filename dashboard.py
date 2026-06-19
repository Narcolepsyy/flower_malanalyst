"""Compatibility wrapper — use ``flmal-dashboard --view live`` instead."""

from __future__ import annotations

import sys
import warnings

warnings.warn(
    "dashboard.py is deprecated. Use 'flmal-dashboard --view live' instead.",
    DeprecationWarning,
    stacklevel=1,
)

from dashboard_interactive import main

if __name__ == "__main__":
    if "--view" not in sys.argv:
        sys.argv.extend(["--view", "live"])
    main()
