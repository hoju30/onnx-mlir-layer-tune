#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the mixed-precision numerical suite; missing prerequisites are failures.

Use ordinary ``python -m unittest`` when intentionally running only the tests
whose optional dependencies are available. This entry point is for validation
and CI, where a skipped test must not be reported as a successful full run.
"""

import argparse
from pathlib import Path
import sys
import unittest


class StrictResult(unittest.TextTestResult):
    def wasSuccessful(self):
        return super().wasSuccessful() and not self.skipped and self.testsRun > 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tests", nargs="*", help="Optional unittest module, class or method names"
    )
    args = parser.parse_args()
    loader = unittest.TestLoader()
    if args.tests:
        suite = loader.loadTestsFromNames(args.tests)
    else:
        suite = loader.discover(
            str(Path(__file__).resolve().parent), pattern="test_mixed_precision_*.py"
        )
    result = unittest.TextTestRunner(verbosity=2, resultclass=StrictResult).run(suite)
    if result.skipped:
        print(
            f"Mixed-precision validation failed: {len(result.skipped)} skipped "
            "test(s). Install/build the prerequisites reported above.",
            file=sys.stderr,
        )
    if result.testsRun == 0:
        print("Mixed-precision validation failed: no tests ran.", file=sys.stderr)
    return int(
        not result.wasSuccessful() or bool(result.skipped) or result.testsRun == 0
    )


if __name__ == "__main__":
    sys.exit(main())
