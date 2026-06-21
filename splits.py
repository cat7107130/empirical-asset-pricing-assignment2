"""
splits.py
=========
Generates the GKX recursive (expanding-window) train / validation / test splits.

GKX scheme followed exactly:
  * Training window EXPANDS: it always starts at ``start_year``.
  * Validation window ROLLS forward: it is the ``val_years`` years immediately
    preceding the test year.
  * Each test year is a single out-of-sample year; models are refit every year.

With start_year=1971, train_years=18, val_years=12:
  first test year = 1971 + 18 + 12 = 2001
  test 2001 -> train 1971-1988, validation 1989-2000
  test 2002 -> train 1971-1989, validation 1990-2001   (train grows, val rolls)
  ...

All window lengths are parameters so the same code path drives both the
lightweight pilot and the full 1971-2025 run.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    """One recursive step: year ranges are inclusive."""
    test_year: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int


def generate_splits(start_year: int, end_year: int,
                    train_years: int, val_years: int):
    """Yield Split objects for every test year from the first feasible year
    through ``end_year`` (inclusive).

    first_test_year = start_year + train_years + val_years
    """
    first_test = start_year + train_years + val_years
    for test_year in range(first_test, end_year + 1):
        yield Split(
            test_year=test_year,
            train_start=start_year,
            train_end=test_year - val_years - 1,
            val_start=test_year - val_years,
            val_end=test_year - 1,
        )


if __name__ == "__main__":
    # Show the full-run schedule.
    for s in generate_splits(1971, 2025, 18, 12):
        print(s)
