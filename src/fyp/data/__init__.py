"""Dataset loading and validation for FLAWEDFICTIONS."""

from fyp.data.loader import load_all_splits, load_split
from fyp.data.schema import FlawedFictionsExample, FlawedFictionsSplit
from fyp.data.validation import ValidationError, validate_examples

__all__ = [
    "FlawedFictionsExample",
    "FlawedFictionsSplit",
    "ValidationError",
    "load_all_splits",
    "load_split",
    "validate_examples",
]
