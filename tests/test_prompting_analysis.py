import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location("prompting_analysis", Path("scripts/analyse_prompting_main10.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_paired_counts_include_unchanged_correct_and_wrong():
    outcome = analysis.paired([0, 1, 1, 0], [1, 0, 1, 0])
    assert outcome == {"improved": 1, "regressed": 1, "unchanged": 2, "unchanged_correct": 1, "unchanged_incorrect": 1}


def test_paired_counts_do_not_hide_incomplete_scope():
    with pytest.raises(ValueError, match="identical denominators"):
        analysis.paired([1, 0], [1])


def test_empty_html_fragment_is_reported_without_altering_the_scorer():
    fragments = analysis.empty_fragments(['<p>A sentence. "</p>'])
    assert any(fragment.endswith("</p>") for fragment in fragments)
