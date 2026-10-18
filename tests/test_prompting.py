from __future__ import annotations

import hashlib

from fyp.prompting.parser import parse_official_response
from fyp.prompting.records import InferenceRecord
from fyp.prompting.strategies import get_benchmark_strategy


EXAMPLE = {
    "example_id": "fixture-1",
    "story": "First, Mara locked the blue door. Later, the same door was red.",
}


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def test_official_prompt_template_hashes_are_versioned() -> None:
    expected = {
        "vanilla": "68EEE6C1A3F1617A3F1E4C6A08939E0EA8110EE1C1678B176D832A3D952D6AEA",
        "cot": "D54BF0DA3D20A61C38F2CE786ECF44AAE8971E45ACF2CF9196613293879CADDE",
        "fewshot": "33165F58174AA6D56EEE31ECAF61E937B3D3E752CCE316F3012A26927CEA5614",
        "evidence_first": "F2391B1109137AA0DEB643094CB64392316705BF202CEF1B58DABB2308862238",
        "verify_revise": "8064404B9BE0B0C1FC50F78BC6B203DC83632609D26F658C3AAB71BFF4C3EED0",
    }

    for name, expected_hash in expected.items():
        result = get_benchmark_strategy(name).build_prompt(EXAMPLE)
        assert result.template_hash == expected_hash


def test_vanilla_prompt_structure_and_story_insertion() -> None:
    result = get_benchmark_strategy("vanilla").build_prompt(EXAMPLE)

    assert result.strategy == "vanilla"
    assert result.prompt_version == "vanilla_v1"
    assert result.stages[0].role == "user"
    assert len(result.stages) == 1
    assert "Some tips and tricks for the task:" in result.prompt
    assert "<scratchpad>" not in result.prompt
    assert f"<story>\n{EXAMPLE['story']}\n</story>" in result.prompt
    assert result.prompt.index("Here is the story to analyze:") < result.prompt.index("<story>")
    assert result.prompt.index("<story>") < result.prompt.index("Please format your response")


def test_cot_prompt_is_single_stage_with_scratchpad() -> None:
    result = get_benchmark_strategy("cot").build_prompt(EXAMPLE)

    assert result.strategy == "cot"
    assert result.prompt_version == "cot_v1"
    assert len(result.stages) == 1
    assert "<scratchpad>" in result.prompt
    assert "Let's think step by step:" in result.prompt
    assert result.prompt.index("<scratchpad>") < result.prompt.index("<explanation>")


def test_fewshot_prompt_inserts_examples_before_final_story_in_official_order() -> None:
    result = get_benchmark_strategy("fewshot").build_prompt(EXAMPLE)

    assert result.strategy == "fewshot"
    assert result.prompt_version == "fewshot_v1"
    assert len(result.stages) == 1
    assert "<examples>" in result.prompt
    assert "<example1>" in result.prompt
    assert "<example2>" in result.prompt
    assert result.prompt.index("<example1>") < result.prompt.index("<example2>")
    assert result.prompt.index("<examples>") < result.prompt.index("Finally, here is the story to analyze:")
    assert result.prompt.index("</examples>") < result.prompt.index(f"<story>\n{EXAMPLE['story']}\n</story>")
    assert "Princess Moonlight" in result.prompt
    assert "No continuity error found." in result.prompt


def test_prompt_construction_is_deterministic_and_hashes_rendered_prompt() -> None:
    strategy = get_benchmark_strategy("fewshot")

    first = strategy.build_prompt(EXAMPLE)
    second = strategy.build_prompt(EXAMPLE)

    assert first == second
    assert first.prompt_hash == _sha256(first.prompt)
    assert first.stages[0].prompt_hash == first.prompt_hash
    assert first.chat_messages == [{"role": "user", "content": first.prompt}]


def test_evidence_first_prompt_is_single_stage_and_quote_pair_focused() -> None:
    result = get_benchmark_strategy("evidence_first").build_prompt(EXAMPLE)

    assert result.strategy == "evidence_first"
    assert result.prompt_version == "evidence_first_v1"
    assert len(result.stages) == 1
    assert result.stages[0].role == "user"
    assert "Use an evidence-first procedure" in result.prompt
    assert "quote both" in result.prompt
    assert "<error_lines>" in result.prompt
    assert "<contradicted_lines>" in result.prompt
    assert f"<story>\n{EXAMPLE['story']}\n</story>" in result.prompt
    assert "cont_error" not in result.prompt


def test_verify_revise_prompt_is_single_stage_and_hides_draft_answer() -> None:
    result = get_benchmark_strategy("verify_revise").build_prompt(EXAMPLE)

    assert result.strategy == "verify_revise"
    assert result.prompt_version == "verify_revise_v1"
    assert len(result.stages) == 1
    assert "Use a verify-and-revise procedure" in result.prompt
    assert "Do not expose a separate draft answer" in result.prompt
    assert "false negative" in result.prompt
    assert "false positive" in result.prompt
    assert "<decision>" in result.prompt
    assert f"<story>\n{EXAMPLE['story']}\n</story>" in result.prompt
    assert "cont_error" not in result.prompt


def test_valid_output_parsing_positive() -> None:
    raw = """<response>
<explanation>
The door color changes.
</explanation>
<error_lines>
- Later, the same door was red.
</error_lines>
<contradicted_lines>
* First, Mara locked the blue door.
</contradicted_lines>
<decision>
There is a continuity error.
</decision>
</response>"""

    parsed = parse_official_response(raw)

    assert parsed.cont_error == 1.0
    assert parsed.cont_error_expl == "The door color changes."
    assert parsed.cont_error_lines == ("Later, the same door was red.",)
    assert parsed.contradicted_lines == ("First, Mara locked the blue door.",)
    assert parsed.parsing_failures == ()


def test_valid_output_parsing_negative() -> None:
    raw = """<response>
<explanation>
The story is consistent.
</explanation>
<error_lines>

</error_lines>
<contradicted_lines>

</contradicted_lines>
<decision>
No continuity error found.
</decision>
</response>"""

    parsed = parse_official_response(raw)

    assert parsed.cont_error == 0.0
    assert parsed.cont_error_lines == ()
    assert parsed.contradicted_lines == ()
    assert parsed.parsing_failures == ()


def test_malformed_output_records_failures_but_uses_official_fallbacks() -> None:
    raw = """<response>
<explanation>Partial explanation
<error_lines>Line A
<decision>There is a continuity error."""

    parsed = parse_official_response(raw)

    assert parsed.cont_error == 1.0
    assert "missing_response_close_tag" in parsed.parsing_failures
    assert "missing_explanation_close_tag" in parsed.parsing_failures
    assert "missing_error_lines_close_tag" in parsed.parsing_failures
    assert "missing_contradicted_lines_tag" in parsed.parsing_failures
    assert "missing_decision_close_tag" in parsed.parsing_failures


def test_empty_response_defaults_to_no_error_and_records_failure() -> None:
    parsed = parse_official_response("")

    assert parsed.cont_error == 0.0
    assert parsed.cont_error_expl == "No continuity error found"
    assert parsed.cont_error_lines == ()
    assert parsed.contradicted_lines == ()
    assert parsed.parsing_failures == ("empty_response", "missing_response_tag")


def test_unexpected_decision_text_preserves_official_positive_label() -> None:
    raw = """<response>
<explanation>Unclear.</explanation>
<error_lines></error_lines>
<contradicted_lines></contradicted_lines>
<decision>Maybe.</decision>
</response>"""

    parsed = parse_official_response(raw)

    assert parsed.cont_error == 1.0
    assert "unexpected_decision_text" in parsed.parsing_failures


def test_additional_unwanted_text_is_recorded_without_blocking_parse() -> None:
    raw = """Preamble
<response>
<explanation>Consistent.</explanation>
<error_lines></error_lines>
<contradicted_lines></contradicted_lines>
<decision>No continuity error found.</decision>
</response>
Postscript"""

    parsed = parse_official_response(raw)

    assert parsed.cont_error == 0.0
    assert "text_before_response_tag" in parsed.parsing_failures
    assert "text_after_response_tag" in parsed.parsing_failures


def test_inference_record_preserves_minimum_output_fields() -> None:
    build = get_benchmark_strategy("vanilla").build_prompt(EXAMPLE)
    parsed = parse_official_response("<response><decision>No continuity error found.</decision></response>")

    record = InferenceRecord(
        example_id="fixture-1",
        subset="flawed_fictions",
        model="qwen3:4b",
        strategy=build.strategy,
        prompt_version=build.prompt_version,
        prompt_hash=build.prompt_hash,
        raw_response="raw text",
        parsed_prediction=parsed,
        thinking="separate native thinking",
    )

    assert record.example_id == "fixture-1"
    assert record.subset == "flawed_fictions"
    assert record.prompt_hash == build.prompt_hash
    assert record.raw_response == "raw text"
    assert record.thinking == "separate native thinking"
    assert record.intermediate_stage_outputs == ()
