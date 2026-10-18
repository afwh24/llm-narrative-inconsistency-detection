"""Official-style parser for FLAWEDFICTIONS benchmark responses."""

from __future__ import annotations

from dataclasses import dataclass


NO_ERROR_PHRASE = "no continuity error found"


@dataclass(frozen=True)
class ParsedPrediction:
    """Parsed model output using the released benchmark parser semantics."""

    cont_error: float
    cont_error_expl_detailed: str
    cont_error_expl: str
    cont_error_expl_short: str
    cont_error_lines: tuple[str, ...]
    contradicted_lines: tuple[str, ...]
    parsing_failures: tuple[str, ...] = ()

    def to_official_result(self) -> dict[str, object]:
        """Return a dict shaped like the released evaluator expects."""

        return {
            "cont_error": self.cont_error,
            "cont_error_expl_detailed": self.cont_error_expl_detailed,
            "cont_error_expl": self.cont_error_expl,
            "cont_error_expl_short": self.cont_error_expl_short,
            "cont_error_lines": list(self.cont_error_lines),
            "contradicted_lines": list(self.contradicted_lines),
        }


def parse_official_response(response_content: str) -> ParsedPrediction:
    """Parse a response with the released FLAWEDFICTIONS string-split rules.

    The label decision mirrors `external/flawedfictions/src/detect_cont_errors.py`:
    if a `<decision>` section exists, any decision that does not contain the
    phrase "no continuity error found" is treated as positive.
    """

    failures: list[str] = []
    if response_content == "":
        failures.append("empty_response")

    if "<response>" not in response_content:
        failures.append("missing_response_tag")
        return ParsedPrediction(
            cont_error=0.0,
            cont_error_expl_detailed=response_content,
            cont_error_expl="No continuity error found",
            cont_error_expl_short="No continuity error found",
            cont_error_lines=(),
            contradicted_lines=(),
            parsing_failures=tuple(failures),
        )

    _record_outer_text_failures(response_content, failures)

    cont_error_expl, explanation_failures = _extract_section(
        response_content,
        "explanation",
        missing_default=response_content,
    )
    failures.extend(explanation_failures)

    cont_error_lines, error_line_failures = _extract_section(
        response_content,
        "error_lines",
        missing_default="No error lines found",
    )
    failures.extend(error_line_failures)

    contradicted_lines, contradicted_failures = _extract_section(
        response_content,
        "contradicted_lines",
        missing_default="No contradicted lines found",
    )
    failures.extend(contradicted_failures)

    if "<decision>" not in response_content:
        cont_error = 0.0
        cont_error_expl_short = "No decision found"
        failures.append("missing_decision_tag")
    elif "</decision>" not in response_content:
        decision = response_content.split("<decision>", 1)[1].strip()
        cont_error_expl_short = decision
        cont_error = _decision_to_label(decision, failures)
        failures.append("missing_decision_close_tag")
    else:
        decision = response_content.split("<decision>", 1)[1].split("</decision>", 1)[0].strip()
        cont_error_expl_short = decision
        cont_error = _decision_to_label(decision, failures)

    return ParsedPrediction(
        cont_error=cont_error,
        cont_error_expl_detailed=response_content,
        cont_error_expl=cont_error_expl,
        cont_error_expl_short=cont_error_expl_short,
        cont_error_lines=tuple(_clean_lines(cont_error_lines)),
        contradicted_lines=tuple(_clean_lines(contradicted_lines)),
        parsing_failures=tuple(failures),
    )


def _extract_section(
    response_content: str,
    tag: str,
    missing_default: str,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if open_tag not in response_content:
        failures.append(f"missing_{tag}_tag")
        return missing_default, failures
    if close_tag not in response_content:
        failures.append(f"missing_{tag}_close_tag")
        return response_content.split(open_tag, 1)[1].strip(), failures
    return response_content.split(open_tag, 1)[1].split(close_tag, 1)[0].strip(), failures


def _decision_to_label(decision: str, failures: list[str]) -> float:
    if decision == "":
        failures.append("empty_decision")
    lowered = decision.lower()
    if NO_ERROR_PHRASE in lowered:
        return 0.0
    if "continuity error" not in lowered:
        failures.append("unexpected_decision_text")
    return 1.0


def _clean_lines(lines: str) -> list[str]:
    split_lines = lines.split("\n")
    cleaned = [line.strip().replace("-", "").replace("*", "").strip() for line in split_lines]
    return [line for line in cleaned if line != ""]


def _record_outer_text_failures(response_content: str, failures: list[str]) -> None:
    before_response = response_content.split("<response>", 1)[0].strip()
    if before_response:
        failures.append("text_before_response_tag")
    if "</response>" in response_content:
        after_response = response_content.split("</response>", 1)[1].strip()
        if after_response:
            failures.append("text_after_response_tag")
    else:
        failures.append("missing_response_close_tag")
