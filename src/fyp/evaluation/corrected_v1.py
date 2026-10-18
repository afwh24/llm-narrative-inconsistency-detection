"""Opt-in offline scoring correction; frozen benchmark modules stay unchanged.

Explicit label recognition is an audit policy, not claimed official parity.
See configs/experiments/main100_scoring_correction_v1.json for its grammar.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

from fyp.evaluation.localization import (
    HTML_MARKERS, LocalizationScore, listify_lines, preprocess_sent, sentence_tokenize,
)


@dataclass(frozen=True)
class CorrectedPrediction:
    label: int | None
    decision: str | None
    invalid_reasons: tuple[str, ...]
    field_warnings: tuple[str, ...]
    error_lines: tuple[str, ...]
    contradicted_lines: tuple[str, ...]


def _field(raw: str, name: str) -> tuple[str, list[str]]:
    """Prevent a missing close tag from absorbing later response fields."""
    opening = f"<{name}>"
    if opening not in raw:
        return "", [f"missing_{name}_tag"]
    tail = raw.split(opening, 1)[1]
    boundary = re.search(
        r"</?(?:response|explanation|error_lines|contradicted_lines|decision)>|</(?!li>|p>|m>)[a-z_]+>",
        tail,
    )
    warnings = []
    if boundary is None or boundary.group() != f"</{name}>":
        warnings.append(f"bounded_recovery_{name}_missing_or_wrong_close")
    if raw.count(opening) != 1:
        warnings.append(f"duplicate_{name}_tag")
    return tail[:boundary.start()] if boundary else tail, warnings


def parse_response(raw: Any, policy: Mapping[str, Any], *, inference_failure=False) -> CorrectedPrediction:
    reasons: list[str] = []
    warnings: list[str] = []
    decision = None
    label = None
    if inference_failure:
        reasons.append("inference_failure")
    if not isinstance(raw, str):
        reasons.append("missing_raw_response")
        raw = ""
    elif not raw.strip():
        reasons.append("empty_raw_response")
    for name in ("response", "explanation"):
        if f"<{name}>" not in raw or f"</{name}>" not in raw:
            warnings.append(f"malformed_{name}_wrapper")
    matches = list(re.finditer(r"<decision>(.*?)</decision>", raw, flags=re.DOTALL))
    if raw.count("<decision>") != 1 or raw.count("</decision>") != 1 or len(matches) != 1:
        reasons.append("missing_or_malformed_decision_tags")
    else:
        decision = matches[0].group(1).strip()
        normalized = re.sub(r"\s+", " ", decision).lower()
        if not normalized:
            reasons.append("empty_decision")
        elif re.search(r"</?[a-z_]+>", decision):
            reasons.append("nested_decision_tags")
        else:
            negative = any(re.search(p, normalized) for p in policy["negative_patterns"])
            positive = any(re.search(p, normalized) for p in policy["positive_patterns"])
            # A second, opposing explicit sentence is ambiguous, not a first-match win.
            sentences = re.split(r"(?<=[.!])\s+", normalized)
            claims = {v for s in sentences for v, key in ((0, "negative_patterns"), (1, "positive_patterns"))
                      if any(re.search(p, s) for p in policy[key])}
            if len(claims) > 1:
                reasons.append("conflicting_decision_claims")
            elif negative == positive:
                reasons.append("unrecognized_or_ambiguous_decision")
            else:
                label = int(positive)
    fields = []
    for name in ("error_lines", "contradicted_lines"):
        value, issues = _field(raw, name)
        warnings.extend(issues)
        # The legacy parser splits on newlines (not HTML list boundaries).
        fields.append(tuple(line.strip().replace("-", "").replace("*", "").strip()
                            for line in value.splitlines()
                            if line.strip().replace("-", "").replace("*", "").strip()))
    return CorrectedPrediction(None if reasons else label, decision, tuple(reasons), tuple(warnings), *fields)


def expand_evidence(story: str, snippets: Sequence[str]) -> dict[str, Any]:
    """Legacy containment with the minimal nonempty-fragment guard, plus trace."""
    cleaned_story = re.sub(r"\[.*?\]", "", story)
    for marker in HTML_MARKERS:
        cleaned_story = cleaned_story.replace(marker, " ")
    sentences = sentence_tokenize(cleaned_story)
    matched, rejected = [], []
    for snippet in snippets:
        for fragment in sentence_tokenize(re.sub(r"\[.*?\]", "", snippet)):
            normalized = preprocess_sent(fragment)
            if not normalized:
                rejected.append(fragment)
                continue
            matched.extend(line for line in sentences if normalized in preprocess_sent(line))
    return {"sentences": matched, "rejected_empty_fragments": rejected}


def score_localization(example, label, error_lines, contradicted_lines):
    """Retain correct-negative credit and side-specific non-strict overlap."""
    gold_error = expand_evidence(example.story, listify_lines(example.cont_error_lines_raw))
    gold_contradicted = expand_evidence(example.story, listify_lines(example.contradicted_lines_raw))
    error = expand_evidence(example.story, error_lines)
    contradicted = expand_evidence(example.story, contradicted_lines)
    trace = {"gold_error": gold_error, "gold_contradicted": gold_contradicted,
             "predicted_error": error, "predicted_contradicted": contradicted}
    a = b = 0.0
    if label == example.cont_error:
        if label == 0:
            a = b = 1.0
        else:
            a = float(bool(set(error["sentences"]) & set(gold_error["sentences"])))
            b = float(bool(set(contradicted["sentences"]) & set(gold_contradicted["sentences"])))
    return LocalizationScore(a*b, a, b, tuple(error["sentences"]), tuple(contradicted["sentences"]),
                             tuple(gold_error["sentences"]), tuple(gold_contradicted["sentences"])), trace


def detection_metrics(gold: Sequence[int], predicted: Sequence[int | None]) -> dict[str, Any]:
    """Abstention-aware all-scope metrics; invalids are never assigned a class."""
    if len(gold) != len(predicted) or any(x not in (0, 1) for x in gold):
        raise ValueError("Mismatched scope or nonbinary gold")
    if any(x not in (0, 1, None) for x in predicted):
        raise ValueError("Predictions must be binary or None")
    count = lambda a,b: sum(y == a and p == b for y,p in zip(gold,predicted))
    tp, fp, tn, fn = count(1,1), count(0,1), count(0,0), count(1,0)
    ip, ine = count(1,None), count(0,None)
    n, pos, neg = len(gold), sum(gold), len(gold)-sum(gold)
    div = lambda a,b: a/b if b else 0.0
    f1 = div(2*tp, 2*tp+fp+fn+ip)
    negative_f1 = div(2*tn, 2*tn+fp+fn+ine)
    result = dict(total=n, valid=n-ip-ine, invalid=ip+ine, gold_positive=pos, gold_negative=neg,
                  tp=tp, fp=fp, tn=tn, fn=fn, invalid_gold_positive=ip, invalid_gold_negative=ine,
                  predicted_positive=tp+fp, predicted_positive_rate=div(tp+fp,n),
                  accuracy=div(tp+tn,n), precision=div(tp,tp+fp), recall=div(tp,pos),
                  f1=f1, negative_f1=negative_f1, macro_f1=(f1+negative_f1)/2,
                  specificity=div(tn,neg), false_positive_rate=div(fp,neg),
                  balanced_accuracy=(div(tp,pos)+div(tn,neg))/2,
                  coverage=div(n-ip-ine,n))
    return result


def aggregate(rows):
    gold, pred = [r["gold"] for r in rows], [r["predicted"] for r in rows]
    result = detection_metrics(gold, pred)
    valid = [r for r in rows if r["predicted"] is not None]
    result["valid_only_diagnostic"] = detection_metrics([r["gold"] for r in valid], [r["predicted"] for r in valid])
    n, pos = len(rows), sum(gold)
    result.update(ceeval_full=sum(r["full"] for r in rows)/n if n else 0.0,
                  ceeval_pos=sum(r["full"] for r in rows if r["gold"]==1)/pos if pos else 0.0,
                  cont_error_lines=sum(r["error_side"] for r in rows)/n if n else 0.0,
                  contradicted_lines=sum(r["contradicted_side"] for r in rows)/n if n else 0.0)
    return result
