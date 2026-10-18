"""Prompt strategies for the released FLAWEDFICTIONS benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol

from fyp.data.schema import FlawedFictionsExample


PROMPT_DIR = Path("prompts/benchmark")
EXPERIMENTAL_PROMPT_DIR = Path("prompts/experimental")


class PromptableExample(Protocol):
    story: str


@dataclass(frozen=True)
class PromptStage:
    """One model-call stage in a benchmark prompt strategy."""

    index: int
    name: str
    role: str
    content: str
    prompt_hash: str


@dataclass(frozen=True)
class PromptBuildResult:
    """Rendered prompt plus metadata needed for reproducible inference."""

    strategy: str
    prompt_version: str
    template_path: Path
    template_hash: str
    prompt_hash: str
    stages: tuple[PromptStage, ...]

    @property
    def prompt(self) -> str:
        """Content of the single released benchmark model call."""

        return self.stages[0].content

    @property
    def chat_messages(self) -> list[dict[str, str]]:
        """Official code sends the rendered prompt as one user message."""

        return [{"role": stage.role, "content": stage.content} for stage in self.stages]


@dataclass(frozen=True)
class BenchmarkPromptStrategy:
    """A released FLAWEDFICTIONS benchmark prompt condition."""

    name: str
    prompt_version: str
    template_path: Path
    examples_path: Path | None = None

    @property
    def num_model_calls(self) -> int:
        return 1

    def build_prompt(self, example: FlawedFictionsExample | Mapping[str, object]) -> PromptBuildResult:
        story = _get_story(example)
        template_bytes = self.template_path.read_bytes()
        template = self.template_path.read_text(encoding="utf-8")
        template_hash = _sha256_bytes(template_bytes)
        if "{examples}" in template:
            if self.examples_path is None:
                raise ValueError(f"Strategy {self.name!r} requires examples_path")
            prompt = template.format(story=story, examples=_load_fewshot_examples(self.examples_path))
        else:
            prompt = template.format(story=story)
        prompt_hash = _sha256_text(prompt)
        stage = PromptStage(
            index=1,
            name="detect_continuity_error",
            role="user",
            content=prompt,
            prompt_hash=prompt_hash,
        )
        return PromptBuildResult(
            strategy=self.name,
            prompt_version=self.prompt_version,
            template_path=self.template_path,
            template_hash=template_hash,
            prompt_hash=prompt_hash,
            stages=(stage,),
        )


STRATEGIES: dict[str, BenchmarkPromptStrategy] = {
    "vanilla": BenchmarkPromptStrategy(
        name="vanilla",
        prompt_version="vanilla_v1",
        template_path=PROMPT_DIR / "vanilla_v1.txt",
    ),
    "cot": BenchmarkPromptStrategy(
        name="cot",
        prompt_version="cot_v1",
        template_path=PROMPT_DIR / "cot_v1.txt",
    ),
    "fewshot": BenchmarkPromptStrategy(
        name="fewshot",
        prompt_version="fewshot_v1",
        template_path=PROMPT_DIR / "fewshot_v1.txt",
        examples_path=PROMPT_DIR / "fewshot_examples_v1.json",
    ),
    "evidence_first": BenchmarkPromptStrategy(
        name="evidence_first",
        prompt_version="evidence_first_v1",
        template_path=EXPERIMENTAL_PROMPT_DIR / "evidence_first_v1.txt",
    ),
    "verify_revise": BenchmarkPromptStrategy(
        name="verify_revise",
        prompt_version="verify_revise_v1",
        template_path=EXPERIMENTAL_PROMPT_DIR / "verify_revise_v1.txt",
    ),
    "structured_check": BenchmarkPromptStrategy(
        name="structured_check",
        prompt_version="structured_check_v1",
        template_path=EXPERIMENTAL_PROMPT_DIR / "structured_check_v1.txt",
    ),
}


VERSIONED_STRATEGIES = {
    ("evidence_first", "evidence_first_v2"): BenchmarkPromptStrategy(
        name="evidence_first", prompt_version="evidence_first_v2",
        template_path=EXPERIMENTAL_PROMPT_DIR / "evidence_first_v2.txt",
    ),
    ("verify_revise", "verify_revise_v2"): BenchmarkPromptStrategy(
        name="verify_revise", prompt_version="verify_revise_v2",
        template_path=EXPERIMENTAL_PROMPT_DIR / "verify_revise_v2.txt",
    ),
}


def get_benchmark_strategy(name: str, version: str | None = None) -> BenchmarkPromptStrategy:
    """Return one of the released benchmark prompt strategies."""

    try:
        strategy = STRATEGIES[name]
    except KeyError as exc:
        allowed = ", ".join(sorted(STRATEGIES))
        raise ValueError(f"Unknown prompt strategy {name!r}; expected one of: {allowed}") from exc
    if version is None or version == strategy.prompt_version:
        return strategy
    try:
        return VERSIONED_STRATEGIES[name, version]
    except KeyError as exc:
        raise ValueError(f"Unknown prompt version {version!r} for strategy {name!r}") from exc


def _load_fewshot_examples(path: Path) -> str:
    examples = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(examples, list):
        raise ValueError(f"Few-shot examples file {path} must contain a list")
    rendered = ""
    for idx, example in enumerate(examples):
        if not isinstance(example, dict):
            raise ValueError(f"Few-shot example {idx} must be an object")
        input_text = example.get("input")
        output_text = example.get("output")
        if not isinstance(input_text, str) or not isinstance(output_text, str):
            raise ValueError(f"Few-shot example {idx} must contain string input/output")
        rendered += f"<example{idx+1}>\n{input_text}\n{output_text}\n</example{idx+1}>\n\n"
    return rendered


def _get_story(example: FlawedFictionsExample | Mapping[str, object]) -> str:
    if isinstance(example, FlawedFictionsExample):
        return example.story
    story = example.get("story")
    if not isinstance(story, str):
        raise ValueError("Prompt example must provide a string 'story' field")
    return story


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()
