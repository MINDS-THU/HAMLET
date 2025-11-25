from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hamlet.tools.visual_qa.visual_qa import Visualizer


@pytest.fixture
def image_file(working_dir: Path) -> Path:
    target = working_dir / "sample.png"
    target.write_bytes(b"\x89PNG\r\n")
    return target


def test_visualizer_returns_model_answer(monkeypatch: pytest.MonkeyPatch, image_file: Path, working_dir: Path) -> None:
    def fake_post(url: str, headers: dict[str, str], json: dict[str, object]):  # noqa: ARG001
        assert "Bearer" in headers["Authorization"]
        return SimpleNamespace(json=lambda: {"choices": [{"message": {"content": "answer"}}]})

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr("hamlet.tools.visual_qa.visual_qa.requests.post", fake_post)

    tool = Visualizer(str(working_dir))
    result = tool(image_path=image_file.name, question="What is shown?")

    assert result == "answer"


def test_visualizer_adds_caption_when_question_missing(monkeypatch: pytest.MonkeyPatch, image_file: Path, working_dir: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(
        "hamlet.tools.visual_qa.visual_qa.requests.post",
        lambda *args, **kwargs: SimpleNamespace(json=lambda: {"choices": [{"message": {"content": "desc"}}]}),
    )

    tool = Visualizer(str(working_dir))
    message = tool(image_path=image_file.name)

    assert message.startswith("You did not provide")
    assert "desc" in message


def test_visualizer_validates_paths(working_dir: Path) -> None:
    tool = Visualizer(str(working_dir))
    outside = Path(working_dir).parent / "other.png"
    outside.write_bytes(b"fake")

    error = tool(image_path="../other.png", question="?" )

    assert "Access outside" in error


def test_visualizer_handles_missing_file(working_dir: Path) -> None:
    tool = Visualizer(str(working_dir))
    result = tool(image_path="missing.png", question="?")
    assert "does not exist" in result
