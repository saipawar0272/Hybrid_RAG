
import pytest

from langchain_core.messages import AIMessage

import src.cli as cli


class FakeGraph:

    def __init__(self):
        self.calls = []

    def invoke(self, input_data, config):
        self.calls.append({
            "input": input_data,
            "config": config
        })

        return {
            "messages": [
                AIMessage(content="Mock assistant response")
            ]
        }


def setup_fake_cli(monkeypatch):
    fake_graph = FakeGraph()

    monkeypatch.setattr(
        cli,
        "build_graph",
        lambda: fake_graph
    )

    return fake_graph


def test_cli_uses_default_thread_id(monkeypatch, capsys):

    fake_graph = setup_fake_cli(monkeypatch)

    inputs = iter([
        "",              # Default thread ID
        "Hello assistant",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs)
    )

    cli.main()

    assert len(fake_graph.calls) == 1

    call = fake_graph.calls[0]

    assert call["config"]["configurable"]["thread_id"] == "demo-1"

    assert call["input"]["messages"][0].content == "Hello assistant"

    output = capsys.readouterr().out

    assert "Mock assistant response" in output


def test_cli_uses_custom_thread_id(monkeypatch):

    fake_graph = setup_fake_cli(monkeypatch)

    inputs = iter([
        "customer-123",
        "What is RAG?",
        "quit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs)
    )

    cli.main()

    assert len(fake_graph.calls) == 1

    call = fake_graph.calls[0]

    assert call["config"]["configurable"]["thread_id"] == (
        "customer-123"
    )


def test_cli_skips_empty_questions(monkeypatch):

    fake_graph = setup_fake_cli(monkeypatch)

    inputs = iter([
        "test-thread",
        "",
        "   ",
        "Hello",
        "exit"
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs)
    )

    cli.main()

    assert len(fake_graph.calls) == 1

    assert (
        fake_graph.calls[0]["input"]["messages"][0].content
        == "Hello"
    )


@pytest.mark.parametrize("exit_command", ["exit", "quit", "EXIT", "QUIT"])
def test_cli_terminates_on_exit_commands(
    monkeypatch,
    exit_command
):

    fake_graph = setup_fake_cli(monkeypatch)

    inputs = iter([
        "test-thread",
        exit_command
    ])

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt="": next(inputs)
    )

    cli.main()

    assert fake_graph.calls == []
