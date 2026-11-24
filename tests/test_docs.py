"""Keep the CLI docs in sync with the argument parser."""
from pathlib import Path

from quantbt.cli import build_parser

CLI_DOC = Path(__file__).resolve().parents[1] / "docs" / "cli.md"


def _run_parser():
    sub = next(a for a in build_parser()._actions if a.dest == "command")
    return sub.choices["run"]


def test_every_run_option_is_documented():
    text = CLI_DOC.read_text()
    missing = [
        flag
        for action in _run_parser()._actions
        for flag in action.option_strings
        if flag.startswith("--") and flag not in ("--help",) and f"`{flag}`" not in text
    ]
    assert not missing, f"undocumented options: {missing}"


def test_every_strategy_is_documented():
    text = CLI_DOC.read_text()
    strategy = next(a for a in _run_parser()._actions if a.dest == "strategy")
    missing = [name for name in strategy.choices if f"`{name}`" not in text]
    assert not missing, f"undocumented strategies: {missing}"
