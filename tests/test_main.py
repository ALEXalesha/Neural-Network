import sys

import pytest

import main


@pytest.fixture(autouse=True)
def keep_sys(monkeypatch):
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    monkeypatch.setattr(sys, "path", list(sys.path))


@pytest.mark.parametrize("argv", [[], ["--foo"], ["some.txt"]])
def test_gui_mode_is_default(argv):
    assert main.run_child(argv) is None


def test_pyfile_runs_script_next_to_its_module(tmp_path, capsys):
    (tmp_path / "solution.py").write_text("def add(a, b):\n    return a + b\n")
    script = tmp_path / "t.py"
    script.write_text("from solution import add\nassert add(2, 3) == 5\nprint('ALL TESTS PASS')\n")
    assert main.run_child(["--pyfile", str(script)]) == 0
    assert "ALL TESTS PASS" in capsys.readouterr().out


@pytest.mark.parametrize("src,code", [("raise ValueError('x')", 1), ("import sys; sys.exit(3)", 3),
                                      ("import sys; sys.exit('msg')", 1), ("input()", 1)])
def test_pyfile_exit_codes(tmp_path, monkeypatch, src, code):
    monkeypatch.setattr(sys, "stdin", open(__import__("os").devnull))
    script = tmp_path / "t.py"
    script.write_text(src)
    assert main.run_child(["--pyfile", str(script)]) == code


def test_run_rejects_unknown_module():
    assert main.run_child(["--run", "os"]) == 2
    assert main.run_child(["--run"]) == 2
