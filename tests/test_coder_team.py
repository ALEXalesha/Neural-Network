import re
import textwrap

import pytest
from hypothesis import given, strategies as st

import coder_team_v2 as ct
import paths


@pytest.mark.parametrize("text", [
    "NOT APPROVED. 1. off-by-one", "The code is not approved yet: bug in loop",
    "Не одобрено: есть баги", "Unapproved - see issues", "I cannot approve this. LGTM would be wrong",
])
def test_negated_approval_is_rejected(text):
    assert not ct.is_approved(text)


@pytest.mark.parametrize("text", ["APPROVED", "**APPROVED** - clean code", "LGTM", "Одобрено, замечаний нет",
                                  "Looks good, no bugs."])
def test_plain_approval(text):
    assert ct.is_approved(text)


top_level = st.sampled_from([
    "import os", "x = 1", "def f():", "class A:", "while True:", "while\tx:", "if __name__ == '__main__':",
    "input('q')", "print(1)", "for i in range(3):", "@decorator", "",
])
body = st.sampled_from(["    return 1", "    pass", "    x += 1", "        input()", "    while y:"])
first_line = top_level.filter(bool)
lines = st.tuples(first_line, st.lists(top_level | body, max_size=40)).map(lambda t: [t[0], *t[1]])


@given(lines)
def test_extract_defs_only(src_lines):
    src = "\n".join(src_lines)
    out = ct._extract_defs_only(src)
    out_lines = out.splitlines()
    it = iter(textwrap.dedent(src).splitlines())
    assert all(any(line == s for s in it) for line in out_lines), "output must be an ordered subset"
    for line in out_lines:
        if line and not line[0].isspace():
            assert not line.startswith(("while ", "while\t", "if __name__", "input("))
    assert ct._extract_defs_only(out) == out


@given(st.text(max_size=80).filter(lambda s: "```" not in s), st.sampled_from(["", "python", "py", "c++"]),
       st.text(max_size=80).filter(lambda s: "```" not in s))
def test_extract_code_takes_first_fence(before, lang, code):
    text = f"{before}\n```{lang}\n{code}\n```\ntrailing ```x\nignored\n```"
    assert ct.extract_code(text) == code.strip()


stdlib_imports = st.sampled_from(["import os", "import sys", "from typing import List", "import unittest",
                                  "from collections import Counter", "import numpy as np", "import pytest"])
user_imports = st.from_regex(r"\A(import [a-z_]{3,10}|from [a-z_]{3,10} import x)\Z").filter(
    lambda s: s.split()[1] not in ct._SAFE_MODULES)


@given(st.lists(stdlib_imports | user_imports, max_size=10))
def test_normalize_imports(import_lines):
    src = "\n".join(import_lines)
    out = ct._normalize_imports(src)
    for a, b in zip(import_lines, out.splitlines()):
        mod = a.split()[1]
        if mod in ct._SAFE_MODULES:
            assert a == b
        else:
            assert re.match(r"(from solution import|import solution)", b), b
    assert ct._normalize_imports(out) == out


def test_frozen_commands(monkeypatch):
    monkeypatch.setattr(paths, "FROZEN", True)
    assert paths.script_cmd("coder_team_v2", "--review", 3)[1:] == ["--run", "coder_team_v2", "--review", "3"]
    assert paths.pyfile_cmd("t.py")[1:] == ["--pyfile", "t.py"]
    monkeypatch.setattr(paths, "FROZEN", False)
    assert paths.script_cmd("coder_team")[1:3] == ["-u", str(paths.APP_DIR / "coder_team.py")]


def test_tester_runs_generated_tests(monkeypatch):
    replies = iter(["```python\nfrom solution import add\nassert add(2, 3) == 5\nprint('ALL TESTS PASS')\n```",
                    "```python\nfrom solution import add\nassert add(2, 2) == 5\n```"])
    monkeypatch.setattr(ct, "ask", lambda *a, **kw: next(replies))
    code = "def add(a, b):\n    return a + b\n\nwhile True:\n    pass\n"
    assert ct.agent_tester("add", code)[1] is True
    assert ct.agent_tester("add", code)[1] is False
