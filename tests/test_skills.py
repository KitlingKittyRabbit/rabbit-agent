"""Skill 发现与提示拼装（项目级覆盖全局，不自动进上下文）。"""

from pathlib import Path

import agent.core.skills as skills


def _write_skill(base: Path, name: str, body: str) -> None:
    d = base / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body, encoding="utf-8")


def test_list_skills_project_overrides_global(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "global-skills"
    monkeypatch.setattr(skills, "GLOBAL_SKILL_DIR", global_dir)
    project = tmp_path / "proj"
    _write_skill(global_dir, "alpha", "---\ndescription: 全局版\n---\n正文")
    _write_skill(global_dir, "beta", "# 标题\n第一行描述")
    _write_skill(project / ".agent" / "skills", "alpha", "---\ndescription: 项目版\n---\n正文")

    found = {s["name"]: s for s in skills.list_skills(project)}
    assert sorted(found) == ["alpha", "beta"]
    assert found["alpha"]["scope"] == "project" and found["alpha"]["description"] == "项目版"
    assert found["beta"]["description"] == "第一行描述"       # 无 front-matter 取首行


def test_invalid_names_and_missing_files_skipped(tmp_path, monkeypatch) -> None:
    global_dir = tmp_path / "g"
    monkeypatch.setattr(skills, "GLOBAL_SKILL_DIR", global_dir)
    _write_skill(global_dir, "ok", "正文")
    (global_dir / "bad name").mkdir(parents=True, exist_ok=True)
    (global_dir / "bad name" / "SKILL.md").write_text("x", encoding="utf-8")
    (global_dir / "empty").mkdir(parents=True, exist_ok=True)     # 无 SKILL.md

    assert [s["name"] for s in skills.list_skills(None)] == ["ok"]


def test_find_skill_and_prompt(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(skills, "GLOBAL_SKILL_DIR", tmp_path / "g")
    project = tmp_path / "p"
    _write_skill(project / ".agent" / "skills", "demo",
                 "---\ndescription: 演示\n---\n步骤一\n步骤二")

    spec = skills.find_skill(project, "demo")
    assert spec is not None and spec["scope"] == "project"
    assert skills.find_skill(project, "../escape") is None
    assert skills.find_skill(project, "nope") is None

    prompt = skills.skill_prompt(spec, "帮我做点事")
    assert prompt.startswith("[技能 demo]") and "步骤一" in prompt
    assert prompt.rstrip().endswith("帮我做点事") and "[用户指令]" in prompt
    assert "[用户指令]" not in skills.skill_prompt(spec, "")
