#!/usr/bin/env python3
"""Validate the repository's authored agent-skill metadata."""

from __future__ import annotations

import re
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SKILLS = ROOT / "skills"
NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def load_frontmatter(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise ValueError("missing opening YAML frontmatter delimiter")
    try:
        raw, _body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise ValueError("missing closing YAML frontmatter delimiter") from exc
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError("frontmatter must be a mapping")
    return data


def validate(skill_dir: Path) -> list[str]:
    errors = []
    skill_path = skill_dir / "SKILL.md"
    agent_path = skill_dir / "agents" / "openai.yaml"
    try:
        frontmatter = load_frontmatter(skill_path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return [f"{skill_path}: {exc}"]

    name = frontmatter.get("name")
    description = frontmatter.get("description")
    if name != skill_dir.name or not isinstance(name, str) or not NAME.fullmatch(name):
        errors.append(f"{skill_path}: name must match directory and use kebab-case")
    if not isinstance(description, str) or not description.strip():
        errors.append(f"{skill_path}: description is required")
    elif len(description) > 1024:
        errors.append(f"{skill_path}: description exceeds 1024 characters")

    try:
        agent = yaml.safe_load(agent_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        errors.append(f"{agent_path}: {exc}")
        return errors
    interface = agent.get("interface") if isinstance(agent, dict) else None
    if not isinstance(interface, dict):
        errors.append(f"{agent_path}: interface mapping is required")
        return errors
    for key in ("display_name", "short_description", "default_prompt"):
        if not isinstance(interface.get(key), str) or not interface[key].strip():
            errors.append(f"{agent_path}: interface.{key} is required")
    prompt = interface.get("default_prompt", "")
    if isinstance(name, str) and f"${name}" not in prompt:
        errors.append(f"{agent_path}: default_prompt must mention ${name}")
    return errors


def main() -> int:
    skill_dirs = sorted(path for path in SKILLS.iterdir() if path.is_dir())
    if not skill_dirs:
        print("no skills found")
        return 1
    errors = [error for path in skill_dirs for error in validate(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"validated {len(skill_dirs)} skill(s): " + ", ".join(path.name for path in skill_dirs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
