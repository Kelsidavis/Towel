"""Template generator skill — scaffold project files and boilerplate."""

from __future__ import annotations
from pathlib import Path
from typing import Any
from towel.skills.base import Skill, ToolDefinition

TEMPLATES = {
    "python-script": ('main.py', '''#!/usr/bin/env python3
"""{{description}}"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="{{description}}")
    parser.add_argument("input", help="Input file")
    parser.add_argument("-o", "--output", default="-", help="Output file")
    args = parser.parse_args()

    # TODO: implement
    print(f"Processing {args.input}")


if __name__ == "__main__":
    main()
'''),
    "python-package": ('__init__.py', '''"""{{name}} — {{description}}"""

__version__ = "0.1.0"
'''),
    "dockerfile": ('Dockerfile', '''FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "-m", "{{name}}"]
'''),
    "github-action": ('.github/workflows/ci.yml', '''name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - run: pip install -e ".[dev]"
      - run: pytest
'''),
    "makefile": ('Makefile', '''.PHONY: install test lint run clean

install:
\tpip install -e ".[dev]"

test:
\tpytest -v

lint:
\truff check .

run:
\tpython -m {{name}}

clean:
\trm -rf build dist *.egg-info __pycache__ .pytest_cache
'''),
    "readme": ('README.md', '''# {{name}}

{{description}}

## Quick Start

```bash
pip install -e .
```

## Usage

```bash
{{name}} --help
```

## License

MIT
'''),
    "gitignore-python": ('.gitignore', '''__pycache__/
*.pyc
*.pyo
.venv/
venv/
dist/
build/
*.egg-info/
.pytest_cache/
.ruff_cache/
.mypy_cache/
.env
*.db
'''),
    "fastapi": ('app.py', '''"""{{name}} — {{description}}"""

from fastapi import FastAPI

app = FastAPI(title="{{name}}", description="{{description}}")


@app.get("/")
async def root():
    return {"message": "Hello from {{name}}"}


@app.get("/health")
async def health():
    return {"status": "ok"}
'''),
}


class TemplateGenSkill(Skill):
    @property
    def name(self) -> str: return "scaffold"
    @property
    def description(self) -> str: return "Generate project scaffolding and boilerplate files"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="scaffold_list", description="List available templates",
                parameters={"type":"object","properties":{}}),
            ToolDefinition(name="scaffold_generate", description="Generate a file from a template",
                parameters={"type":"object","properties":{
                    "template":{"type":"string","description":f"Template name: {', '.join(TEMPLATES.keys())}"},
                    "name":{"type":"string","description":"Project/module name"},
                    "description":{"type":"string","description":"Short description"},
                    "output_dir":{"type":"string","description":"Output directory (default: cwd)"},
                },"required":["template"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "scaffold_list": return self._list()
            case "scaffold_generate": return self._generate(
                arguments["template"], arguments.get("name","myproject"),
                arguments.get("description","A new project"), arguments.get("output_dir","."))
            case _: return f"Unknown tool: {tool_name}"

    def _list(self) -> str:
        lines = ["Available templates:"]
        for name, (filename, _) in TEMPLATES.items():
            lines.append(f"  {name} -> {filename}")
        return "\n".join(lines)

    def _generate(self, template: str, name: str, description: str, output_dir: str) -> str:
        if template not in TEMPLATES:
            return f"Unknown template: {template}. Available: {', '.join(TEMPLATES.keys())}"
        filename, content = TEMPLATES[template]
        rendered = content.replace("{{name}}", name).replace("{{description}}", description)
        target = Path(output_dir).expanduser() / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            return f"File already exists: {target}"
        target.write_text(rendered, encoding="utf-8")
        return f"Created: {target} ({len(rendered)} bytes)"
