"""Code snippet generator — common patterns and boilerplate for multiple languages."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition

SNIPPETS = {
    "python-class": '''class {name}:
    """{description}"""

    def __init__(self{params}):
{init_body}

    def __repr__(self) -> str:
        return f"{name}({repr_fields})"
''',
    "python-dataclass": '''from dataclasses import dataclass, field
from typing import Any

@dataclass
class {name}:
    """{description}"""
{fields}
''',
    "python-cli": '''import click

@click.command()
@click.argument("input")
@click.option("--output", "-o", default="-", help="Output file")
@click.option("--verbose", "-v", is_flag=True)
def main(input: str, output: str, verbose: bool) -> None:
    """{description}"""
    if verbose:
        click.echo(f"Processing {{input}}")
    # TODO: implement

if __name__ == "__main__":
    main()
''',
    "python-test": """import pytest


class Test{name}:
    @pytest.fixture
    def subject(self):
        return {name}()

    def test_creation(self, subject):
        assert subject is not None

    def test_basic_behavior(self, subject):
        # TODO: implement
        pass
""",
    "python-async": '''import asyncio

async def {name}({params}) -> {return_type}:
    """{description}"""
    # TODO: implement
    pass

async def main():
    result = await {name}()
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
''',
    "js-fetch": """async function {name}(url, options = {{}}) {{
  const response = await fetch(url, {{
    headers: {{ 'Content-Type': 'application/json', ...options.headers }},
    ...options,
  }});
  if (!response.ok) throw new Error(`HTTP ${{response.status}}`);
  return response.json();
}}
""",
    "js-express": """import express from 'express';

const app = express();
app.use(express.json());

app.get('/', (req, res) => {{
  res.json({{ message: '{description}' }});
}});

app.listen(3000, () => console.log('Listening on :3000'));
""",
    "sql-create-table": """CREATE TABLE {name} (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    -- TODO: add columns
);

CREATE INDEX idx_{name}_created_at ON {name}(created_at);
""",
    "docker-compose": """version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/{name}
    depends_on:
      - db

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: {name}
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
""",
    "github-pr-template": """## Summary
<!-- What does this PR do? -->

## Changes
-

## Test Plan
- [ ] Unit tests pass
- [ ] Manual testing done

## Screenshots
<!-- If applicable -->
""",
}


class SnippetGenSkill(Skill):
    @property
    def name(self) -> str:
        return "codegen"

    @property
    def description(self) -> str:
        return "Generate code snippets and common patterns for multiple languages"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="codegen_list",
                description="List available code snippet templates",
                parameters={"type": "object", "properties": {}},
            ),
            ToolDefinition(
                name="codegen_generate",
                description="Generate a code snippet from a template",
                parameters={
                    "type": "object",
                    "properties": {
                        "template": {
                            "type": "string",
                            "description": f"Template: {', '.join(SNIPPETS.keys())}",
                        },
                        "name": {
                            "type": "string",
                            "description": "Name (class/function/table name)",
                        },
                        "description": {"type": "string", "description": "Description"},
                        "params": {"type": "string", "description": "Parameters (optional)"},
                    },
                    "required": ["template", "name"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "codegen_list":
                return "Code templates:\n" + "\n".join(f"  {k}" for k in sorted(SNIPPETS.keys()))
            case "codegen_generate":
                return self._gen(
                    arguments["template"],
                    arguments["name"],
                    arguments.get("description", ""),
                    arguments.get("params", ""),
                )
            case _:
                return f"Unknown tool: {tool_name}"

    def _gen(self, template: str, name: str, desc: str, params: str) -> str:
        if template not in SNIPPETS:
            return f"Unknown template: {template}"
        code = SNIPPETS[template]
        # Basic substitutions
        code = code.replace("{name}", name)
        code = code.replace("{description}", desc or f"A {name}")
        code = code.replace("{params}", f", {params}" if params else "")
        code = code.replace("{return_type}", "Any")
        # Generate init body for class template
        if params:
            fields = [p.strip().split(":")[0].strip() for p in params.split(",")]
            code = code.replace("{init_body}", "\n".join(f"        self.{f} = {f}" for f in fields))
            code = code.replace("{repr_fields}", ", ".join(f"{f}={{self.{f}!r}}" for f in fields))
            code = code.replace("{fields}", "\n".join(f"    {f}: str" for f in fields))
        else:
            code = code.replace("{init_body}", "        pass")
            code = code.replace("{repr_fields}", "")
            code = code.replace("{fields}", "    pass")
        return code
