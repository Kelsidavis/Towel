"""Placeholder data skill — generate test data, lorem ipsum, fake names."""

from __future__ import annotations

import random
import string
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_LOREM = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua Ut enim ad minim veniam quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur Excepteur sint occaecat cupidatat non proident sunt in culpa qui officia deserunt mollit anim id est laborum".split()

_FIRST_NAMES = ["Alice","Bob","Charlie","Diana","Eve","Frank","Grace","Hank","Iris","Jack","Karen","Leo","Mia","Noah","Olivia","Pete","Quinn","Rose","Sam","Tina","Uma","Vic","Wendy","Xena","Yuri","Zoe"]
_LAST_NAMES = ["Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez","Anderson","Taylor","Thomas","Moore","Jackson","Martin","Lee","Harris","Clark","Lewis"]
_DOMAINS = ["example.com","test.org","demo.io","sample.net","fake.dev"]


class PlaceholderSkill(Skill):
    @property
    def name(self) -> str: return "placeholder"
    @property
    def description(self) -> str: return "Generate test data — lorem ipsum, fake names, emails, sample JSON"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="lorem", description="Generate lorem ipsum text",
                parameters={"type":"object","properties":{
                    "words":{"type":"integer","description":"Number of words (default: 50)"},
                    "paragraphs":{"type":"integer","description":"Number of paragraphs (default: 1)"},
                }}),
            ToolDefinition(name="fake_users", description="Generate fake user data (name, email, age)",
                parameters={"type":"object","properties":{
                    "count":{"type":"integer","description":"Number of users (default: 5)"},
                    "format":{"type":"string","enum":["json","csv","table"],"description":"Output format"},
                },"required":[]}),
            ToolDefinition(name="fake_data", description="Generate random data of a specific type",
                parameters={"type":"object","properties":{
                    "type":{"type":"string","enum":["email","phone","ip","date","color","url","sentence"],"description":"Data type"},
                    "count":{"type":"integer","description":"How many (default: 5)"},
                },"required":["type"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "lorem": return self._lorem(arguments.get("words",50), arguments.get("paragraphs",1))
            case "fake_users": return self._users(arguments.get("count",5), arguments.get("format","json"))
            case "fake_data": return self._data(arguments["type"], arguments.get("count",5))
            case _: return f"Unknown tool: {tool_name}"

    def _lorem(self, words: int, paragraphs: int) -> str:
        result = []
        for _ in range(paragraphs):
            w = [random.choice(_LOREM) for _ in range(words)]
            w[0] = w[0].capitalize()
            result.append(" ".join(w) + ".")
        return "\n\n".join(result)

    def _users(self, count: int, fmt: str) -> str:
        import json
        users = []
        for _ in range(min(count, 100)):
            first = random.choice(_FIRST_NAMES)
            last = random.choice(_LAST_NAMES)
            users.append({
                "name": f"{first} {last}",
                "email": f"{first.lower()}.{last.lower()}@{random.choice(_DOMAINS)}",
                "age": random.randint(18, 75),
            })
        if fmt == "csv":
            lines = ["name,email,age"]
            for u in users: lines.append(f"{u['name']},{u['email']},{u['age']}")
            return "\n".join(lines)
        elif fmt == "table":
            lines = [f"{'Name':<20} {'Email':<35} {'Age':>3}"]
            lines.append("-" * 60)
            for u in users: lines.append(f"{u['name']:<20} {u['email']:<35} {u['age']:>3}")
            return "\n".join(lines)
        return json.dumps(users, indent=2)

    def _data(self, dtype: str, count: int) -> str:
        items = []
        for _ in range(min(count, 100)):
            match dtype:
                case "email":
                    items.append(f"{''.join(random.choices(string.ascii_lowercase, k=8))}@{random.choice(_DOMAINS)}")
                case "phone":
                    items.append(f"+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}")
                case "ip":
                    items.append(f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}")
                case "date":
                    items.append(f"20{random.randint(20,29)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}")
                case "color":
                    items.append(f"#{random.randint(0,0xFFFFFF):06x}")
                case "url":
                    path = "".join(random.choices(string.ascii_lowercase, k=6))
                    items.append(f"https://{random.choice(_DOMAINS)}/{path}")
                case "sentence":
                    words = [random.choice(_LOREM) for _ in range(random.randint(5,12))]
                    words[0] = words[0].capitalize()
                    items.append(" ".join(words) + ".")
                case _:
                    return f"Unknown type: {dtype}"
        return "\n".join(items)
