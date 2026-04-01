# Contributing to Towel

Don't Panic. Contributions are welcome.

## Development Setup

```bash
git clone https://github.com/Kelsidavis/Towel.git
cd Towel
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest                    # full suite
pytest tests/test_foo.py  # single file
pytest -x                 # stop on first failure
```

## Code Style

We use [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/
```

## Adding a Skill

```bash
towel skill-init my_skill    # generates ~/.towel/skills/my_skill_skill.py
```

Edit the generated file, restart Towel, and your skill is loaded.

## Pull Requests

1. Fork and branch from `main`
2. Write tests for new features
3. Run `pytest` and `ruff check` before pushing
4. Open a PR with a clear description
