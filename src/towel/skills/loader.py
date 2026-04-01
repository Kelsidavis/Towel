"""Skill loader — discovers and loads skills from filesystem directories.

Skill directories can contain:
  1. Single-file skills:  my_skill.py  (module with a Skill subclass)
  2. Package skills:      my_skill/    (package with __init__.py exposing a Skill subclass)
  3. Manifest skills:     my_skill/    (package with towel-skill.toml metadata)

The loader scans each directory, imports modules, finds Skill subclasses,
instantiates them, and registers them with the SkillRegistry.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

from towel.skills.base import Skill
from towel.skills.registry import SkillRegistry

log = logging.getLogger("towel.skills.loader")


class SkillLoadError:
    """Record of a failed skill load attempt."""

    def __init__(self, path: Path, error: Exception) -> None:
        self.path = path
        self.error = error

    def __repr__(self) -> str:
        return f"SkillLoadError({self.path}, {self.error!r})"


class SkillLoader:
    """Discovers and loads skills from filesystem directories."""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        self.errors: list[SkillLoadError] = []

    def load_from_dirs(self, dirs: list[str]) -> int:
        """Scan directories for skills and register them.

        Returns the number of skills successfully loaded.
        """
        loaded = 0
        for dir_str in dirs:
            path = Path(dir_str).expanduser().resolve()
            if not path.is_dir():
                log.debug(f"Skills directory does not exist: {path}")
                continue
            loaded += self._scan_directory(path)
        return loaded

    def _scan_directory(self, directory: Path) -> int:
        """Scan a single directory for skill modules and packages."""
        loaded = 0

        for entry in sorted(directory.iterdir()):
            # Skip hidden files/dirs and __pycache__
            if entry.name.startswith((".", "_")):
                continue

            try:
                if entry.is_file() and entry.suffix == ".py":
                    skills = self._load_module_file(entry)
                elif entry.is_dir() and (entry / "__init__.py").exists():
                    skills = self._load_package(entry)
                else:
                    continue

                for skill in skills:
                    if skill.name in self.registry.list_skills():
                        log.warning(
                            f"Skill '{skill.name}' from {entry} conflicts with "
                            f"already-registered skill, skipping"
                        )
                        continue
                    self.registry.register(skill)
                    log.info(f"Loaded skill '{skill.name}' from {entry}")
                    loaded += 1

            except Exception as e:
                log.warning(f"Failed to load skill from {entry}: {e}")
                self.errors.append(SkillLoadError(entry, e))

        return loaded

    def _load_module_file(self, path: Path) -> list[Skill]:
        """Load skill classes from a single .py file."""
        module_name = f"towel_skill_{path.stem}"
        return self._load_module(module_name, path)

    def _load_package(self, path: Path) -> list[Skill]:
        """Load skill classes from a package directory."""
        module_name = f"towel_skill_{path.name}"
        init_path = path / "__init__.py"
        return self._load_module(module_name, init_path)

    def _load_module(self, module_name: str, file_path: Path) -> list[Skill]:
        """Import a module from a file path and extract Skill subclasses."""
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot create module spec for {file_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        return self._find_skills_in_module(module)

    def _find_skills_in_module(self, module: Any) -> list[Skill]:
        """Find and instantiate all Skill subclasses in a module."""
        skills: list[Skill] = []

        for _name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Skill) and obj is not Skill and not inspect.isabstract(obj):
                try:
                    instance = obj()
                    skills.append(instance)
                except Exception as e:
                    log.warning(f"Failed to instantiate {obj.__name__}: {e}")

        return skills
