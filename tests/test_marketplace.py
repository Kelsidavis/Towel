"""Tests for the skill marketplace."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from towel.skills.marketplace import (
    COMMUNITY_SKILLS,
    install_skill,
    list_installed,
    remove_skill,
    search_marketplace,
)


def _mock_get(status_code: int = 200, text: str = "") -> AsyncMock:
    """Create an AsyncMock that returns a fake httpx Response."""
    fake_resp = MagicMock()
    fake_resp.status_code = status_code
    fake_resp.text = text
    return AsyncMock(return_value=fake_resp)


class TestMarketplaceRegistry:
    def test_has_skills(self):
        assert len(COMMUNITY_SKILLS) >= 10

    def test_skills_have_required_fields(self):
        for s in COMMUNITY_SKILLS:
            assert "name" in s
            assert "description" in s
            assert "url" in s

    def test_names_unique(self):
        names = [s["name"] for s in COMMUNITY_SKILLS]
        assert len(names) == len(set(names))


class TestSearch:
    def test_search_by_name(self):
        results = search_marketplace("weather")
        assert any(r["name"] == "weather" for r in results)

    def test_search_by_tag(self):
        results = search_marketplace("finance")
        assert len(results) >= 1

    def test_search_no_results(self):
        results = search_marketplace("xyznonexistent")
        assert len(results) == 0

    def test_search_case_insensitive(self):
        results = search_marketplace("WEATHER")
        assert len(results) >= 1


class TestInstalled:
    def test_list_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", tmp_path / "empty")
        assert list_installed() == []

    def test_list_with_skills(self, tmp_path, monkeypatch):
        d = tmp_path / "skills"
        d.mkdir()
        (d / "weather_skill.py").write_text("pass")
        (d / "reddit_skill.py").write_text("pass")
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", d)
        installed = list_installed()
        assert "weather" in installed
        assert "reddit" in installed

    def test_remove(self, tmp_path, monkeypatch):
        d = tmp_path / "skills"
        d.mkdir()
        (d / "weather_skill.py").write_text("pass")
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", d)
        result = remove_skill("weather")
        assert "Removed" in result
        assert not (d / "weather_skill.py").exists()

    def test_remove_not_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", tmp_path)
        result = remove_skill("nonexistent")
        assert "Not installed" in result


class TestInstall:
    @pytest.mark.asyncio
    async def test_install_success(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", tmp_path)
        get = _mock_get(text="class WeatherSkill: pass")
        with patch("httpx.AsyncClient.get", get):
            result = await install_skill("weather")
        assert "Installed: weather" in result
        assert (tmp_path / "weather_skill.py").read_text() == "class WeatherSkill: pass"

    @pytest.mark.asyncio
    async def test_install_unknown_skill(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", tmp_path)
        result = await install_skill("nonexistent")
        assert "Skill not found" in result

    @pytest.mark.asyncio
    async def test_install_already_installed(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)
        (tmp_path / "weather_skill.py").write_text("pass")
        result = await install_skill("weather")
        assert "Already installed" in result

    @pytest.mark.asyncio
    async def test_install_download_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.marketplace.SKILLS_DIR", tmp_path)
        get = _mock_get(status_code=404)
        with patch("httpx.AsyncClient.get", get):
            result = await install_skill("weather")
        assert "Download failed" in result
        assert not (tmp_path / "weather_skill.py").exists()
