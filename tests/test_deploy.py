"""Tests for deployment generator."""

from towel.cli.deploy import generate_deploy


class TestDeploy:
    def test_docker(self, tmp_path):
        files = generate_deploy("docker", tmp_path)
        assert len(files) == 3
        names = {f.name for f in files}
        assert "Dockerfile" in names
        assert "docker-compose.yml" in names
        assert ".env.example" in names
        assert "towel" in (tmp_path / "Dockerfile").read_text()

    def test_systemd(self, tmp_path):
        files = generate_deploy("systemd", tmp_path, user="kelsi")
        assert any("towel.service" in str(f) for f in files)
        content = (tmp_path / "towel.service").read_text()
        assert "kelsi" in content

    def test_heroku(self, tmp_path):
        files = generate_deploy("heroku", tmp_path)
        assert any("Procfile" in str(f) for f in files)
        assert "towel serve" in (tmp_path / "Procfile").read_text()

    def test_fly(self, tmp_path):
        files = generate_deploy("fly", tmp_path)
        assert any("fly.toml" in str(f) for f in files)

    def test_all(self, tmp_path):
        files = generate_deploy("all", tmp_path)
        names = {f.name for f in files}
        assert "Dockerfile" in names
        assert "towel.service" in names
        assert "Procfile" in names
        assert "fly.toml" in names

    def test_cli_registered(self):
        from towel.cli.main import cli

        assert "deploy" in [c.name for c in cli.commands.values()]
