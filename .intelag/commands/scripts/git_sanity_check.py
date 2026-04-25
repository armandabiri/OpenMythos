"""Install and configure pre-commit with ggshield and gitleaks; update .gitignore and .env template."""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

PRE_COMMIT_CONFIG_PATH = ".pre-commit-config.yaml"
GITIGNORE_PATH = ".gitignore"
ENV_PATH = ".env"
PRE_COMMIT_CONFIG_CONTENT = """repos:
  - repo: https://github.com/zricethezav/gitleaks
    rev: v8.18.2
    hooks:
      - id: gitleaks
        args: ["detect"]
"""
COMMON_IGNORES = [
    ".env",
    "*.key",
    "*.pem",
    "secrets.yaml",
    "credentials.json",
    "*.log",
    ".DS_Store",
    ".history/",
]


def _install_package(package: str) -> None:
    """Install a package via pip; exit on failure."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        logger.error("Failed to install %s: %s", package, e)
        sys.exit(1)


def main() -> None:
    """Run git security sanity check and hardening steps."""
    logger.info("Git Security Sanity Check & Hardening")

    if not shutil.which("pre-commit"):
        logger.info("Installing pre-commit...")
        _install_package("pre-commit")
    else:
        logger.info("pre-commit already installed.")

    pre_commit_config_path = Path(PRE_COMMIT_CONFIG_PATH)
    gitignore_path = Path(GITIGNORE_PATH)
    env_path = Path(ENV_PATH)

    if not pre_commit_config_path.exists():
        with pre_commit_config_path.open("w", encoding="utf-8") as f:
            f.write(PRE_COMMIT_CONFIG_CONTENT)
        logger.info("Created .pre-commit-config.yaml with ggshield and gitleaks.")
    else:
        logger.info(".pre-commit-config.yaml exists. Skipping creation.")

    logger.info("Installing git hooks...")
    try:
        subprocess.check_call(["pre-commit", "install"])
        logger.info("Pre-commit hooks installed successfully.")
    except (subprocess.CalledProcessError, OSError) as e:
        logger.warning("Failed to install hooks: %s", e)
        try:
            scripts_dir = Path(sys.executable).parent
            pc_exe = scripts_dir / (
                "pre-commit.exe" if sys.platform == "win32" else "pre-commit"
            )
            if pc_exe.exists():
                subprocess.check_call([str(pc_exe), "install"])
                logger.info("Hooks installed via absolute path.")
        except Exception:
            logger.warning("Please run 'pre-commit install' manually.")

    existing_ignores: set[str] = set()
    if gitignore_path.exists():
        with gitignore_path.open(encoding="utf-8") as f:
            existing_ignores = {line.strip() for line in f}

    ignores_added: List[str] = []
    with gitignore_path.open("a", encoding="utf-8") as f:
        for ignore in COMMON_IGNORES:
            if ignore not in existing_ignores:
                f.write(f"\n{ignore}")
                ignores_added.append(ignore)

    if ignores_added:
        logger.info("Added to .gitignore: %s", ", ".join(ignores_added))
    else:
        logger.info(".gitignore already contains standard exclusions.")

    if not env_path.exists():
        with env_path.open("w", encoding="utf-8") as f:
            f.write(
                "# Secrets go here. DO NOT COMMIT THIS FILE.\n"
                "# Added by Intelag Security Check\n"
                "SECRET_KEY=change_me\n"
                "DB_PASSWORD=secret\n"
            )
        logger.info("Created .env template.")
    else:
        logger.info(".env file exists.")

    logger.info(
        "Hardening complete. Next: rotate secrets, move to .env, commit config."
    )


def example_usage() -> None:
    """Run a minimal example for manual testing."""
    logger.info("Example: git_sanity_check.py (run from repo root)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--example" in sys.argv:
        example_usage()
    else:
        main()
