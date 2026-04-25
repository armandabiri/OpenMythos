"""Bootstrap .venv: create virtual environment if missing, install packages.

Standalone usage (from commands.json expressions):
    python .intelag/commands/scripts/_venv_utils.py
    python .intelag/commands/scripts/_venv_utils.py --deps ruff,isort
    python .intelag/commands/scripts/_venv_utils.py --deps pyyaml --venv .venv

Importable usage (from other scripts):
    from _venv_utils import ensure_venv, install
    venv_py = ensure_venv()
    install(["pyyaml"])
"""

import subprocess
import sys
from pathlib import Path

_IS_WIN = sys.platform == "win32"


def venv_python(venv_dir: str = ".venv") -> str:
    """Return path to the Python executable inside the venv."""
    venv_path = Path(venv_dir)
    if _IS_WIN:
        return str(venv_path / "Scripts" / "python.exe")
    return str(venv_path / "bin" / "python")


def ensure_venv(venv_dir: str = ".venv") -> str:
    """Create the virtual environment if it does not exist. Returns venv python path."""
    py_path = Path(venv_python(venv_dir))
    if py_path.exists():
        return str(py_path)
    print(f"[venv] Creating virtual environment at {venv_dir} ...")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    if not py_path.exists():
        print(f"[venv] ERROR: {py_path} not found after creation", file=sys.stderr)
        sys.exit(1)
    print("[venv] Created successfully.")
    return str(py_path)


def install(packages: list[str], venv_dir: str = ".venv") -> None:
    """Install packages into the venv. No-op if list is empty."""
    if not packages:
        return
    py = ensure_venv(venv_dir)
    print(f"[venv] Ensuring packages: {', '.join(packages)}")
    subprocess.run(
        [py, "-m", "pip", "install", "--quiet", *packages],
        check=True,
    )


def main() -> None:
    """Parse CLI arguments and ensure/install venv dependencies."""
    deps: list[str] = []
    venv: str = ".venv"
    args: list[str] = sys.argv[1:]
    i: int = 0
    while i < len(args):
        if args[i] == "--deps" and i + 1 < len(args):
            deps = [d.strip() for d in args[i + 1].split(",") if d.strip()]
            i += 2
        elif args[i] == "--venv" and i + 1 < len(args):
            venv = args[i + 1]
            i += 2
        else:
            i += 1
    ensure_venv(venv)
    install(deps, venv)


if __name__ == "__main__":
    main()
