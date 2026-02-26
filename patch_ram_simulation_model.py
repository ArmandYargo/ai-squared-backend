# patch_ram_simulation_model.py
"""
One-time patcher:
Wrap the import-time "# Example usage" block in RAM_Simulation_Model.py inside:
    if __name__ == "__main__":

This prevents the simulation from running when imported by the agent.

Default target (your project layout):
    ./ram_module/RAM_Simulation_Model.py

Usage:
    python patch_ram_simulation_model.py
or:
    python patch_ram_simulation_model.py "full/path/to/RAM_Simulation_Model.py"
"""

from __future__ import annotations
import sys
from pathlib import Path


def _default_model_path() -> Path:
    return Path.cwd() / "ram_module" / "RAM_Simulation_Model.py"


def main():
    # Allow explicit path, else default to ram_module layout
    if len(sys.argv) > 1:
        model_path = Path(sys.argv[1]).expanduser()
    else:
        model_path = _default_model_path()

    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find RAM model at: {model_path.resolve()}")

    text = model_path.read_text(encoding="utf-8", errors="ignore")

    # Already patched?
    if 'if __name__ == "__main__":' in text and "# Example usage" in text:
        print("Already patched ✅")
        print(f"File: {model_path.resolve()}")
        return

    i_example = text.find("# Example usage")
    i_runner = text.find("def run_ram_simulation(")

    if i_example == -1:
        raise RuntimeError("Could not find '# Example usage' block in RAM_Simulation_Model.py")
    if i_runner == -1:
        raise RuntimeError("Could not find 'def run_ram_simulation(' in RAM_Simulation_Model.py")
    if i_runner <= i_example:
        raise RuntimeError("Unexpected layout: run_ram_simulation appears before Example usage block.")

    before = text[:i_example]
    block = text[i_example:i_runner]
    after = text[i_runner:]

    # indent the example block under main guard
    indented = "\n".join(("    " + line) if line.strip() else line for line in block.splitlines())
    patched = before + 'if __name__ == "__main__":\n' + indented + "\n\n" + after

    backup = model_path.with_suffix(model_path.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    model_path.write_text(patched, encoding="utf-8")

    print("Patched successfully ✅")
    print(f"Backup:  {backup.resolve()}")
    print(f"Patched: {model_path.resolve()}")


if __name__ == "__main__":
    main()
