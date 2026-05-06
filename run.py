"""Convenience wrapper for the combined repair + super-resolution CLI."""

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / "combined_repair_sr2.0" / "combined_repair_sr_optimized.py"


def main():
    spec = importlib.util.spec_from_file_location("combined_repair_sr_optimized", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Unable to load {MODULE_PATH}")
    spec.loader.exec_module(module)
    module.main()


if __name__ == "__main__":
    main()
