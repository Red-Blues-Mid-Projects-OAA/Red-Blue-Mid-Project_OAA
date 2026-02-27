"""Compatibility wrapper for DB.build_master_dataset."""

from DB.build_master_dataset import *  # noqa: F401,F403


if __name__ == "__main__":
    import runpy

    runpy.run_module("DB.build_master_dataset", run_name="__main__")
