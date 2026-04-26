"""Pytest configuration to exclude slow integration tests"""
import pytest

collect_ignore_glob = [
    "scenarios/*.py",
    "standalone/*.py",
    "integration/*.py",  # Integration tests are slow too
    "manual/*.py",       # Manual scripts; have module-level side effects
]
