"""Pytest configuration: custom marks and shared fixtures."""
from __future__ import annotations

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: mark test as slow (runs neural-network training or many env steps — skip with -m 'not slow')",
    )
