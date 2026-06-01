import pytest


def pytest_addoption(parser):
    parser.addoption("--plot", action="store_true", help="Show plots at end of tests")
