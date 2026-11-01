"""Test setup: the repository root and the CPU stand-ins in tests/stubs come first on sys.path.

The stand-ins replace the CUDA rasterizer, LPIPS and MS-SSIM so that the whole pipeline runs on a
CPU-only machine; see tests/stubs. Nothing outside tests/ imports them.
"""

import os
import sys

import pytest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
STUBS_DIR = os.path.join(TESTS_DIR, "stubs")
for path in (TESTS_DIR, REPO_ROOT, STUBS_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def synthetic_sequence(tmp_path_factory):
    """A 12-frame synthetic Replica-format sequence: (data_root, data_yaml)."""
    from synthetic_scene import write_sequence

    root = tmp_path_factory.mktemp("data")
    return str(root), write_sequence(str(root))
