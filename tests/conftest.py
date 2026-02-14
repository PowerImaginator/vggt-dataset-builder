"""Pytest configuration and fixtures for vggt-dataset-builder tests."""

import sys
from pathlib import Path

# Add the workspace root to sys.path so tests can import modules
workspace_root = Path(__file__).parent.parent
sys.path.insert(0, str(workspace_root))

import pytest


@pytest.fixture
def workspace_root_path():
    """Provide the workspace root path to tests."""
    return Path(__file__).parent.parent


@pytest.fixture
def test_data_dir():
    """Provide the test data directory path."""
    return Path(__file__).parent / "data"


@pytest.fixture
def cache_dir():
    """Provide the workspace cache directory path.

    Returns the .cache directory at the workspace root, creating it if needed.
    This is used for downloaded models and other cached resources.
    """
    cache_path = Path(__file__).parent.parent / ".cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path


@pytest.fixture
def input_img_dir():
    """Provide the input-img directory path for consistent test image access."""
    return Path(__file__).parent / "input-img"


@pytest.fixture
def test_image_path(input_img_dir):
    """Provide a default test image path from input-img folder.

    Uses the first available image from input-img/02/ (PNG files) for testing.
    Modify this fixture if tests need different image formats.
    """
    # Use PNG from folder 02 as default test image
    test_img = input_img_dir / "02" / "frame_00001.png"
    if test_img.exists():
        return test_img

    # Fallback: find any image in input-img
    for subdir in input_img_dir.iterdir():
        if subdir.is_dir():
            for img in subdir.iterdir():
                if img.suffix.lower() in (".png", ".jpg", ".jpeg", ".heic"):
                    return img

    # No test images found
    pytest.skip("No test images found in input-img directory")
