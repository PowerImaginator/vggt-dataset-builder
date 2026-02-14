#!/usr/bin/env python3
"""Test script for image validation utilities.

Tests the validate_image_file function to ensure proper error handling
for corrupted, invalid, and missing image files in the dataset pipeline.
"""

from pathlib import Path
import pytest
from PIL import Image
import io

from dataset_utils import validate_image_file


def test_validate_valid_image(tmp_path):
    """Test validation of a valid image file (happy path)."""
    # Create a simple valid image
    img_path = tmp_path / "valid_image.jpg"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, "JPEG")

    # Should not raise any exception
    validate_image_file(img_path, "test_image")


def test_validate_corrupted_image(tmp_path):
    """Test validation fails on corrupted image file.

    Critical test: ensures corrupted images are caught early in the dataset
    pipeline before they cause training failures downstream.
    """
    # Create a file that looks like an image but has corrupted data
    img_path = tmp_path / "corrupted.jpg"
    img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"corrupt data that isn't a valid JPEG")

    # Should raise ValueError with descriptive message
    with pytest.raises(ValueError, match="Cannot read test_image image"):
        validate_image_file(img_path, "test_image")


def test_validate_nonexistent_file(tmp_path):
    """Test validation fails gracefully on non-existent file.

    Edge case: ensures missing files are caught with clear error messages.
    """
    img_path = tmp_path / "does_not_exist.jpg"

    # Should raise ValueError (FileNotFoundError wrapped)
    with pytest.raises(ValueError, match="Cannot read test_image image"):
        validate_image_file(img_path, "test_image")


def test_validate_empty_file(tmp_path):
    """Test validation fails on empty file.

    Edge case: empty files should be rejected as invalid images.
    """
    img_path = tmp_path / "empty.jpg"
    img_path.write_bytes(b"")

    # Should raise ValueError
    with pytest.raises(ValueError, match="Cannot read test_image image"):
        validate_image_file(img_path, "test_image")


def test_validate_wrong_extension(tmp_path):
    """Test validation fails when file extension doesn't match content.

    Edge case: a .txt file renamed to .jpg should fail validation.
    """
    img_path = tmp_path / "fake_image.jpg"
    img_path.write_text("This is just text, not an image")

    # Should raise ValueError because PIL cannot open text as image
    with pytest.raises(ValueError, match="Cannot read test_image image"):
        validate_image_file(img_path, "test_image")


def test_validate_truncated_image(tmp_path):
    """Test validation fails on truncated image file.

    Critical test: ensures incomplete image files (e.g., from interrupted
    downloads or failed copies) are detected.
    """
    # Create a valid image, then truncate it mid-stream
    img = Image.new("RGB", (100, 100), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, "JPEG")
    full_data = buffer.getvalue()

    # Write only first half of the image data (truncated)
    img_path = tmp_path / "truncated.jpg"
    img_path.write_bytes(full_data[: len(full_data) // 2])

    # Should raise ValueError when trying to load the truncated image
    with pytest.raises(ValueError, match="Cannot read test_image image"):
        validate_image_file(img_path, "test_image")


def test_validate_custom_file_type_name():
    """Test that custom file_type parameter appears in error messages.

    Verifies error messages use the provided file_type for better debugging.
    """
    from pathlib import Path

    # Use a non-existent file to trigger error
    fake_path = Path("nonexistent.jpg")

    # Error message should include custom file type 'splats'
    with pytest.raises(ValueError, match="Cannot read splats image"):
        validate_image_file(fake_path, "splats")

    # Error message should include custom file type 'reference'
    with pytest.raises(ValueError, match="Cannot read reference image"):
        validate_image_file(fake_path, "reference")


if __name__ == "__main__":
    # Allow running as a script
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        print("Running validation tests...")

        test_validate_valid_image(tmp)
        print("✓ test_validate_valid_image passed")

        test_validate_corrupted_image(tmp)
        print("✓ test_validate_corrupted_image passed")

        test_validate_nonexistent_file(tmp)
        print("✓ test_validate_nonexistent_file passed")

        test_validate_empty_file(tmp)
        print("✓ test_validate_empty_file passed")

        test_validate_wrong_extension(tmp)
        print("✓ test_validate_wrong_extension passed")

        test_validate_truncated_image(tmp)
        print("✓ test_validate_truncated_image passed")

        test_validate_custom_file_type_name()
        print("✓ test_validate_custom_file_type_name passed")

        print("\n✅ All validation tests passed!")
