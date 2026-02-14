#!/usr/bin/env python3
"""
Shared utilities for dataset extraction and model inference scripts.

This module contains common functions used by multiple scripts including:
- aitoolkit-dataset.py, modelscope-dataset.py (dataset extraction)
- build_warp_dataset.py, vggt_point_cloud_viewer.py (model inference)

Functions are organized to eliminate code duplication across the codebase.
"""

import shutil
from pathlib import Path
from typing import Callable, Optional

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    Image = None
    HAS_PIL = False

try:
    import torch

    HAS_TORCH = True
except (ImportError, OSError):
    torch = None
    HAS_TORCH = False

try:
    from vggt.models.vggt import VGGT

    HAS_VGGT = True
except (ImportError, OSError):
    VGGT = None
    HAS_VGGT = False


def get_file_extension(pattern_path: Optional[Path]) -> str:
    """Get file extension from a path, or return empty string if None.

    Args:
        pattern_path: Path object to extract extension from, or None.

    Returns:
        File extension (e.g., '.jpg', '.png') in lowercase, or empty string if path is None.

    Example:
        >>> path = Path("image.jpg")
        >>> get_file_extension(path)
        '.jpg'
    """
    if pattern_path is None:
        return ""
    return pattern_path.suffix.lower()


def find_triplets_in_scene(scene_dir: Path) -> list[dict]:
    """Find all triplets in a scene directory.

    Scans a directory for matching sets of three image files that form a training triplet:
    - *_splats.<ext>: Splat rendering
    - *_reference.<ext>: Reference image
    - *_target.<ext>: Target image

    All three files must share the same stem (base name before the suffix).
    Only .jpg and .png files are matched; .ply, .txt, and other file types are ignored.

    Args:
        scene_dir: Path to the scene directory to scan for triplets.

    Returns:
        A list of dicts, each containing:
            - 'splats': Path to *_splats.<ext> file
            - 'reference': Path to *_reference.<ext> file (excluding .ply)
            - 'target': Path to *_target.<ext> file
            - 'stem': Base name without suffix (e.g., 'image1')

        Only complete triplets (all three files present) are returned.
        The list is sorted by stem name for consistent ordering.

    Example:
        Given a directory with:
            - image1_splats.jpg
            - image1_reference.jpg
            - image1_target.jpg
            - image2_splats.png  (incomplete, missing reference/target)

        Returns:
            [{'splats': Path('image1_splats.jpg'),
              'reference': Path('image1_reference.jpg'),
              'target': Path('image1_target.jpg'),
              'stem': 'image1'}]
    """
    triplets = {}

    for file_path in scene_dir.iterdir():
        if not file_path.is_file():
            continue

        name = file_path.name

        # Extract the stem (e.g., "image1" from "image1_splats.jpg")
        # Only match specific suffixes to avoid metadata files like _reference_intrinsics.txt
        if name.endswith("_splats.jpg") or name.endswith("_splats.png"):
            stem = name.replace("_splats.jpg", "").replace("_splats.png", "")
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["splats"] = file_path
            triplets[stem]["stem"] = stem
        elif name.endswith("_reference.jpg") or name.endswith("_reference.png"):
            # Only match JPG and PNG, NOT PLY or TXT files
            stem = name.replace("_reference.jpg", "").replace("_reference.png", "")
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["reference"] = file_path
            triplets[stem]["stem"] = stem
        elif name.endswith("_target.jpg") or name.endswith("_target.png"):
            stem = name.replace("_target.jpg", "").replace("_target.png", "")
            if stem not in triplets:
                triplets[stem] = {}
            triplets[stem]["target"] = file_path
            triplets[stem]["stem"] = stem

    # Filter to only complete triplets and return as list
    complete_triplets = []
    for stem, files in sorted(triplets.items()):
        if "splats" in files and "reference" in files and "target" in files:
            complete_triplets.append(files)

    return complete_triplets


def validate_image_file(file_path: Path, file_type: str = "image") -> None:
    """Validate that an image file is readable using PIL.

    Args:
        file_path: Path to the image file to validate.
        file_type: Descriptive name for the file type (used in error messages).

    Raises:
        ImportError: If PIL is not installed.
        ValueError: If the image cannot be read or is invalid.

    Example:
        >>> validate_image_file(Path("image.jpg"), "splats")
    """
    if not HAS_PIL:
        raise ImportError("PIL (Pillow) is required for image validation")

    try:
        with Image.open(file_path) as img:
            # Force load to verify the file is valid
            img.load()
    except Exception as e:
        raise ValueError(f"Cannot read {file_type} image: {file_path} - {e}") from e


def extract_dataset_generic(
    output_dir: Path,
    target_base_dir: Path,
    naming_fn: Callable[[int, str], dict[str, Path]],
    prompt: Optional[str] = None,
    verbose: bool = True,
) -> int:
    """Generic dataset extraction function for dataset preparation scripts.

    Eliminates code duplication across aitoolkit-dataset.py and modelscope-dataset.py
    by providing a reusable extraction pipeline that:

    1. Iterates through scene directories
    2. Finds triplets of images in each scene
    3. Validates image files are readable
    4. Copies files to output locations with custom naming
    5. Optionally writes prompt files
    6. Tracks progress with verbose output

    Args:
        output_dir: Input directory with scene outputs from build_warp_dataset.py
        target_base_dir: Base output directory for the dataset
        naming_fn: Callable that takes (folder_counter, file_extension) and returns
                   a dict with keys 'splats', 'reference', 'target' (required) and
                   optionally 'prompt'. Each value should be a Path object where
                   the file should be written.
        prompt: Optional prompt text to save for each training triplet (if naming_fn
                provides a 'prompt' key, the prompt will be written to that path).
        verbose: Print progress information to stdout.

    Returns:
        Total number of training triplets extracted.

    Raises:
        ValueError: If an image file fails validation.

    Example:
        >>> def naming_fn(counter, ext):
        ...     return {
        ...         'splats': target_dir / f'{counter}{ext}',
        ...         'reference': target_dir / f'{counter}{ext}',
        ...         'target': target_dir / f'{counter}{ext}',
        ...     }
        >>> extract_dataset_generic(output_dir, target_dir, naming_fn, verbose=True)
    """
    folder_counter = 1
    total_triplets = 0

    if verbose:
        print(f"Extracting dataset to {target_base_dir}")

    # Iterate through scene folders
    scene_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name

        # Skip hidden directories
        if scene_name.startswith("."):
            continue

        if verbose:
            print(f"Processing {scene_name}...")

        triplets = find_triplets_in_scene(scene_dir)

        if not triplets:
            if verbose:
                print(
                    f"  Warning: No complete triplets found in {scene_name}, skipping"
                )
            continue

        # Process each triplet
        for files in triplets:
            ext = get_file_extension(files["splats"])

            # Get output paths from the naming function
            paths = naming_fn(folder_counter, ext)

            # Create parent directories as needed
            for path_dict_key, path in paths.items():
                if path_dict_key != "prompt":  # Don't create dir for prompt yet
                    path.parent.mkdir(parents=True, exist_ok=True)

            # Validate image files are readable before copying
            if HAS_PIL:
                for src_file, dest_name in [
                    (files["splats"], "splats"),
                    (files["reference"], "reference"),
                    (files["target"], "target"),
                ]:
                    try:
                        validate_image_file(src_file, dest_name)
                    except ValueError as e:
                        print(f"  ERROR: Invalid image file {src_file.name}: {e}")
                        raise

            # Copy files to output locations
            shutil.copy2(files["splats"], paths["splats"])
            shutil.copy2(files["reference"], paths["reference"])
            shutil.copy2(files["target"], paths["target"])

            # Write prompt file if provided and naming_fn includes 'prompt' key
            if prompt is not None and "prompt" in paths:
                prompt_path = paths["prompt"]
                prompt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(prompt_path, "w", encoding="utf-8") as f:
                    f.write(prompt)

            if verbose:
                print(f"  Created triplet {folder_counter} from {files['stem']}:")
                for key in ["splats", "reference", "target", "prompt"]:
                    if key in paths:
                        print(f"    - {paths[key].name}")

            folder_counter += 1
            total_triplets += 1

    if verbose:
        print(f"\nExtraction complete!")
        print(f"Created {total_triplets} training triplets in {target_base_dir}")

    return total_triplets


def load_model(device: "torch.device") -> "VGGT":
    """Load VGGT model for depth estimation.

    Loads the facebook/VGGT-1B model, moves it to the specified device,
    and sets it to evaluation mode. This function is used by both
    build_warp_dataset.py and vggt_point_cloud_viewer.py.

    Args:
        device: torch.device to load the model on (e.g., torch.device('cuda') or 'cpu').

    Returns:
        VGGT model in evaluation mode on the specified device.

    Raises:
        ImportError: If torch or VGGT is not installed.

    Example:
        >>> device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        >>> model = load_model(device)
    """
    if not HAS_TORCH:
        raise ImportError("torch and vggt are required for load_model()")

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)
    model.eval()
    return model


def build_view_matrix(extrinsic: "np.ndarray") -> "np.ndarray":
    """Build a 4x4 view matrix from extrinsic (camera pose) matrix.

    Converts an extrinsic camera matrix (rotation + translation) to a view matrix
    using OpenGL coordinate system conventions. This function is used by both
    build_warp_dataset.py and vggt_point_cloud_viewer.py for point cloud rendering.

    The conversion applies:
    1. Identity + extrinsic orientation and position
    2. Coordinate system conversion: [1, -1, -1, 1] (RDF to RUB convention)

    Args:
        extrinsic: 4x4 extrinsic matrix or 3x4 camera pose matrix.
                  Should contain rotation in [:3, :3] and translation in [:3, 3].

    Returns:
        4x4 view matrix in OpenGL-compatible format (float32).

    Raises:
        ImportError: If numpy is not installed.

    Example:
        >>> extrinsic = np.eye(4)  # Identity pose
        >>> view = build_view_matrix(extrinsic)
        >>> view.shape
        (4, 4)
    """
    if not HAS_NUMPY:
        raise ImportError("numpy is required for build_view_matrix()")

    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = extrinsic[:3, :3]
    view[:3, 3] = extrinsic[:3, 3]
    conversion = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)
    return conversion @ view
