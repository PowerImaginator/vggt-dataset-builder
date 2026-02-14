#!/usr/bin/env python3
"""
Prepare dataset for AI Toolkit.

AI Toolkit expects three separate folders:
  - control1: splats images
  - control2: reference images
  - target: target images and prompts

Input structure (from build_warp_dataset.py):
  output/
    scene1/
      image1_splats.jpg
      image1_reference.jpg
      image1_target.jpg
      ...

Output structure:
  aitoolkit-dataset/
    control1/
      1.jpg
      2.jpg
      ...
    control2/
      1.jpg
      2.jpg
      ...
    target/
      1.jpg
      1.txt
      2.jpg
      2.txt
      ...
"""

import argparse
from pathlib import Path
from typing import Optional

from dataset_utils import extract_dataset_generic


def extract_dataset(
    output_dir: Path,
    aitoolkit_dir: Path,
    prompt: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Extract dataset for AI Toolkit training.

    Uses the generic extraction function from dataset_utils to prepare a dataset
    with three subdirectories (control1, control2, target) for AI Toolkit training.

    Args:
        output_dir: Path to output folder from build_warp_dataset.py
        aitoolkit_dir: Path to output aitoolkit-dataset folder
        prompt: Optional prompt text to save for each triplet
        verbose: Print progress information
    """

    # Define a naming function that maps counter & extension to output paths
    # for AI Toolkit's three-subdirectory structure
    def naming_fn(counter: int, ext: str) -> dict[str, Path]:
        return {
            "splats": aitoolkit_dir / "control1" / f"{counter}{ext}",
            "reference": aitoolkit_dir / "control2" / f"{counter}{ext}",
            "target": aitoolkit_dir / "target" / f"{counter}{ext}",
            "prompt": aitoolkit_dir / "target" / f"{counter}.txt",
        }

    extract_dataset_generic(
        output_dir,
        aitoolkit_dir,
        naming_fn,
        prompt=prompt,
        verbose=verbose,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract dataset from build_warp_dataset.py output for AI Toolkit."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory with scene outputs from build_warp_dataset.py (default: output)",
    )
    parser.add_argument(
        "--aitoolkit-dir",
        type=Path,
        default=Path("aitoolkit-dataset"),
        help="Output directory for AI Toolkit dataset (default: aitoolkit-dataset)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to save for each training triplet (saved as target/<N>.txt)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"Error: Output directory not found: {args.output_dir}")
        return 1

    if not args.output_dir.is_dir():
        print(f"Error: Output path is not a directory: {args.output_dir}")
        return 1

    extract_dataset(
        args.output_dir,
        args.aitoolkit_dir,
        prompt=args.prompt,
        verbose=not args.quiet,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
