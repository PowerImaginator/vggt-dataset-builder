#!/usr/bin/env python3
"""
Extract dataset from output folder for LoRA training.

This script reorganizes the output from build_warp_dataset.py into a format
suitable for training LoRA models with ModelScope.

Structure:
  output/
    scene1/
      image1_splats.jpg
      image1_reference.jpg
      image1_target.jpg
      ...
    scene2/
      ...

Output:
  modelscope-dataset/
    1_start_1.jpg
    1_start_2.jpg
    1_end.jpg
    2_start_1.jpg
    2_start_2.jpg
    2_end.jpg
    ...
"""

import argparse
from pathlib import Path
from typing import Optional

from dataset_utils import extract_dataset_generic


def extract_dataset(
    output_dir: Path,
    modelscope_dir: Path,
    prompt: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Extract dataset from output folder for ModelScope LoRA training.
    
    Uses the generic extraction function from dataset_utils to prepare a dataset
    in ModelScope's flat directory structure with custom naming conventions.
    
    Args:
        output_dir: Path to output folder from build_warp_dataset.py
        modelscope_dir: Path to output modelscope-dataset folder
        prompt: Optional prompt text to save for each triplet
        verbose: Print progress information
    """
    # Define a naming function that maps counter & extension to output paths
    # for ModelScope's flat directory structure with custom naming
    def naming_fn(counter: int, ext: str) -> dict[str, Path]:
        return {
            "splats": modelscope_dir / f"{counter}_start_1{ext}",
            "reference": modelscope_dir / f"{counter}_start_2{ext}",
            "target": modelscope_dir / f"{counter}_end{ext}",
            "prompt": modelscope_dir / f"{counter}.txt",
        }
    
    extract_dataset_generic(
        output_dir,
        modelscope_dir,
        naming_fn,
        prompt=prompt,
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract dataset from build_warp_dataset.py output for ModelScope LoRA training."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory with scene outputs from build_warp_dataset.py (default: output)",
    )
    parser.add_argument(
        "--modelscope-dir",
        type=Path,
        default=Path("modelscope-dataset"),
        help="Output directory for ModelScope dataset (default: modelscope-dataset)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt text to save for each training triplet (saved as <N>.txt)",
    )
    
    args = parser.parse_args()
    
    # Validate output directory
    if not args.output_dir.exists():
        print(f"Error: Output directory not found: {args.output_dir}")
        return 1
    
    if not args.output_dir.is_dir():
        print(f"Error: Output path is not a directory: {args.output_dir}")
        return 1
    
    extract_dataset(
        args.output_dir,
        args.modelscope_dir,
        prompt=args.prompt,
        verbose=not args.quiet,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
