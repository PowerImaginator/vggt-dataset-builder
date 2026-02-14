# Dryer's Journal 🏜️

## 2026-02-14 - Extract Generic Dataset Extraction Pipeline

**Learning:** Callback/strategy pattern is highly effective for eliminating duplication when two codebases share 90% of logic but differ in output file naming or directory structure. Rather than creating multiple variants or conditional logic, passing a simple naming function preserves both readability and flexibility.

**Action:** When encountering duplication in extraction/processing pipelines, identify the invariant logic (iteration, validation, copying) and parameterize only the variant parts (file naming, directory structure). This keeps shared code DRY without sacrificing clarity or creating over-generalized utilities.

**Context:** The aitoolkit-dataset.py and modelscope-dataset.py scripts were nearly identical (70+ lines each) except for how they named output files (AIToolkit used 3 subdirectories with simple naming; ModelScope used flat naming with suffixes). A single generic function with a configurable naming callback eliminated ~100 lines of duplication while making both scripts clearer as thin wrappers.

## 2026-02-14 - Extract Model Utilities for VGGT Inference

**Learning:** VGGT model loading and coordinate transformation logic are universal across inference-based scripts. These pure utility functions should be extracted to shared modules with comprehensive docstrings. Optional imports for torch/VGGT allow graceful degradation when those dependencies aren't needed.

**Action:** When multiple scripts independently repeat the same OpenGL/3D math utilities or model initialization code, extract to dataset_utils.py (even if it's not strictly a "dataset" utility—naming can be refactored later if needed). Add clear docstrings explaining the function's role across multiple scripts.

**Context:** build_warp_dataset.py and vggt_point_cloud_viewer.py both contained identical `load_model(device)` and `build_view_matrix(extrinsic)` functions. Extracted both to dataset_utils.py with docstrings explaining their use in VGGT depth estimation and point cloud rendering. Reduced 30 lines of duplication; both scripts now import from shared utils. Tests pass; no behavior changes.


