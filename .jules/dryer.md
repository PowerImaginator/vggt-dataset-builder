# Dryer's Journal 🏜️

## 2026-02-14 - Lazy Import Fix for VGGT Path Setup

**Critical Bug Fix:** After extracting `setup_vggt_path()` utility, discovered that `load_model()` failed with `AttributeError: 'NoneType' object has no attribute 'from_pretrained'`. Root cause: Module-level imports of VGGT in dataset_utils.py execute before user scripts call `setup_vggt_path()`, causing the import to fail (path not yet in sys.path), setting VGGT to None.

**Solution:** Implemented lazy import inside `load_model()` - move the VGGT import to function-level so it happens AFTER `setup_vggt_path()` has been called. Added proper error handling to guide users if import still fails.

**Learning:** When extracting shared utilities that depend on path setup (or other initialization), be careful about module-level imports. If an import depends on setup that happens at runtime, use lazy (function-level) imports instead. This is especially important when dealing with submodule paths that are added to sys.path dynamically.

**Impact:** vggt_point_cloud_viewer.py now runs successfully again. Tests pass (3/4 baseline maintained).

## 2026-02-14 - Extract VGGT Path Setup Utility

**Learning:** The vggt path setup pattern (`sys.path.insert(0, str(Path(__file__).parent / "vggt"))`) is not trivial duplication despite being only 1-2 lines per script. When a pattern appears 3+ times across scripts and involves path construction + conditional checking (in some files), centralizing it prevents path inconsistencies and makes future changes (e.g., if vggt submodule moves) automatically propagate everywhere. Simple utility functions with zero dependencies are ideal extraction targets.

**Action:** When identifying duplication, don't dismiss small patterns just because they're 1-2 lines each. If they appear 3+ times and involve logic/path construction, extract them. Prioritize patterns with zero external dependencies (like path setup) as they're safe to extract and quick to verify.

**Context:** Found that build_warp_dataset.py, vggt_point_cloud_viewer.py, and vggt_comfy_nodes.py all had nearly identical vggt path setup code. build_warp_dataset.py and vggt_comfy_nodes.py used simple `sys.path.insert()`, while vggt_point_cloud_viewer.py added a conditional check to avoid duplicate paths. Extracted `setup_vggt_path()` to dataset_utils.py with idempotent logic (checks if path already exists before inserting). Now all three scripts call `setup_vggt_path()` right after importing dataset_utils, making the pattern consistent and centralized. Benefits: (1) Removes 9 lines of duplication across 3 scripts, (2) All scripts now use identical/safer path setup logic, (3) Future module reorganizations only need to change one place, (4) Black formatting check passes, tests pass (3/4, with pre-existing failure unrelated to changes).

## 2026-02-14 - Extract Generic Dataset Extraction Pipeline

**Learning:** Callback/strategy pattern is highly effective for eliminating duplication when two codebases share 90% of logic but differ in output file naming or directory structure. Rather than creating multiple variants or conditional logic, passing a simple naming function preserves both readability and flexibility.

**Action:** When encountering duplication in extraction/processing pipelines, identify the invariant logic (iteration, validation, copying) and parameterize only the variant parts (file naming, directory structure). This keeps shared code DRY without sacrificing clarity or creating over-generalized utilities.

**Context:** The aitoolkit-dataset.py and modelscope-dataset.py scripts were nearly identical (70+ lines each) except for how they named output files (AIToolkit used 3 subdirectories with simple naming; ModelScope used flat naming with suffixes). A single generic function with a configurable naming callback eliminated ~100 lines of duplication while making both scripts clearer as thin wrappers.

## 2026-02-14 - Extract Model Utilities for VGGT Inference

**Learning:** VGGT model loading and coordinate transformation logic are universal across inference-based scripts. These pure utility functions should be extracted to shared modules with comprehensive docstrings. Optional imports for torch/VGGT allow graceful degradation when those dependencies aren't needed.

**Action:** When multiple scripts independently repeat the same OpenGL/3D math utilities or model initialization code, extract to dataset_utils.py (even if it's not strictly a "dataset" utility—naming can be refactored later if needed). Add clear docstrings explaining the function's role across multiple scripts.

**Context:** build_warp_dataset.py and vggt_point_cloud_viewer.py both contained identical `load_model(device)` and `build_view_matrix(extrinsic)` functions. Extracted both to dataset_utils.py with docstrings explaining their use in VGGT depth estimation and point cloud rendering. Reduced 30 lines of duplication; both scripts now import from shared utils. Tests pass; no behavior changes.

## 2026-02-14 - Extract Torch Device and Dtype Selection Utilities

**Learning:** Device and dtype selection logic in PyTorch scripts is small (~7 lines each) but highly repetitive and follows a standard pattern: auto-detect CUDA, select bfloat16 for modern GPUs, fallback to float16. Even though the duplication is modest per-script, centralizing this logic ensures consistency and makes future hardware capability changes trivial to update.

**Action:** Extract small but repetitive patterns when they involve complex conditional logic (like GPU capability detection) or when consistency across scripts matters. Don't dismiss extractions just because they're "only 7 lines"—if those 7 lines contain hardware-specific logic that could change, centralization adds value beyond line-count savings.

**Context:** Both build_warp_dataset.py and vggt_point_cloud_viewer.py duplicated device selection (`device_arg or cuda if available else cpu`) and dtype selection (bfloat16 for compute capability >= 8, else float16). Extracted `select_device()` and `select_dtype()` to dataset_utils.py with comprehensive docstrings. Eliminated 14 lines of duplication across 2 scripts. Benefits: (1) Consistency: all scripts now use identical device/dtype logic, (2) Maintainability: hardware capability logic centralized, (3) Future-proof: new scripts can import these utilities immediately.

## 2026-02-14 - Standardize Test Image Access with Fixtures

**Learning:** Test infrastructure benefits as much from DRY principles as production code. When tests each reference images differently (hardcoded paths, command-line args, synthetic data), it creates friction and inconsistency. A centralized fixture for test image access makes tests more portable and maintainable.

**Action:** When adding test data folders, immediately create pytest fixtures in conftest.py to provide consistent access. Consider multiple fixtures: (1) directory fixture for flexibility, (2) default image fixture for convenience. Document which test images are available and their intended use cases.

**Context:** User added `tests/input-img/` folder with test images in subdirectories (01/ with HEIC files, 02/ with PNG files). test_sky_filter.py was broken because it expected an `image_path` fixture that didn't exist. Created three new fixtures in conftest.py: `input_img_dir()` for directory access, `test_image_path()` for default test image, and updated test_sky_filter.py to work both as pytest test (using fixture) and standalone script (taking CLI arg). Benefits: (1) Fixed broken test, (2) Consistent test image access pattern for future tests, (3) Tests can now discover available test images programmatically.


**Learning:** The vggt path setup pattern (`sys.path.insert(0, str(Path(__file__).parent / "vggt"))`) is not trivial duplication despite being only 1-2 lines per script. When a pattern appears 3+ times across scripts and involves path construction + conditional checking (in some files), centralizing it prevents path inconsistencies and makes future changes (e.g., if vggt submodule moves) automatically propagate everywhere. Simple utility functions with zero dependencies are ideal extraction targets.

**Action:** When identifying duplication, don't dismiss small patterns just because they're 1-2 lines each. If they appear 3+ times and involve logic/path construction, extract them. Prioritize patterns with zero external dependencies (like path setup) as they're safe to extract and quick to verify.

**Context:** Found that build_warp_dataset.py, vggt_point_cloud_viewer.py, and vggt_comfy_nodes.py all had nearly identical vggt path setup code. build_warp_dataset.py and vggt_comfy_nodes.py used simple `sys.path.insert()`, while vggt_point_cloud_viewer.py added a conditional check to avoid duplicate paths. Extracted `setup_vggt_path()` to dataset_utils.py with idempotent logic (checks if path already exists before inserting). Now all three scripts call `setup_vggt_path()` right after importing dataset_utils, making the pattern consistent and centralized. Benefits: (1) Removes 9 lines of duplication across 3 scripts, (2) All scripts now use identical/safer path setup logic, (3) Future module reorganizations only need to change one place, (4) Black formatting check passes, tests pass (3/4, with pre-existing failure unrelated to changes).

## 2026-02-14 - Extract Generic Dataset Extraction Pipeline

**Learning:** Callback/strategy pattern is highly effective for eliminating duplication when two codebases share 90% of logic but differ in output file naming or directory structure. Rather than creating multiple variants or conditional logic, passing a simple naming function preserves both readability and flexibility.

**Action:** When encountering duplication in extraction/processing pipelines, identify the invariant logic (iteration, validation, copying) and parameterize only the variant parts (file naming, directory structure). This keeps shared code DRY without sacrificing clarity or creating over-generalized utilities.

**Context:** The aitoolkit-dataset.py and modelscope-dataset.py scripts were nearly identical (70+ lines each) except for how they named output files (AIToolkit used 3 subdirectories with simple naming; ModelScope used flat naming with suffixes). A single generic function with a configurable naming callback eliminated ~100 lines of duplication while making both scripts clearer as thin wrappers.

## 2026-02-14 - Extract Model Utilities for VGGT Inference

**Learning:** VGGT model loading and coordinate transformation logic are universal across inference-based scripts. These pure utility functions should be extracted to shared modules with comprehensive docstrings. Optional imports for torch/VGGT allow graceful degradation when those dependencies aren't needed.

**Action:** When multiple scripts independently repeat the same OpenGL/3D math utilities or model initialization code, extract to dataset_utils.py (even if it's not strictly a "dataset" utility—naming can be refactored later if needed). Add clear docstrings explaining the function's role across multiple scripts.

**Context:** build_warp_dataset.py and vggt_point_cloud_viewer.py both contained identical `load_model(device)` and `build_view_matrix(extrinsic)` functions. Extracted both to dataset_utils.py with docstrings explaining their use in VGGT depth estimation and point cloud rendering. Reduced 30 lines of duplication; both scripts now import from shared utils. Tests pass; no behavior changes.

## 2026-02-14 - Extract Torch Device and Dtype Selection Utilities

**Learning:** Device and dtype selection logic in PyTorch scripts is small (~7 lines each) but highly repetitive and follows a standard pattern: auto-detect CUDA, select bfloat16 for modern GPUs, fallback to float16. Even though the duplication is modest per-script, centralizing this logic ensures consistency and makes future hardware capability changes trivial to update.

**Action:** Extract small but repetitive patterns when they involve complex conditional logic (like GPU capability detection) or when consistency across scripts matters. Don't dismiss extractions just because they're "only 7 lines"—if those 7 lines contain hardware-specific logic that could change, centralization adds value beyond line-count savings.

**Context:** Both build_warp_dataset.py and vggt_point_cloud_viewer.py duplicated device selection (`device_arg or cuda if available else cpu`) and dtype selection (bfloat16 for compute capability >= 8, else float16). Extracted `select_device()` and `select_dtype()` to dataset_utils.py with comprehensive docstrings. Eliminated 14 lines of duplication across 2 scripts. Benefits: (1) Consistency: all scripts now use identical device/dtype logic, (2) Maintainability: hardware capability logic centralized, (3) Future-proof: new scripts can import these utilities immediately.

## 2026-02-14 - Standardize Test Image Access with Fixtures

**Learning:** Test infrastructure benefits as much from DRY principles as production code. When tests each reference images differently (hardcoded paths, command-line args, synthetic data), it creates friction and inconsistency. A centralized fixture for test image access makes tests more portable and maintainable.

**Action:** When adding test data folders, immediately create pytest fixtures in conftest.py to provide consistent access. Consider multiple fixtures: (1) directory fixture for flexibility, (2) default image fixture for convenience. Document which test images are available and their intended use cases.

**Context:** User added `tests/input-img/` folder with test images in subdirectories (01/ with HEIC files, 02/ with PNG files). test_sky_filter.py was broken because it expected an `image_path` fixture that didn't exist. Created three new fixtures in conftest.py: `input_img_dir()` for directory access, `test_image_path()` for default test image, and updated test_sky_filter.py to work both as pytest test (using fixture) and standalone script (taking CLI arg). Benefits: (1) Fixed broken test, (2) Consistent test image access pattern for future tests, (3) Tests can now discover available test images programmatically.


