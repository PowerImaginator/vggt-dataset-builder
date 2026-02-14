# Dryer's Journal 🏜️

## 2026-02-14 - Extract Generic Dataset Extraction Pipeline

**Learning:** Callback/strategy pattern is highly effective for eliminating duplication when two codebases share 90% of logic but differ in output file naming or directory structure. Rather than creating multiple variants or conditional logic, passing a simple naming function preserves both readability and flexibility.

**Action:** When encountering duplication in extraction/processing pipelines, identify the invariant logic (iteration, validation, copying) and parameterize only the variant parts (file naming, directory structure). This keeps shared code DRY without sacrificing clarity or creating over-generalized utilities.

**Context:** The aitoolkit-dataset.py and modelscope-dataset.py scripts were nearly identical (70+ lines each) except for how they named output files (AIToolkit used 3 subdirectories with simple naming; ModelScope used flat naming with suffixes). A single generic function with a configurable naming callback eliminated ~100 lines of duplication while making both scripts clearer as thin wrappers.
