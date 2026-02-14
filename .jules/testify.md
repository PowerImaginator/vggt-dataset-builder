# Testify's Journal 🧪

Critical learnings and testing insights for vggt-dataset-builder.

---

## 2026-02-14 - Module Cache Pollution in test_cache_smoke.py

**Learning:** Setting `sys.modules["hole_filling_renderer"] = mock_object` at module level in one test file pollutes the import cache for all subsequently loaded test files. This caused `test_viewer_smoke.py` to fail with `TypeError: object() takes no arguments` because it imported the mock `object` class instead of the real `HoleFillingRenderer`.

**Action:** When mocking modules in `sys.modules` for isolated testing, IMMEDIATELY restore the original state after loading dependent code. Use explicit cleanup right after setup, not deferred fixtures, to prevent pytest's module collection phase from importing polluted modules.

```python
# BAD: Cleanup in fixture runs too late
sys.modules["module"] = mock
load_code()

@pytest.fixture(autouse=True)
def cleanup():
    yield
    restore()  # Too late - other tests already imported!

# GOOD: Restore immediately after use
original = sys.modules.get("module")
sys.modules["module"] = mock
load_code()
sys.modules["module"] = original  # Clean up NOW
```

---
