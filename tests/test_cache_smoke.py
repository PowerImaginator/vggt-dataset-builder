from pathlib import Path
import importlib.util
import sys
import numpy as np
import pytest

# Load build_warp_dataset module by path to avoid import issues
root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "build_warp_dataset", str(root / "build_warp_dataset.py")
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
import types

# Store original module state to restore later
_original_hole_filling_renderer = sys.modules.get("hole_filling_renderer")

# Provide a minimal dummy for hole_filling_renderer to avoid importing heavy runtime deps
sys.modules["hole_filling_renderer"] = types.SimpleNamespace(HoleFillingRenderer=object)
spec.loader.exec_module(module)

save_frame_cache = module.save_frame_cache
load_frame_cache = module.load_frame_cache
_cache_file_for_image = module._cache_file_for_image

# IMMEDIATELY restore sys.modules to avoid polluting other tests
if _original_hole_filling_renderer is not None:
    sys.modules["hole_filling_renderer"] = _original_hole_filling_renderer
else:
    sys.modules.pop("hole_filling_renderer", None)


def test_cache_smoke():
    """Test frame cache save/load functionality without importing heavy renderer deps."""
    scene_cache = Path(".cache") / "smoke_scene"
    scene_cache.mkdir(parents=True, exist_ok=True)
    cache_path = scene_cache / "test_image.npz"

    frame = {
        "points": np.random.rand(5, 3).astype(np.float32),
        "colors": np.random.rand(5, 3).astype(np.float32),
        "confidences": np.random.rand(5).astype(np.float32),
        "s0": 0.123,
    }

    print("Saving cache to", cache_path)
    save_frame_cache(cache_path, frame)
    loaded = load_frame_cache(cache_path)
    print("Loaded keys:", list(loaded.keys()))
    print("points shape:", loaded["points"].shape)
    print("colors shape:", loaded["colors"].shape)
    print("confidences shape:", loaded["confidences"].shape)
    print("s0:", loaded.get("s0"))

    # Assertions to verify cache works correctly
    assert "points" in loaded
    assert "colors" in loaded
    assert "confidences" in loaded
    assert loaded["points"].shape == (5, 3)
    assert loaded["colors"].shape == (5, 3)
    assert loaded["confidences"].shape == (5,)
    assert loaded["s0"] == 0.123
