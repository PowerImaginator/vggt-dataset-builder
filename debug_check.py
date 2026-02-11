#!/usr/bin/env python3
from pathlib import Path

scene_dir = Path("input/01")
output_dir = Path("output")
output_ext = "jpg"

scene_output_dir = output_dir / scene_dir.name
print(f"Checking: {scene_output_dir}")
print()

for f in sorted(scene_output_dir.glob("*")):
    print(f"{f.name}: exists={f.is_file()}")
print()

# Check what we're looking for
next_name = "IMG_6636_rescaled"
forward_splats = scene_output_dir / f"{next_name}_splats.{output_ext}"
forward_target = scene_output_dir / f"{next_name}_target.{output_ext}"
forward_reference = scene_output_dir / f"{next_name}_reference.{output_ext}"

print(f"Looking for:")
print(f"  {forward_splats.name}: {forward_splats.exists()}")
print(f"  {forward_target.name}: {forward_target.exists()}")  
print(f"  {forward_reference.name}: {forward_reference.exists()}")
