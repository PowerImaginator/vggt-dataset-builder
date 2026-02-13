## Dataset builder

- Make sure to clone this repo with `git clone --recurse-submodules`
- Create and activate a new virtualenv, then run `pip install -r vggt/requirements.txt`, then run `pip install -r vggt/requirements_demo.txt`, then run `pip install moderngl glcontext pyglet`
- Run `PYTHONPATH=vggt python build_warp_dataset.py`, optionally with arguments to customize behavior.
- (Settings I'm experimenting with currently: `PYTHONPATH=vggt python build_warp_dataset.py --resize-width 1216 --resize-height 832 --sigma 12 --upsample-depth --auto-s0`, plus `--auto-skip --limit 10` if working with DL3DV)

## Viewer

- `PYTHONPATH=vggt python vggt_point_cloud_viewer.py [input image filenames] --output [output image filename]`
- You can use `--viewport-width` and `--viewport-height` to set the output resolution of the viewport and output image.
- You can also use `--resize-width` and `--resize-height` to upsample the point cloud; when you do, it behaves as if `--upsample-depth` is on, as there's no reason otherwise to do this in the viewer app.
- `--sigma` and `--auto-s0` are also available, as in the builder.
