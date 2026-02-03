# VGGT Dataset Builder

## Setup

1. Clone this repo with submodules:
   ```bash
   git clone --recurse-submodules <repo-url>
   cd vggt-dataset-builder
   ```

2. Create and activate a virtual environment using uv:
   ```bash
   uv venv --python 3.10 --seed
   # On Windows:
   .venv\Scripts\activate
   # On Linux/macOS:
   source .venv/bin/activate
   ```

3. Install PyTorch with CUDA 12.8 support:
   ```bash
   uv pip install torch==2.8.0+cu128 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu128
   ```

4. Install remaining dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

5. Log in to Hugging Face (required for gated model access):
   ```bash
   uv run python -c "from huggingface_hub import login; login()"
   ```

## Usage

Run the dataset builder:
```bash
uv run python build_warp_dataset.py
```

### Example settings:
```bash
uv run python build_warp_dataset.py --resize-width 1216 --resize-height 832 --sigma 12 --upsample-depth --auto-s0
```

For DL3DV datasets, add:
```bash
uv run python build_warp_dataset.py --resize-width 1216 --resize-height 832 --sigma 12 --upsample-depth --auto-s0 --auto-skip --limit 10
```

