# NeRF-PyTorch - Neural Radiance Fields PyTorch Implementation
Vanilla NeRF implementation using PyTorch

## Overview

This repository contains an implementation of Neural Radiance Fields (NeRF) using PyTorch. NeRF is a method for synthesizing novel views of complex 3D scenes by optimizing a continuous volumetric scene representation using deep learning.

## Features

- Implements vanilla NeRF using PyTorch
- Supports positional encoding for improved performance
- Uses ray marching for volume rendering
- Trains on custom datasets and synthetic datasets
- Supports rendering of novel views from trained models

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- Pillow

<!-- ### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/nerf-pytorch.git
cd nerf-pytorch
pip install -r requirements.txt
``` -->

## Usage

### Training

To train NeRF on a dataset:

```bash
python main.py
```

<!-- ### Rendering Novel Views

After training, render new views:

```bash
python render.py --checkpoint path/to/trained_model.pth --output path/to/output_images
``` -->

### Dataset Preparation

Ensure your dataset follows the NeRF standard:

- A `transforms_<split>.json` file specifying camera poses where split = train, test, val
- Images stored in a directory
- Example dataset structures can be found in the `data/` folder

## Model Details

- Uses an MLP to predict color and density values for queried 3D points
- Employs hierarchical volume sampling for better efficiency
- Implements positional encoding for high-frequency details

## Results

Will include the results soon.
<!-- After training, you can expect high-quality novel view synthesis from the trained model. Example output images and videos can be found in the `outputs/` directory. -->

## References

- [Original NeRF Paper](https://arxiv.org/abs/2003.08934)
- [Reference repo - yenchenlin](https://github.com/yenchenlin/nerf-pytorch/tree/master)
- [Reference repo - krrish94](https://github.com/krrish94/nerf-pytorch)

## License

This project is licensed under the MIT License.

## Contact

For questions and contributions, feel free to open an issue or reach out via GitHub.
