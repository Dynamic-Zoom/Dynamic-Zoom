# Dynamic-Zoom

## Bicubic plus plus - by Aselsan Research group.

This is the 3x super resolution model proposed in 2023 Super-resolution challenge. 
- Link: https://github.com/aselsan-research-imaging-team/bicubic-plusplus
- Pretrained model weights is in `weights\bicubic_pp_x3.pth`

## Usage
- Packages needed:
    - Pytorch
    - Scipy
    - Open-CV
    - Cudacanvas
    - Numpy
    - Tqdm
    - Pillow
    - Matplotlib
    - Scipy
- Install these packages and it should run with CUDA enabled devices. 

- `bicubic_plus_plus.ipynb` demonstrates the usage of the workflow.
- `Bicubic Plus Plus` gives about `115` FPS for 720p to 4K video super resolution and about `400` FPS for 360p to 1080p super resolution on a GTX 1070.
- Place the video in `test_videos/` folder and change the video path name in `bicubic_plus_plus.ipynb`. 
    - The bunny test video can be found in https://peach.blender.org/download/


