# Dynamic-Zoom

## Bicubic plus plus - by Aselsan Research group.

This is the 3x super resolution model proposed in 2023 Super-resolution challenge. 
- Link: https://github.com/aselsan-research-imaging-team/bicubic-plusplus
- Pretrained model weights is in `weights\bicubic_pp_x3.pth`

### TODO items:

- [x] Base file structure
- [ ] Define conda + virtual environment files for installation
- [ ] Test FrameBuffer functionality
- [ ] Test FrameBuffer integration with ModelExecutor and FileWriter
- [ ] InputStream Implementation (validate corner cases for parallel execution)
- [ ] OutputStream Implementation (validate corner cases for parallel execution)
- [ ] Write E2E pipeline
- [ ] CLI run arguments
- [ ] Sanity unit tests (Optional)

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

- `Bicubic Plus Plus` gives about `115` FPS for 720p to 4K video super resolution and about `400` FPS for 360p to 1080p super resolution on a GTX 1070.
- The test bunny video can be found at https://www.peach.themazzone.com/big_buck_bunny_720p_h264.mov. Explore other videos in https://peach.blender.org/download/


