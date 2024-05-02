# Dynamic Zoom - Real-time Video Resolution Enhancement

Dive into the advanced super-resolution technology of Dynamic Zoom, a real-time video enhancement system designed for digital media.

# Links

- Website: https://dynamic-zoom.github.io/home/
- Repository: https://github.com/Dynamic-Zoom/Dynamic-Zoom

## Usage

- Ensure Conda (Miniconda/Anaconda) is installed on your system. ([Official Conda website for installation guide](https://conda.io/projects/conda/en/latest/user-guide/install/index.html))
- Run `conda env create -f environment.yml` to install necessary packages.
- Activate your new Conda environment using `conda activate dynamic-zoom`. (Check for appropriate PyTorch version installation to enable GPU acceleration)
- Run `python run.py --help` to check Command line arguments
- Example run: `python run.py --input "data/input.mp4" --output "output/output.mp4"`

## Command line arguments explanation

| Argument         | Description                                  | Supported values                                                                     |
| ---------------- | -------------------------------------------- | ------------------------------------------------------------------------------------ |
| -h, --help       | Show help message and exit                   |                                                                                      |
| -r, --roi        | Define size of your ROI window               | 2 (space separated) numbers: `H W`                                                   |
| --input          | Relative File path of the input file         |                                                                                      |
| --output         | Relative File path of the output file        |                                                                                      |
| -m, --model_name | Select upscaling model (default = bicubic++) | `"bicubic++"` <br> `"srgan2x"` <br> `"srgan4x"`                                      |
| --fps            | Fps of output file (default = 24)            |                                                                                      |
| -v, --verbosity  | Select output verbosity (default = 0)        | 0 = only pipeline stage logs(default) <br> 1 = per frame logs <br> 2 = detailed logs |

## Supported Models

1. Bicubic++ - by Aselsan Research group: https://github.com/aselsan-research-imaging-team/bicubic-plusplus
2. Swift-SRGAN - by Koushik Sivarama Krishnan, Karthik Sivarama Krishnan: https://github.com/Koushik0901/Swift-SRGAN
