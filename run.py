import argparse
from src.pipeline import run_pipeline


def main():
    # Parse necessary params and pass to run_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--roi",
        type=int,
        nargs=2,
        default=(180, 240),
        help="Define size of your ROI window",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input.mp4",
        help="Relative File path of the input file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/output.mp4",
        help="Relative File path of the output file",
    )
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        choices=["bicubic++", "srgan2x", "srgan4x"],
        default="bicubic++",
        help="Select upscaling model (default = bicubic++)",
    )
    parser.add_argument(
        "--fps", type=int, default=24, help="Fps of output file (default = 24)"
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="Select output verbosity: 0=only pipeline stage logs(default); 1=per frame logs; 2=detailed logs;",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
