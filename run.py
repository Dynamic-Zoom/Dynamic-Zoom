import argparse
from src.pipeline import run_pipeline


def main():
    # Parse necessary params and pass to run_pipeline

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, default="data/input.mp4", help="input file path"
    )
    parser.add_argument(
        "--output_file", type=str, default="output.mp4", help="output file path"
    )
    parser.add_argument("--fps", type=int, default=24, help="fps of output file")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        choices=["bicubic++", "srgan2x", "srgan4x"],
        default="bicubic++",
        help="select upscaling model",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        type=int,
        choices=[0, 1, 2],
        default=0,
        help="increase output verbosity",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
