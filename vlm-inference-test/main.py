import argparse
from inference.inferencer import Inferencer

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-VL Inference CLI")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", default="Describe the image.", help="Text prompt for inference")
    parser.add_argument("--config", default="configs/qwen2_5_vl.yaml", help="Path to YAML config")

    args = parser.parse_args()

    inferencer = Inferencer(args.config)
    result = inferencer.predict(args.image, args.prompt)

    print("\nModel Output:\n", result)

if __name__ == "__main__":
    main()
