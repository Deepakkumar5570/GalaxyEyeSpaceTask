import argparse

from utils.model_utils import (
    infer_image_pair,
    load_model,
    save_prediction
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    parser.add_argument(
        "--pre_image",
        type=str,
        required=True
    )

    parser.add_argument(
        "--post_image",
        type=str,
        required=True
    )

    args = parser.parse_args()

    model, config, device = load_model(args.config)

    prediction = infer_image_pair(
        model,
        config,
        args.pre_image,
        args.post_image,
        device
    )

    save_path = save_prediction(prediction)

    changed_pixels = int((prediction == 1).sum())
    total_pixels = int(prediction.size)
    change_percent = (changed_pixels / total_pixels) * 100

    print("✅ Model loaded successfully")
    print(f"Change Percentage: {change_percent:.2f}%")
    print(f"Prediction saved at: {save_path}")


if __name__ == "__main__":
    main()
