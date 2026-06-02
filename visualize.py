import os
import cv2
import random
import argparse
import matplotlib.pyplot as plt

from utils.model_utils import (
    infer_image_pair,
    load_model
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        required=True
    )

    args = parser.parse_args()

    model, config, device = load_model(args.config)


    root = f'{config["DATASET"]["ROOT"]}/test'
    pre_dir = os.path.join(root, "pre-event")
    post_dir = os.path.join(root, "post-event")
    mask_dir = os.path.join(root, "target")

    save_dir = "outputs/visualizations"
    os.makedirs(save_dir, exist_ok=True)

    files = sorted(os.listdir(pre_dir))
    random_files = random.sample(files, min(5, len(files)))
    patch_size = config["DATASET"]["PATCH_SIZE"]

    for idx, file_name in enumerate(random_files):
        pre_path = os.path.join(pre_dir, file_name)
        post_path = os.path.join(post_dir, file_name)
        mask_path = os.path.join(mask_dir, file_name)

        pre_image = cv2.imread(pre_path)
        post_image = cv2.imread(post_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        pre_image = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
        post_image = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

        pre_image = cv2.resize(pre_image, (patch_size, patch_size))
        post_image = cv2.resize(post_image, (patch_size, patch_size))
        mask = cv2.resize(mask, (patch_size, patch_size))

        prediction = infer_image_pair(
            model,
            config,
            pre_path,
            post_path,
            device
        )

        fig, axes = plt.subplots(
            1,
            4,
            figsize=(20, 5)
        )

        axes[0].imshow(pre_image)
        axes[0].set_title("Pre Image")
        axes[0].axis("off")

        axes[1].imshow(post_image)
        axes[1].set_title("Post Image")
        axes[1].axis("off")

        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

        axes[3].imshow(prediction, cmap="gray")
        axes[3].set_title("Prediction")
        axes[3].axis("off")

        save_path = os.path.join(
            save_dir,
            f"prediction_{idx+1}.png"
        )

        plt.savefig(save_path)
        plt.close()

        print(f"Saved: {save_path}")

    print("✅ Visualization completed")


if __name__ == "__main__":
    main()