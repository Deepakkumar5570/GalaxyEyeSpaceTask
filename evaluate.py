
import os
import warnings
import cv2
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

from models.attention.attention_siamese_unet import AttentionSiameseUNet


# Configuration
DATASET_PATH = r"C:\Users\lenovo\Downloads\GalexyEye"
MODEL_PATH = r"C:\Users\lenovo\Downloads\best_attention_model.pth"

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing Device: {DEVICE}")

# Paths
PRE_DIR = os.path.join(DATASET_PATH, "test", "pre-event")
POST_DIR = os.path.join(DATASET_PATH, "test", "post-event")
MASK_DIR = os.path.join(DATASET_PATH, "test", "target")

# Load model
print("\nLoading Model...\n")
model = AttentionSiameseUNet()
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state_dict)
model = model.to(DEVICE)
model.eval()
print("✅ Model Loaded Successfully!")

# Prepare dataset
files = sorted([f for f in os.listdir(PRE_DIR) if f.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))])
random_files = random.sample(files, 5)
print(f"\nRandom Samples Selected: {len(random_files)}")

# Output directory
SAVE_DIR = "prediction_results"
os.makedirs(SAVE_DIR, exist_ok=True)
PATCH_SIZE = 256


# Inference loop
for idx, file_name in enumerate(random_files):
    print("\n" + "=" * 80)
    print(f"Processing Image {idx+1} | File: {file_name}")

    # Load and preprocess images
    pre_path = os.path.join(PRE_DIR, file_name)
    post_path = os.path.join(POST_DIR, file_name)
    mask_path = os.path.join(MASK_DIR, file_name)

    pre_image = cv2.imread(pre_path, cv2.IMREAD_COLOR)
    post_image = cv2.imread(post_path, cv2.IMREAD_COLOR)
    actual_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Convert BGR to RGB and resize
    pre_rgb = cv2.cvtColor(pre_image, cv2.COLOR_BGR2RGB)
    post_rgb = cv2.cvtColor(post_image, cv2.COLOR_BGR2RGB)

    pre_resized = cv2.resize(pre_rgb, (PATCH_SIZE, PATCH_SIZE))
    post_resized = cv2.resize(post_rgb, (PATCH_SIZE, PATCH_SIZE))
    actual_mask = cv2.resize(actual_mask, (PATCH_SIZE, PATCH_SIZE))

    # Normalize and convert to tensors
    pre_tensor = torch.tensor(pre_resized).permute(2, 0, 1).float() / 255.0
    post_tensor = torch.tensor(post_resized).permute(2, 0, 1).float() / 255.0
    pre_tensor = pre_tensor.unsqueeze(0).to(DEVICE)
    post_tensor = post_tensor.unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        logits = model(pre_tensor, post_tensor)
        prediction = torch.argmax(logits, dim=1)

    # Convert to numpy
    prediction = prediction.squeeze().cpu().numpy().astype(np.uint8)
    print(f"Unique Prediction Values: {np.unique(prediction)}")

    # Calculate change percentage
    changed_pixels = np.sum(prediction == 1)
    total_pixels = prediction.size
    change_percent = (changed_pixels / total_pixels) * 100
    predicted_class = "CHANGE DETECTED" if change_percent > 5 else "NO SIGNIFICANT CHANGE"
    
    print(f"Predicted Class: {predicted_class}")
    print(f"Change Percentage: {change_percent:.2f}%")

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(pre_resized)
    axes[0].set_title("Pre Image")
    axes[0].axis("off")

    axes[1].imshow(post_resized)
    axes[1].set_title("Post Image")
    axes[1].axis("off")

    axes[2].imshow(actual_mask, cmap="gray")
    axes[2].set_title("Actual Mask")
    axes[2].axis("off")

    axes[3].imshow(prediction, cmap="gray")
    axes[3].set_title(f"Prediction\n{predicted_class}\nChange: {change_percent:.2f}%")
    axes[3].axis("off")

    plt.suptitle(f"Image: {file_name}", fontsize=14)
    plt.tight_layout()

    # Save and display
    save_path = os.path.join(SAVE_DIR, f"prediction_{idx+1}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved Result: {save_path}")

print("\n" + "=" * 80)
print("✅ ALL INFERENCE COMPLETED!")
