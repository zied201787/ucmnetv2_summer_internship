import os
import numpy as np

def load_npy(path):
    print(f"Loading {path} ...")
    return np.load(path)

def main():
    base_dirs = ["data/ISIC2017", "data/ISIC2018"]
    splits = ["train", "val", "test"]

    all_images = []
    all_masks = []

    for base in base_dirs:
        for split in splits:
            img_path = os.path.join(base, f"{split}_images.npy")
            mask_path = os.path.join(base, f"{split}_masks.npy")

            imgs = load_npy(img_path)
            masks = load_npy(mask_path)

            all_images.append(imgs)
            all_masks.append(masks)

    # Concatenate all splits and datasets into single arrays
    combined_images = np.concatenate(all_images, axis=0)
    combined_masks = np.concatenate(all_masks, axis=0)

    # Save combined dataset
    os.makedirs("combined_dataset", exist_ok=True)
    np.save("combined_dataset/images.npy", combined_images)
    np.save("combined_dataset/masks.npy", combined_masks)

    print(f"✅ Combined dataset saved to combined_dataset/images.npy and masks.npy")
    print(f"Total images: {combined_images.shape[0]}")

if __name__ == "__main__":
    main()
