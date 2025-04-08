import os
import csv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# ----------------------------
# Config & Paths
# ----------------------------

INPUT_DIR = "/scratch/network/jy3107/IW/inputs"
RESIZED_DIR = "/scratch/network/jy3107/IW/resized_inputs"
OUTPUT_DIR = "/scratch/network/jy3107/IW/outputs"
CSV_OUTPUT = "/scratch/network/jy3107/IW/blip_captions.csv"
HF_CACHE_DIR = "/scratch/network/jy3107/hf_cache"

# ----------------------------
# Model Setup
# ----------------------------

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir=HF_CACHE_DIR
)
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    cache_dir=HF_CACHE_DIR
).to("cuda")

# ----------------------------
# Helper Functions
# ----------------------------

def generate_caption(image_path):
    """Generate a caption for a single image using BLIP."""
    try:
        raw_image = Image.open(image_path).convert("RGB")
        inputs = processor(raw_image, return_tensors="pt").to("cuda")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "ERROR"

def get_all_image_paths(directory, extensions={".jpg", ".jpeg", ".png"}):
    """Recursively collects image file paths from a directory and sorts them."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)

def run_captioning_and_save_csv(resized_dir, output_dir, csv_path):
    """Captions resized input and output images, saving results to a CSV."""
    resized_images = get_all_image_paths(resized_dir)
    output_images = get_all_image_paths(output_dir)

    print(f"Found {len(resized_images)} resized input images.")
    print(f"Found {len(output_images)} output images.")

    all_rows = []

    # Process resized input images
    for img_path in resized_images:
        caption = generate_caption(img_path)

        # Extract base ID (remove '_resized' and change extension to .jpg)
        filename = os.path.basename(img_path)
        id_part = filename.rsplit("_resized", 1)[0]
        cleaned_name = f"{id_part}.jpg"

        all_rows.append([cleaned_name, caption])
        print(f"Captioned {cleaned_name}: {caption}")

    # Process output images
    for img_path in output_images:
        caption = generate_caption(img_path)
        img_name = os.path.basename(img_path)
        all_rows.append([img_name, caption])
        print(f"Captioned {img_name}: {caption}")

    # Save to CSV
    with open(csv_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "Caption"])
        writer.writerows(all_rows)


# ----------------------------
# Entry Point
# ----------------------------

if __name__ == "__main__":
    run_captioning_and_save_csv(RESIZED_DIR, OUTPUT_DIR, CSV_OUTPUT)
