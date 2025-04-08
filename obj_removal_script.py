import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, resize, gaussian_blur
import os
from PIL import Image

# Define paths directly within the script
INPUT_DIR = "/scratch/network/jy3107/IW/inputs"
MASK_DIR = "/scratch/network/jy3107/IW/mask_images"
OUTPUT_DIR = "/scratch/network/jy3107/IW/outputs"
RESIZED_INPUT_DIR = "/scratch/network/jy3107/IW/resized_inputs"
os.makedirs(RESIZED_INPUT_DIR, exist_ok=True)

# Set device and data type
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32  

# Custom pipeline path (ensure the file exists)
CUSTOM_PIPELINE_PATH = "/scratch/network/jy3107/IW/AttentiveEraser-master/pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py"

# Load the pipeline
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    custom_pipeline=CUSTOM_PIPELINE_PATH,
    scheduler=scheduler,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
).to(device="cuda")  

# pipeline = torch.compile(pipeline)


def preprocess_image(image_path, device):
    print("Preprocessing image")
    image = load_image(image_path)
    image_resized = image.resize((1024, 1024), Image.LANCZOS)  # Resize using PIL

    # Get the image ID (basename without extension)
    image_name = os.path.basename(image_path)
    image_id = os.path.splitext(image_name)[0]  # e.g., DW0001 from DW0001.jpg

    # Create the new resized file name with "_resized" suffix after the ID
    resized_filename = f"{image_id}_resized{os.path.splitext(image_name)[1]}"
    
    # Define the resized image path
    resized_path = os.path.join(RESIZED_INPUT_DIR, resized_filename)
    image_resized.save(resized_path)
    print(f"Saved resized input image: {resized_path}")

    # Convert the resized image to a tensor
    image_tensor = to_tensor(image_resized).unsqueeze_(0).float()
    if image_tensor.shape[1] != 3:
        image_tensor = image_tensor.expand(-1, 3, -1, -1)
    return image_tensor.to(dtype).to(device)



def preprocess_mask(mask_path, device):
    print("Preprocessing mask")
    mask = load_image(mask_path, convert_method=lambda img: img.convert('L'))
    mask = mask.resize((1024, 1024), Image.LANCZOS)  # Resize using PIL
    mask = to_tensor(mask).unsqueeze_(0).float()
    mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = torch.clamp(mask, 0, 1)
    return mask.to(dtype).to(device)


def remove_obj(input_image_path, mask_paths, output_dir):
    """
    Processes a single image and removes objects using the Attentive Eraser pipeline.
    Accepts a list of mask image paths and saves outputs directly in the output directory.
    """
    print("Removing object")
    image_basename = os.path.splitext(os.path.basename(input_image_path))[0]

    # Load and preprocess source image
    source_image = preprocess_image(input_image_path, device)

    for mask_path in mask_paths:
        mask_filename = os.path.basename(mask_path)
        mask_basename = os.path.splitext(mask_filename)[0]

        try:
            Image.open(mask_path)  # Validate mask
        except Exception as e:
            print(f"Error processing mask {mask_filename}: {e}")
            continue

        # Preprocess mask and run pipeline
        prompt = ""  # No prompt needed for object removal
        generator = torch.Generator(device=device).manual_seed(123)
        mask = preprocess_mask(mask_path, device)

        image = pipeline(
            prompt=prompt,
            image=source_image,
            mask_image=mask,
            height=1024,
            width=1024,
            AAS=True,
            strength=0.8,
            rm_guidance_scale=9,
            ss_steps=9,
            ss_scale=0.3,
            AAS_start_step=0,
            AAS_start_layer=34,
            AAS_end_layer=70,
            num_inference_steps=50,
            generator=generator,
            guidance_scale=1,
        )

        output_image = image.images[0]

        # Save output directly in OUTPUT_DIR using combined name
        output_filename = f"{image_basename}_CO.jpg"
        output_image.save(os.path.join(output_dir, output_filename))
        print(f"Saved output: {output_filename}")




def main():
    """
    Main function to recursively process all images in nested INPUT_DIR.
    """
    # Step 1: Collect all masks by ID prefix
    masks_by_id = {}
    for mask_filename in os.listdir(MASK_DIR):
        mask_path = os.path.join(MASK_DIR, mask_filename)
        mask_id = os.path.splitext(mask_filename)[0].split("_")[0]  # e.g., DW0001 from DW0001_wheelchair.jpg
        if mask_id not in masks_by_id:
            masks_by_id[mask_id] = []
        masks_by_id[mask_id].append(mask_path)

    # Step 2: Recursively find all images in INPUT_DIR
    for root, _, files in os.walk(INPUT_DIR):
        for filename in files:
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue  # skip non-image files

            input_path = os.path.join(root, filename)
            image_id = os.path.splitext(filename)[0]  # e.g., DW0001 from DW0001.jpg

            try:
                Image.open(input_path)  # Validate image
            except Exception as e:
                print(f"Skipping invalid image {filename}: {e}")
                continue

            # Step 3: Match masks by ID prefix
            matching_masks = masks_by_id.get(image_id)
            if not matching_masks:
                print(f"No matching masks found for {filename}, skipping.")
                continue

            remove_obj(input_path, matching_masks, OUTPUT_DIR)



if __name__ == "__main__":
    main()
