import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from diffusers.utils import load_image
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, gaussian_blur
import os
from PIL import Image

# Define paths directly within the script
INPUT_DIR = "/home/jy3107/IW/test"
MASK_DIR = "/home/jy3107/IW/mask_images"
OUTPUT_DIR = "/home/jy3107/IW/outputs"

# Set device and data type
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
dtype = torch.float16 if torch.cuda.is_available() else torch.float32  

# Custom pipeline path (ensure the file exists)
CUSTOM_PIPELINE_PATH = "/home/jy3107/IW/AttentiveEraser-master/pipelines/pipeline_stable_diffusion_xl_attentive_eraser.py"

# Load the pipeline
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    custom_pipeline=CUSTOM_PIPELINE_PATH,
    scheduler=scheduler,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float32,
).to(device)


def preprocess_image(image_path, device):
    image = to_tensor(load_image(image_path)).unsqueeze_(0).float()
    if image.shape[1] != 3:
        image = image.expand(-1, 3, -1, -1)
    image = F.interpolate(image, (2048, 2048))
    return image.to(dtype).to(device)


def preprocess_mask(mask_path, device):
    mask = to_tensor(load_image(mask_path, convert_method=lambda img: img.convert('L'))).unsqueeze_(0).float()
    mask = F.interpolate(mask, (2048, 2048))
    mask = gaussian_blur(mask, kernel_size=(77, 77))
    mask[mask < 0.1] = 0
    mask[mask >= 0.1] = 1
    mask = torch.clamp(mask, 0, 1)
    return mask.to(dtype).to(device)


def remove_obj(input_image_path, mask_dir, output_dir):
    """
    Processes a single image and removes objects using the Attentive Eraser pipeline.
    """
    name = os.path.splitext(os.path.basename(input_image_path))[0]
    output_path = os.path.join(output_dir, name)
    os.makedirs(output_path, exist_ok=True)

    # Load and preprocess source image
    source_image = preprocess_image(input_image_path, device)

    # Process each mask in the mask directory
    for mask_filename in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_filename)

        try:
            Image.open(mask_path)  # Validate image
        except Exception as e:
            print(f"Error processing mask {mask_filename}: {e}")
            continue

        # Generate the output using the pipeline
        prompt = ""  # No prompt needed for object removal
        generator = torch.Generator(device=device).manual_seed(123)
        mask = preprocess_mask(mask_path, device)

        image = pipeline(
            prompt=prompt,
            image=source_image.to(dtype),
            mask_image=mask.to(dtype),
            height=2048,
            width=2048,
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
        output_image.save(os.path.join(output_path, mask_filename))
        print(f"Object removal for {mask_filename} completed.")


def main():
    """
    Main function to process all images in the input directory.
    """
    for filename in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, filename)

        try:
            Image.open(input_path)  # Validate image
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

        name, _ = os.path.splitext(filename)
        mask_subdir = os.path.join(MASK_DIR, name)

        if os.path.exists(mask_subdir):
            remove_obj(input_path, mask_subdir, OUTPUT_DIR)
        else:
            print(f"No mask directory found for {filename}")


if __name__ == "__main__":
    main()