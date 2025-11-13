"""
Example: Multi-Image Editing with Qwen-Image-Edit-2509
Demonstrates how to use the pipeline with multiple input images
"""

import os
import torch
from PIL import Image
from diffusers import QwenImageEditPlusPipeline

# Initialize pipeline
print("[~] Loading Qwen-Image-Edit-2509 pipeline...")
pipeline = QwenImageEditPlusPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",
    torch_dtype=torch.bfloat16
)
print("[+] Pipeline loaded")

pipeline.to('cuda')
pipeline.set_progress_bar_config(disable=None)

# Example 1: Multi-Image Input (2 images)
print("\n" + "="*60)
print("Example 1: Person + Person Composition")
print("="*60)

# Load your images (replace with actual paths)
image1 = Image.open("input1.png")  # First person
image2 = Image.open("input2.png")  # Second person

prompt = "The magician bear is on the left, the alchemist bear is on the right, facing each other in the central park square."

inputs = {
    "image": [image1, image2],
    "prompt": prompt,
    "generator": torch.manual_seed(0),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

print(f"[~] Prompt: {prompt}")
print(f"[~] Processing {len(inputs['image'])} images...")

with torch.inference_mode():
    output = pipeline(**inputs)
    output_image = output.images[0]
    output_path = "output_multi_image_example.png"
    output_image.save(output_path)
    print(f"[+] Image saved: {os.path.abspath(output_path)}")


# Example 2: Single Image Edit
print("\n" + "="*60)
print("Example 2: Single Image Text Edit")
print("="*60)

image_single = Image.open("input1.png")
prompt_single = "Add a wizard hat to the bear"

inputs_single = {
    "image": image_single,  # Single image (not in list)
    "prompt": prompt_single,
    "generator": torch.manual_seed(42),
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}

print(f"[~] Prompt: {prompt_single}")
print("[~] Processing single image...")

with torch.inference_mode():
    output_single = pipeline(**inputs_single)
    output_image_single = output_single.images[0]
    output_path_single = "output_single_image_example.png"
    output_image_single.save(output_path_single)
    print(f"[+] Image saved: {os.path.abspath(output_path_single)}")


# Example 3: Person + Product + Scene (3 images)
print("\n" + "="*60)
print("Example 3: Three Image Composition")
print("="*60)

try:
    image_person = Image.open("person.png")
    image_product = Image.open("product.png")
    image_scene = Image.open("scene.png")

    prompt_three = "Professional photo of person holding the product in the scenic location"

    inputs_three = {
        "image": [image_person, image_product, image_scene],
        "prompt": prompt_three,
        "generator": torch.manual_seed(123),
        "true_cfg_scale": 4.0,
        "negative_prompt": "blurry, low quality",
        "num_inference_steps": 50,  # Higher steps for complex composition
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
    }

    print(f"[~] Prompt: {prompt_three}")
    print(f"[~] Processing {len(inputs_three['image'])} images...")

    with torch.inference_mode():
        output_three = pipeline(**inputs_three)
        output_image_three = output_three.images[0]
        output_path_three = "output_three_images_example.png"
        output_image_three.save(output_path_three)
        print(f"[+] Image saved: {os.path.abspath(output_path_three)}")

except FileNotFoundError as e:
    print(f"[!] Skipping Example 3: {e}")
    print("[!] To run this example, provide person.png, product.png, and scene.png")


print("\n" + "="*60)
print("All examples completed!")
print("="*60)
print("\nTips for best results:")
print("  - Use 1-3 images for optimal performance")
print("  - 40-50 inference steps recommended")
print("  - true_cfg_scale=4.0 works well for most cases")
print("  - Be descriptive in prompts about image positions and interactions")
