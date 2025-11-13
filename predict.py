# Prediction interface for Cog ⚙️
# https://cog.run/python

import os
import time
import torch
import mimetypes
from PIL import Image
from typing import List, Iterator
from diffusers import QwenImageEditPlusPipeline
from cog import BasePredictor, Input, Path

mimetypes.add_type("image/webp", ".webp")

MODEL_NAME = "Qwen/Qwen-Image-Edit-2509"
MODEL_CACHE = "checkpoints"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load Qwen-Image-Edit-2509 model"""
        print(f"[~] Loading model: {MODEL_NAME}")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.bfloat16,
            cache_dir=MODEL_CACHE,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        print("[+] Model loaded successfully")

    def predict(
        self,
        image1: Path = Input(description="First input image (required)"),
        image2: Path = Input(description="Second input image (optional, for multi-image editing)", default=None),
        image3: Path = Input(description="Third input image (optional, for multi-image editing)", default=None),
        prompt: str = Input(description="Text prompt describing the desired edit"),
        negative_prompt: str = Input(
            description="Negative prompt to guide what to avoid",
            default=" "
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps. Recommended: 40 for Qwen-Image-Edit-2509",
            default=40,
            ge=1,
            le=100,
        ),
        guidance_scale: float = Input(
            description="Guidance scale for image generation",
            default=1.0,
            ge=1.0,
            le=20.0,
        ),
        true_cfg_scale: float = Input(
            description="True CFG scale parameter for Qwen model",
            default=4.0,
            ge=1.0,
            le=10.0,
        ),
        num_images_per_prompt: int = Input(
            description="Number of images to generate per prompt",
            default=1,
            ge=1,
            le=4,
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed",
            default=None
        ),
        output_format: str = Input(
            description="Format of the output image",
            choices=["webp", "jpg", "png"],
            default="png",
        ),
        output_quality: int = Input(
            description="Quality of the output image, from 0 to 100. 100 is best quality, 0 is lowest quality.",
            default=95,
            ge=0,
            le=100,
        ),
    ) -> Iterator[Path]:
        """
        Run Qwen-Image-Edit-2509 for single or multi-image editing
        Supports 1-3 input images for optimal performance
        """
        if not prompt:
            raise ValueError("Please enter a text prompt.")

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        print(f"[~] Using seed: {seed}")

        # Load input images
        images = [Image.open(image1).convert("RGB")]
        num_images = 1

        if image2 is not None:
            images.append(Image.open(image2).convert("RGB"))
            num_images += 1

        if image3 is not None:
            images.append(Image.open(image3).convert("RGB"))
            num_images += 1

        print(f"[~] Processing {num_images} input image(s)")

        # For compatibility: single image as object, multiple as list
        pipeline_input = images if num_images > 1 else images[0]

        # Generate images
        generator = torch.Generator(device=self.device).manual_seed(seed)

        print("[~] Running inference...")
        with torch.inference_mode():
            result = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pipeline_input,
                generator=generator,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                true_cfg_scale=true_cfg_scale,
                num_images_per_prompt=num_images_per_prompt,
            )

        # Save and yield results
        extension = output_format.lower()
        extension = "jpeg" if extension == "jpg" else extension

        for i, output_image in enumerate(result.images):
            output_path = f"/tmp/output_{i}.{extension}"

            print(f"[~] Saving to {output_path}...")
            save_params = {"format": extension.upper()}
            if output_format != "png":
                save_params["quality"] = output_quality
                save_params["optimize"] = True

            output_image.save(output_path, **save_params)
            yield Path(output_path)

        print("[+] Generation completed successfully")