import io
import json
import boto3
import base64
from typing import List

from PIL import Image


class ImageGenerator:
    def __init__(
        self,
        profile_name: str,
        region_name: str,
        seed: int,
        model_id: str,
        save_local: bool = False,
    ):
        session = boto3.Session(
            profile_name=profile_name,
            region_name=region_name,
        )
        self.seed = seed
        self.model_id = model_id
        self.save_local = save_local
        self.client = session.client("bedrock-runtime")
        self.universal_positive_prompts = [
            "4k",
            "8k",
            "masterpiece",
            "highly detailed",
            "high resolution",
        ]
        self.universal_negative_prompts = [
            "ugly",
            "tiling",
            "poorly drawn hands",
            "poorly drawn feet",
            "poorly drawn face",
            "out of frame",
            "extra limbs",
            "disfigured",
            "deformed",
            "body out of frame",
            "bad anatomy",
            "watermark",
            "signature",
            "cut off",
            "low contrast",
            "underexposed",
            "overexposed",
            "bad art",
            "beginner",
            "amateur",
            "distorted face",
            "sketch",
            "doodle",
            "blurry",
            "out of focus",
        ]

    def generate_image_from_prompt(
        self,
        prompt: str,
        negative_prompts: List[str] = [],
        cfg_scale: int = 7.5,
        steps: int = 30,
        sampler: str = "K_DPMPP_2M",
        clip_guidance_preset: str = "FAST_GREEN",  # CLIP Guidance only supports ancestral samplers.
        style_preset: str = "photographic",
        width: int = 1024,
    ):
        body = json.dumps(
            {
                "text_prompts": (
                    [
                        {
                            "text": prompt
                            + ", "
                            + ", ".join(self.universal_positive_prompts),
                            "weight": 1.0,
                        }
                    ]
                    + [
                        {"text": negprompt, "weight": -1.0}
                        for negprompt in self.universal_negative_prompts
                        + negative_prompts
                    ]
                ),
                "seed": self.seed,
                "cfg_scale": cfg_scale,
                "steps": steps,
                "sampler": sampler,
                "clip_guidance_preset": clip_guidance_preset,
                "style_preset": style_preset,
                "width": width,
            }
        )
        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
        )
        response_body = json.loads(response.get("body").read())
        base_64_img_str = response_body["artifacts"][0].get("base64")
        image = Image.open(
            io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8")))
        )
        if self.save_local:
            filename = f"{prompt.replace(' ', '_')}-{self.seed}-init.jpg"
            image.save(filename)
        return image

    def generate_image_from_prompt_and_mask_image(
        self,
        prompt: str,
        init_image: Image.Image,
        mask_image: Image.Image,
        image_strength: float = 0.05,
        negative_prompts: List[str] = [],
        cfg_scale: int = 7.5,
        steps: int = 20,
        style_preset: str = "photographic",
        width: int = 1024,
    ):
        mask_buffer = io.BytesIO()
        mask_image.save(mask_buffer, format="PNG")
        b64mask = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")

        init_buffer = io.BytesIO()
        init_image.save(init_buffer, format="PNG")
        b64init = base64.b64encode(init_buffer.getvalue()).decode("utf-8")

        body = json.dumps(
            {
                "text_prompts": (
                    [
                        {
                            "text": prompt
                            + ", "
                            + ", ".join(self.universal_positive_prompts),
                            "weight": 1.0,
                        }
                    ]
                    + [
                        {"text": negprompt, "weight": -1.0}
                        for negprompt in self.universal_negative_prompts
                        + negative_prompts
                    ]
                ),
                "seed": self.seed,
                "init_image": b64init,
                "mask_source": "MASK_IMAGE_WHITE",
                "mask_image": b64mask,
                "image_strength": image_strength,
                "cfg_scale": cfg_scale,
                "style_preset": style_preset,
                "steps": steps,
                "width": width,
            }
        )
        response = self.client.invoke_model(
            body=body,
            modelId=self.model_id,
        )
        response_body = json.loads(response.get("body").read())
        base_64_img_str = response_body["artifacts"][0].get("base64")
        image = Image.open(
            io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8")))
        )
        if self.save_local:
            filename = f"{prompt.replace(' ', '_')}-{self.seed}-inpainting.jpg"
            image.save(filename)
        return image

    def __str__(self) -> str:
        return f"prompt: {self.prompt}"

    def __len__(self):
        return len(self.prompt)
