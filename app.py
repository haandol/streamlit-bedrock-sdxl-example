import io
import os
import json
import base64

import boto3
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

SEED = int(os.environ.get("SEED", 329))
MODEL_ID = os.environ.get("MODEL_ID", "stability.stable-diffusion-xl-v1")
SAVE_LOCAL = bool(os.environ.get("SAVE_LOCAL", None))


class ImageGenerator:
    def __init__(self):
        profile_name = os.environ.get("AWS_PROFILE", None)
        region_name = os.environ.get("AWS_REGION", "us-east-1")
        session = boto3.Session(profile_name=profile_name, region_name=region_name)
        self.client = session.client("bedrock-runtime")

    def generate_image_from_prompt(
        self,
        prompt: str,
        negative_prompts: str,
        cfg_scale: int = 5,
        steps: int = 30,
        sampler: str = "K_DPMPP_2M",
        clip_guidance_preset: str = "FAST_GREEN",
        style_preset: str = "photographic",
        width: int = 1024,
    ):
        body = json.dumps(
            {
                "text_prompts": (
                    [{"text": prompt, "weight": 1.0}]
                    + [
                        {"text": negprompt, "weight": -1.0}
                        for negprompt in negative_prompts
                    ]
                ),
                "seed": SEED,
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
            modelId=MODEL_ID,
        )
        response_body = json.loads(response.get("body").read())
        base_64_img_str = response_body["artifacts"][0].get("base64")
        image = Image.open(
            io.BytesIO(base64.decodebytes(bytes(base_64_img_str, "utf-8")))
        )
        if SAVE_LOCAL:
            filename = f"{prompt.replace(' ', '_')}-{SEED}.png"
            image.save(filename)
        return image

    def __str__(self) -> str:
        return f"prompt: {self.prompt}"

    def __len__(self):
        return len(self.prompt)


if __name__ == "__main__":
    st.title("SDXL Demo using Amazon Bedrock")
    st.caption("An app to generate images based on text prompts :sunglasses:")
    with st.sidebar:
        add_selectbox = st.sidebar.selectbox(
            "Prompt examples",
            (
                "",
                "A beautiful mountain landscape",
                "Homer Simpson on a computer wearing a space suit",
                "Mona Lisa with headphones on",
                "A model wearing a scuba diving suit",
                "Optimus Prime on top of a surf board",
            ),
            index=0,
        )
        st.markdown("Use the above drop down box to generate _prompt_ examples")

    prompt = st.text_input("Input the prompt desired")
    if not prompt:
        prompt = add_selectbox

    if prompt:
        st.markdown(
            f"""
This will show an image using **stable diffusion** 
of the desired {prompt} entered:
        """.strip()
        )
        print(prompt)

        image = None
        with st.spinner("Generating image based on prompt"):
            sd = ImageGenerator()
            image = sd.generate_image_from_prompt(
                prompt=prompt,
                negative_prompts=[
                    'ugly,', 'tiling,', 'poorly', 'drawn', 'hands,',
                    'poorly', 'drawn', 'feet,', 'poorly', 'drawn',
                    'face,', 'out', 'of', 'frame,', 'extra', 'limbs,',
                    'disfigured,', 'deformed,', 'body', 'out', 'of',
                    'frame,', 'bad', 'anatomy,', 'watermark,', 'signature,',
                    'cut', 'off,', 'low', 'contrast,', 'underexposed,',
                    'overexposed,', 'bad', 'art,', 'beginner,', 'amateur,',
                    'distorted', 'face'
                ],
            )
            st.success("Generated stable diffusion model")

        if image:
            st.image(image)
