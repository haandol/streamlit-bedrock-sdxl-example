import io
import re
import os
import json
import base64

import boto3
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

PROFILE_NAME = os.environ.get("AWS_PROFILE", None)
REGION_NAME = os.environ.get("AWS_REGION", "us-east-1")

SEED = int(os.environ.get("SEED", 329))
MODEL_ID = os.environ.get("MODEL_ID", "stability.stable-diffusion-xl-v1")
SAVE_LOCAL = bool(os.environ.get("SAVE_LOCAL", None))


class ImageGenerator:
    def __init__(self):
        session = boto3.Session(
            profile_name=PROFILE_NAME,
            region_name=REGION_NAME,
        )
        self.client = session.client("bedrock-runtime")
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


class Translator(object):
    def __init__(self):
        session = boto3.Session(
            profile_name=PROFILE_NAME,
            region_name=REGION_NAME,
        )
        self.pattern = re.compile(r'[^a-zA-Z0-9 ,.]+')
        self.client = session.client(service_name='translate')

    def translate(self, text: str, target_language: str = "en") -> str:
        if text is None:
            return ""
        
        # if text doescontains only English letters, return as is
        if not self.pattern.search(text):
            print('text is in English')
            return text

        response = self.client.translate_text(
            Text=text,
            SourceLanguageCode="auto",
            TargetLanguageCode=target_language,
        )
        return response.get("TranslatedText")


if __name__ == "__main__":
    sdxl = ImageGenerator()
    translator = Translator()

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
        print('source: ', prompt)
        prompt = translator.translate(prompt)
        st.markdown(
            f"""
This will show an image using **stable diffusion** 
of the desired {prompt} entered:
        """.strip()
        )
        print('translated: ', prompt)

        image = None
        with st.spinner("Generating image based on prompt"):
            image = sdxl.generate_image_from_prompt(
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
