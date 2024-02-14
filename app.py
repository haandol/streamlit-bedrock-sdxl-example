import io
import re
import os
import json
import base64
from typing import List, Optional

import boto3
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

PROFILE_NAME = os.environ.get("AWS_PROFILE", None)
REGION_NAME = os.environ.get("AWS_REGION", "us-east-1")

SEED = int(os.environ.get("SEED", 329))
MODEL_ID = os.environ.get("MODEL_ID", "stability.stable-diffusion-xl-v1")
SAVE_LOCAL = bool(os.environ.get("SAVE_LOCAL", None))

LATEST_IMAGE_KEY = "latest_image"


def set_latest_image(image: Image.Image):
    st.session_state[LATEST_IMAGE_KEY] = image


def get_latest_image() -> Optional[Image.Image]:
    return st.session_state.get("latest_image", None)


class ImageGenerator:
    def __init__(self):
        session = boto3.Session(
            profile_name=PROFILE_NAME,
            region_name=REGION_NAME,
        )
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
        return image

    def generate_image_from_prompt_and_mask_image(
        self,
        prompt: str,
        init_image: Image.Image,
        mask_image: Image.Image,
        image_strength: float = 0.35,
        negative_prompts: List[str] = [],
        cfg_scale: int = 7.5,
        steps: int = 30,
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
                "seed": SEED,
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
        self.pattern = re.compile(r"[^a-zA-Z0-9 ,.]+")
        self.client = session.client(service_name="translate")

    def translate(self, text: str, target_language: str = "en") -> str:
        if text is None:
            return ""

        # if text doescontains only English letters, return as is
        if not self.pattern.search(text):
            print("text is in English")
            return text

        L = []
        for t in text.split(","):
            response = self.client.translate_text(
                Text=t.strip(),
                SourceLanguageCode="auto",
                TargetLanguageCode=target_language,
            )
            L.append(response.get("TranslatedText", ""))
        return ", ".join(filter(None, L))


def txt2img_tab(sdxl: ImageGenerator = None, translator: Translator = None):
    with st.sidebar:
        add_selectbox = st.sidebar.selectbox(
            "Prompt examples",
            (
                "",
                "A beautiful mountain landscape",
                "화성에서 검은 말을 타고 있는 우주인",
                "スキューバダイビングスーツを着たモデル",  # "A model wearing a scuba diving suit",
            ),
            index=0,
        )
        st.markdown("Use the above drop down box to generate _prompt_ examples")

    prompt = st.text_input(
        "Input the prompt or select one from the left sidebar", key="txt2img-prompt"
    )

    if st.button("Generate image", key="txt2img-btn"):
        prompt = prompt or add_selectbox
        if not prompt:
            st.error("Please input a prompt or select one from the left sidebar")
            return

        orig_prompt = prompt
        print("original prompt: ", orig_prompt)
        prompt = translator.translate(orig_prompt)
        if orig_prompt == prompt:
            st.markdown(f"User prompted: `{prompt}`".strip())
            print("prompt is already in English")
        else:
            st.markdown(f"User prompted: `{orig_prompt}` => `{prompt}`".strip())
            print("translated: ", prompt)

        with st.spinner("Generating image based on prompt"):
            image = sdxl.generate_image_from_prompt(
                prompt=prompt,
            )
            if SAVE_LOCAL:
                filename = f"{prompt.replace(' ', '_')}-{SEED}.png"
                image.save(filename)
            set_latest_image(image.copy())
            st.success("Generated stable diffusion model")
        st.image(image)


def inpainting(image: Image.Image) -> Optional[Image.Image]:
    brush_size = st.slider(
        "Brush size",
        min_value=1,
        max_value=100,
        value=35,
        step=1,
        key="inpainting-brush",
    )
    canvas_result = st_canvas(
        stroke_width=brush_size,
        stroke_color="black",
        background_image=image,
        update_streamlit=True,
        width=image.width,
        height=image.height,
        drawing_mode="freedraw",
        key="inpainting-canvas",
    )
    if canvas_result.image_data is None:
        return None

    mask = canvas_result.image_data
    mask = mask[:, :, -1] > 0
    if mask.sum() > 0:
        mask = Image.fromarray(mask)
        st.image(mask)
        return mask

    return None


def inpainting_tab(sdxl: ImageGenerator = None, translator: Translator = None):
    init_image = get_latest_image()
    if not init_image:
        st.error("Please generate an image first")
        return

    mask_image = inpainting(init_image)
    print("after inpainting: ", init_image, mask_image)

    image_strength = st.slider(
        "Strength of inpainting (1.0 essentially ignores the masked area of the original input image)",
        min_value=0.0,
        max_value=1.0,
        value=0.35,
        step=0.05,
        key="inpainting-strength",
    )

    prompt = st.text_input("Input the prompt", key="inpainting-prompt")

    if st.button("Generate image", key="inpainting-btn"):
        if not prompt:
            st.error("Please input a prompt or select one from the left sidebar")
            return

        orig_prompt = prompt
        print("original prompt: ", orig_prompt)
        prompt = translator.translate(orig_prompt)
        if orig_prompt == prompt:
            st.markdown(f"User prompted: `{prompt}`".strip())
            print("prompt is already in English")
        else:
            st.markdown(f"User prompted: `{orig_prompt}` => `{prompt}`".strip())
            print("translated: ", prompt)

        with st.spinner("Generating image..."):
            image = sdxl.generate_image_from_prompt_and_mask_image(
                prompt=prompt,
                init_image=init_image,
                mask_image=mask_image,
                image_strength=image_strength,
            )
            set_latest_image(image.copy())
            st.success("Generated stable diffusion model")
        st.image(image)


if __name__ == "__main__":
    st.title("SDXL Demo using Amazon Bedrock")
    st.caption("An app to generate images based on text prompts :sunglasses:")

    sdxl = ImageGenerator()
    translator = Translator()

    st.header("Text to Image")
    txt2img_tab(sdxl=sdxl, translator=translator)
    st.divider()
    st.header("Inpainting")
    inpainting_tab(sdxl=sdxl, translator=translator)
