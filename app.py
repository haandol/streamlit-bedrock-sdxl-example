import os
from typing import Optional

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from dotenv import load_dotenv

from libs.session import (
    set_init_image,
    get_init_image,
    set_inpainted_image,
    get_inpainted_image,
)
from libs.image_generator import ImageGenerator
from libs.translator import Translator

load_dotenv()

PROFILE_NAME = os.environ.get("AWS_PROFILE", None)
REGION_NAME = os.environ.get("AWS_REGION", "us-east-1")

SEED = int(os.environ.get("SEED", 329))
MODEL_ID = os.environ.get("MODEL_ID", "stability.stable-diffusion-xl-v1")
SAVE_LOCAL = bool(os.environ.get("SAVE_LOCAL", None))

sdxl = ImageGenerator(
    profile_name=PROFILE_NAME,
    region_name=REGION_NAME,
    seed=SEED,
    model_id=MODEL_ID,
    save_local=SAVE_LOCAL,
)
translator = Translator(profile_name=PROFILE_NAME, region_name=REGION_NAME)


def txt2img_section():
    st.header("Text to Image")

    prompt = st.text_input(
        "Input the prompt or select one from the left sidebar", key="txt2img-prompt"
    )

    if st.button("Generate image", key="txt2img-btn"):
        prompt = prompt or st.session_state.get("selectbox", None)
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
        set_init_image(image.copy())
        st.success("Generated image based on prompt")


def inpainting(image: Image.Image) -> Optional[Image.Image]:
    brush_size = st.slider(
        "Brush size",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        key="inpainting-brush",
    )
    st.subheader("Draw on the image to mask the area to inpaint")
    canvas_result = st_canvas(
        stroke_width=brush_size,
        stroke_color="white",
        background_image=image,
        update_streamlit=True,
        width=600,
        height=600,
        drawing_mode="freedraw",
        key="inpainting-canvas",
    )
    if canvas_result.image_data is None:
        return None

    mask = canvas_result.image_data
    mask = mask[:, :, -1] > 0
    if mask.sum() > 0:
        return Image.fromarray(mask).resize((image.width, image.height))

    return None


def inpainting_section(init_image: Optional[Image.Image] = None):
    st.header("Inpainting")

    if not init_image:
        st.error("Please generate an image first")
        return

    mask_image = inpainting(init_image)
    if mask_image:
        st.subheader("Masked area")
        st.image(mask_image)

    image_strength = st.slider(
        "Strength of inpainting (1.0 essentially ignores the masked area of the original input image)",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.01,
        key="inpainting-strength",
    )

    prompt = st.text_input("Input the prompt", key="inpainting-prompt")

    if st.button("Generate inpainted image", key="inpainting-btn"):
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

        with st.spinner("Generating inpainted image..."):
            image = sdxl.generate_image_from_prompt_and_mask_image(
                prompt=prompt,
                init_image=init_image,
                mask_image=mask_image,
                image_strength=image_strength,
            )
        set_inpainted_image(image.copy())


if __name__ == "__main__":
    st.title("SDXL Demo using Amazon Bedrock")
    st.caption("An app to generate images based on text prompts :sunglasses:")

    with st.sidebar:
        st.selectbox(
            "Prompt examples",
            (
                "",
                "A beautiful mountain landscape",
                "화성에서 검은 말을 타고 있는 우주인",
                "スキューバダイビングスーツを着たモデル",  # "A model wearing a scuba diving suit",
            ),
            index=0,
            key="selectbox",
        )
        st.markdown("Use the above drop down box to generate _prompt_ examples")

    txt2img_section()
    init_image = get_init_image()
    if init_image:
        st.subheader("Generated image")
        st.image(init_image)

    st.divider()

    inpainting_section(init_image)
    inpainted_image = get_inpainted_image()
    if inpainted_image:
        st.subheader("Inpainted image")
        st.image(inpainted_image)
