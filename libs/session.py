import streamlit as st
from PIL import Image
from typing import Optional

INIT_IMAGE_KEY = "init_image"
INPAINTED_IMAGE_KEY = "inpainted_image"
MASK_IMAGE_KEY = "mask_image"


def set_init_image(image: Image.Image):
    st.session_state[INIT_IMAGE_KEY] = image


def get_init_image() -> Optional[Image.Image]:
    return st.session_state.get(INIT_IMAGE_KEY, None)


def set_inpainted_image(image: Image.Image):
    st.session_state[INPAINTED_IMAGE_KEY] = image


def get_inpainted_image() -> Optional[Image.Image]:
    return st.session_state.get(INPAINTED_IMAGE_KEY, None)
