import streamlit as st  
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np
from my_utils import get_result
import warnings
import time
warnings.filterwarnings("ignore")

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
    page_title="Japanese Character Detection",
    page_icon = ":mango:",
    initial_sidebar_state = 'auto'
)

# seg_model = 
with st.sidebar:
        st.image('data/bg.png')
        st.title("Japanice")
        st.subheader("Accurate dectection of Japanese Character")

st.write("""
         Japanice
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
        
if file is None:
    st.text("Please upload an image file")

else:
    st.image(
            file,
            caption='Uploaded Image.',
            use_container_width=True
        )

    predictions = get_result(image_url=file)
    st.image(predictions)
    # st.balloons()
    # st.sidebar.success(strings)
    
# run: streamlit run stream.py
