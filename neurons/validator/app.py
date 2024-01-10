import time
from math import ceil
from os import listdir

import pandas as pd
import streamlit as st

directory = r"neurons/validator/images"

st.title("Subnet 25 Manual Validator")
prompt_text = st.title("PROMPT:")
empty_image_text = "AWAITING NEW IMAGE ..."

col1, col2, col3, col4, col5 = st.columns(5)

if "vote_1" not in st.session_state:
    st.session_state.vote_1 = False
if "vote_2" not in st.session_state:
    st.session_state.vote_2 = False
if "vote_3" not in st.session_state:
    st.session_state.vote_3 = False
if "vote_4" not in st.session_state:
    st.session_state.vote_4 = False
if "vote_5" not in st.session_state:
    st.session_state.vote_5 = False
if "vote_6" not in st.session_state:
    st.session_state.vote_6 = False
if "vote_7" not in st.session_state:
    st.session_state.vote_7 = False
if "vote_8" not in st.session_state:
    st.session_state.vote_8 = False
if "vote_9" not in st.session_state:
    st.session_state.vote_9 = False
if "vote_10" not in st.session_state:
    st.session_state.vote_10 = False


def input_callback():
    if st.session_state.vote_1:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("1")
            st.session_state.vote_1 = False
    elif st.session_state.vote_2:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("2")
            st.session_state.vote_2 = False
    elif st.session_state.vote_3:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("3")
            st.session_state.vote_3 = False
    elif st.session_state.vote_4:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("4")
            st.session_state.vote_4 = False
    elif st.session_state.vote_5:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("5")
            st.session_state.vote_5 = False
    elif st.session_state.vote_6:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("6")
            st.session_state.vote_6 = False
    elif st.session_state.vote_7:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("7")
            st.session_state.vote_7 = False
    elif st.session_state.vote_8:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("8")
            st.session_state.vote_8 = False
    elif st.session_state.vote_9:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("9")
            st.session_state.vote_9 = False
    elif st.session_state.vote_10:
        with open("neurons/validator/images/vote.txt", "w") as f:
            f.write("10")
            st.session_state.vote_10 = False




with col1:
    placeholder_1 = st.empty()
    vote_1 = st.checkbox("UID 0", key="vote_1", on_change=input_callback)
    placeholder_2 = st.empty()
    vote_2 = st.checkbox("UID 1", key="vote_2", on_change=input_callback)
with col2:
    placeholder_3 = st.empty()
    vote_3 = st.checkbox("UID 2", key="vote_3", on_change=input_callback)
    placeholder_4 = st.empty()
    vote_4 = st.checkbox("UID 3", key="vote_4", on_change=input_callback)
with col3:
    placeholder_5 = st.empty()
    vote_5 = st.checkbox("UID 4", key="vote_5", on_change=input_callback)
    placeholder_6 = st.empty()
    vote_6 = st.checkbox("UID 5", key="vote_6", on_change=input_callback)
with col4:
    placeholder_7 = st.empty()
    vote_7 = st.checkbox("UID 6", key="vote_7", on_change=input_callback)
    placeholder_8 = st.empty()
    vote_8 = st.checkbox("UID 7", key="vote_8", on_change=input_callback)
with col5:
    placeholder_9 = st.empty()
    vote_9 = st.checkbox("UID 8", key="vote_9", on_change=input_callback)
    placeholder_10 = st.empty()
    vote_10 = st.checkbox("UID 9", key="vote_10", on_change=input_callback)

image_list = [
    placeholder_1,
    placeholder_2,
    placeholder_3,
    placeholder_4,
    placeholder_5,
    placeholder_6,
    placeholder_7,
    placeholder_8,
    placeholder_9,
    placeholder_10,
]
while True:
    images = [
        image
        for image in listdir(directory)
        if (".png" in image) and (image != "black.png")
    ]

    if images:
        try:
            prompt = open(f"{directory}/prompt.txt", "r").read()
            prompt_text.text("Prompt: " + f"{prompt}")
            for i in range(0, 10):
                if len(images) > i:
                    image_list[i].image(
                        f"{directory}/{images[i]}", caption=f"UID {i}", width=150
                    )
                else:
                    image_list[i].image(
                        f"{directory}/black.png", caption=f"UID {i}", width=150
                    )
        except:
            for i in range(0, 10):
                image_list[i].text(empty_image_text)

    else:
        for i in range(0, 10):
            prompt_text.text(empty_image_text)
            image_list[i].text("")
