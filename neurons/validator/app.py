import streamlit as st
from os import listdir
from math import ceil
import pandas as pd
import time

padding_top = 0
padding_bottom = 1  
padding_left = 1
padding_right = 1
# max_width_str = f'max-width: 100%;'
# st.markdown(f'''
#             <style>
#                 .reportview-container .sidebar-content {{
#                     padding-top: {padding_top}rem;
#                 }}
#                 .reportview-container .main .block-container {{
#                     padding-top: {padding_top}rem;
#                     padding-right: {padding_right}rem;
#                     padding-left: {padding_left}rem;
#                     padding-bottom: {padding_bottom}rem;
#                 }}
#                 .appview-container .main .block-container {{
#                     padding-top: {padding_top}rem;
#                     padding-right: {padding_right}rem;
#                     padding-left: {padding_left}rem;
#                     padding-bottom: {padding_bottom}rem;
                
#                 }}
#                 :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
#                 :gray[pretty] :rainbow[colors].
#             </style>
#             ''', unsafe_allow_html=True,
# )


# .reportview-container .sidebar-content {{
#     padding-top: {padding_top}rem;
# }}

# st.markdown('''
#     <style>
#     .reportview-container .sidebar-content {{
#         padding-top: {padding_top}rem;
#         padding-right: {padding_right}rem;
#         padding-left: {padding_left}rem;
#         padding-bottom: {padding_bottom}rem;
#     }}
#     </style>
#     :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
#     :gray[pretty] :rainbow[colors].
#     ''', unsafe_allow_html=True)

# directory = r'images\bike'
directory = r"neurons/validator/images"

st.title("Subnet 25 Manual Validator")

# .markdown('''<style>
#     .reportview-container .sidebar-content {{
#         padding-top: {padding_top}rem;
#         padding-right: {padding_right}rem;
#         padding-left: {padding_left}rem;
#         padding-bottom: {padding_bottom}rem;
#     }}
#     </style>''')
Title = st.empty()

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
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("1")
            st.session_state.vote_1 = False
    elif st.session_state.vote_2:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("2")
            st.session_state.vote_2 = False
    elif st.session_state.vote_3:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("3")
            st.session_state.vote_3 = False
    elif st.session_state.vote_4:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("4")
            st.session_state.vote_4 = False
    elif st.session_state.vote_5:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("5")
            st.session_state.vote_5 = False
    elif st.session_state.vote_6:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("6")
            st.session_state.vote_6 = False
    elif st.session_state.vote_7:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("7")
            st.session_state.vote_7 = False
    elif st.session_state.vote_8:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("8")
            st.session_state.vote_8 = False
    elif st.session_state.vote_9:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("9")
            st.session_state.vote_9 = False
    elif st.session_state.vote_10:
        with open('neurons/validator/images/vote.txt', 'w') as f:
            f.write("10")
            st.session_state.vote_10 = False

with col1:
    placeholder_1 = st.empty()
    vote_1 = st.checkbox("UID 0", key ='vote_1', on_change=input_callback)
    placeholder_2 = st.empty()
    vote_2 = st.checkbox("UID 1", key ='vote_2', on_change=input_callback)
with col2:
    placeholder_3 = st.empty()
    vote_3 = st.checkbox("UID 2")
    placeholder_4 = st.empty()
    vote_4 = st.checkbox("UID 3")
with col3:
    placeholder_5 = st.empty()
    vote_5 = st.checkbox("UID 4")
    placeholder_6 = st.empty()
    vote_6 = st.checkbox("UID 5")
with col4:
    placeholder_7 = st.empty()
    vote_7 = st.checkbox("UID 6")
    placeholder_8 = st.empty()
    vote_8 = st.checkbox("UID 7")
with col5:
    placeholder_9 = st.empty()
    vote_9 = st.checkbox("UID 8")
    placeholder_10 = st.empty()
    vote_10 = st.checkbox("UID 9")

image_list = [placeholder_1, placeholder_2, placeholder_3, placeholder_4, placeholder_5, placeholder_6, placeholder_7, placeholder_8, placeholder_9, placeholder_10]
while True:
    images = [image for image in listdir(directory) if ('.png' in image) and (image != 'black.png')]

    if images:
        try:
            # print("images")
            # print(images)
            for i in range(0, 10):  
                if len(images) > i:
                    image_list[i].image(f'{directory}/{images[i]}', caption=f'UID {i}', width=150)
                else: 
                    image_list[i].image(f'{directory}/black.png', caption=f'UID {i}', width=150)
        except:
            for i in range(0, 10):  
                image_list[i].text("AWAITING NEW ...")
 
    else:      
      
        for i in range(0, 10):  
            image_list[i].text("AWAITING NEW ...")

        # vote_1.value = False

# files = [image for image in listdir(directory) if '.png' in image]

# def initialize():    
#     df = pd.DataFrame({'file':files,
#                     'incorrect':[False]*len(files),
#                     'label':['']*len(files)})
#     df.set_index('file', inplace=True)
#     return df

# if 'df' not in st.session_state:
#     df = initialize()
#     st.session_state.df = df
# else:
#     df = st.session_state.df 


# controls = st.columns(3)
# with controls[0]:
#     batch_size = st.select_slider("Batch size:",range(10,110,10))
# with controls[1]:
#     row_size = st.select_slider("Row size:", range(1,6), value = 5)
# num_batches = ceil(len(files)/batch_size)
# with controls[2]:
#     page = st.selectbox("Page", range(1,num_batches+1))


# def update (image, col): 
#     df.at[image,col] = st.session_state[f'{col}_{image}']
#     if st.session_state[f'incorrect_{image}'] == False:
#        st.session_state[f'label_{image}'] = ''
#        df.at[image,'label'] = ''

# batch = files[(page-1)*batch_size : page*batch_size]

# grid = st.columns(row_size)
# col = 0
# for image in batch:
#     with grid[col]:
#         st.image(f'{directory}/{image}', caption='bike')
#         st.checkbox("Incorrect", key=f'incorrect_{image}', 
#                     value = df.at[image,'incorrect'], 
#                     on_change=update, args=(image,'incorrect'))
#         if df.at[image,'incorrect']:
#             st.text_input('New label:', key=f'label_{image}', 
#                           value = df.at[image,'label'],
#                           on_change=update, args=(image,'label'))
#         else:
#             st.write('##')
#             st.write('##')
#             st.write('###')
#     col = (col + 1) % row_size

# st.write('## Corrections')
# df[df['incorrect']==True]

# import gradio as gr
# from os import listdir

# from PIL import Image
# directory = r"~/TV/neurons/validator/images"
# files = listdir(directory)
# images = files
# # breakpoint()

# with gr.Blocks(title = "Annotation Portal") as demo:
#     with gr.Row():
#         image_objects = {}
#         # breakpoint()
#         for i, image in enumerate(images[:2]):
#             image_objects[f'image_{str(i)}'] = gr.Image(Image.open(directory + "/" + image), type="pil", label="Bounding boxes of labeled paragraphs", visible=True)
#         # breakpoint()

# # with gr.Blocks() as demo:
# #     gr.Markdown("# Greetings from Gradio!")
# #     inp = gr.Textbox(placeholder="What is your name?")
# #     out = gr.Textbox()

# #     inp.change(fn=lambda x: f"Welcome, {x}!",
# #                inputs=inp,
# #                outputs=out)

# if __name__ == "__main__":
#     demo.launch()