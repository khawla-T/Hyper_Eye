
import streamlit as st
import pandas as pd
#import os.path
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import json
import poly
import var

def app():
    s=0
    pol=[]
    li=[]
    # Specify canvas parameters in application
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:",
        ("freedraw", "line","rect"),
    )
    #stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 3)

    stroke_color = st.sidebar.color_picker("Stroke color hex: ")
    #bg_color = st.sidebar.color_picker("Background color hex: ", "#eee")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    #realtime_update =st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(640, 640, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=1,
        stroke_color=stroke_color,
        #background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=True,#realtime_update,
        height=640,
        width=640,
        drawing_mode=drawing_mode,
        #point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        #display_toolbar=st.sidebar.checkbox("Display toolbar", True),
        #key="full_app",
    )

    # Do something interesting with the image data and paths
    if canvas_result.image_data is not None:
        st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"])
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
            if col=='type':
                r= objects['type'].item()
            if r  == 'line':
                x1=objects['x1'].item()
                x2=objects['x2'].item()
                y1=objects['y1'].item()
                y2=objects['y1'].item()
                
            elif col== 'path':
                s=objects['path']
                var.poly= poly.Map_to_poly(s)
                print(var.poly)
            elif r == 'rect' : # left  top  width  height
                var.x=objects['left'].item()
                var.y=objects['top'].item()
                var.high_=objects['height'].item()
                var.weid_=objects['width'].item()
                
        st.dataframe(objects)







    