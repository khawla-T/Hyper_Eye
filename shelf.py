import streamlit as st
from streamlit_option_menu import option_menu
import cv2 as cv
import tempfile
from roboflow import Roboflow
import supervision as sv
import numpy as np
rf = Roboflow(api_key="OdpE3HQH2lrjleseCRms")
project = rf.workspace("monynote").project("stockshelfs")
model = project.version(1).model

def app():
    missed=0
    #download vido 
    bg_video = st.file_uploader("Upload video:", type=["mp4", "jpg"])
    
    if bg_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(bg_video.read())
        stframe = st.empty()
        result1 = model.predict_video(tfile.name)
        stframe.video(result1)
        vf = cv.VideoCapture(tfile.name)

        video_info = sv.VideoInfo.from_video_path(tfile.name)
        box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
        def process_frame(frame: np.ndarray, _) -> np.ndarray:
    # detect
            results = model(frame, imgsz=640)[0]
            detections = sv.Detections.from_yolov8(results)
            #sv.Detections.
            detections = detections[detections.class_id == 0]

    # annotate
            box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)
            labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, confidence, class_id, _ in detections]
            frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)
            return frame

    sv.process_video(source_path=tfile.name, target_path=f"/Users/khawlahd/Desktop/cv/my-venv/shelf-result.mp4", callback=process_frame)
       
        
    st.text_area(label ='The total of missed Items=',value=missed, height =100)