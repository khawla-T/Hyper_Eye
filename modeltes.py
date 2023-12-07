import cv2
import inference
import supervision as sv
from roboflow import Roboflow
import streamlit as st
import numpy as np

annotator = sv.BoxAnnotator()

rf = Roboflow(api_key="")
project = rf.workspace("").project("")
model = project.version(1).model






bg_video = st.file_uploader("Upload video:", type=["mp4", "jpg"])
if bg_video is not None:
        #tfile = tempfile.NamedTemporaryFile(delete=False) 
        vid =bg_video.name
        #tfile.write(bg_video.read())
        with open(vid, mode='wb') as f:
            f.write(bg_video.read()) # save video to disk
        cap = cv2.VideoCapture(vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('outofstock.avi',fourcc, 5, (640,640))

        while True:
            ret, frame = cap.read()
            if ret == True:
                b = cv2.resize(frame,(640,640),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                out.write(b)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()


generator = sv.get_video_frames_generator('outofstock.avi')  
def process_frame(frame: np.ndarray, _) -> np.ndarray:
                iterator = iter(generator)
                frame = next(iterator)
                
                # detect
                #results = model.predict(frame ,confidence=40, overlap=30).json()[0]
                results = model.predict(frame )[0]
                results
                detections = sv.Detections.from_yolov8(results)
                sv.Detections.from_
                #detections = sv.Detections.from_yolov8(results)
                #"class": "missing"
                #"class_id": 1
                
                detections = detections[(detections.class_id == 1) & (detections.confidence > 0.5)]
                #detect 
                # if get it notifiy 
                if detections is not None:
                    st.toast('Out 0f stock!!!', icon='🏃')
sv.process_video(source_path='outofstock.avi', target_path=f"/Users/khawlahd/Desktop/cv/my-venv/stockresult.mp4", callback=process_frame)

