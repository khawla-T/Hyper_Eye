import supervision as sv
import streamlit as st
from ultralytics import YOLO
import numpy as np
import  cv2
def app():
    model = YOLO("bestM150.pt")
    cap = cv2.VideoCapture(1) #0
    st.title("Advertisement By Gender")
    frame_placeholder = st.empty()

    start_button_pressed = st.button("start")
    stop_button_pressed = st.button("Stop")
    x= True
    if start_button_pressed is not None:
        cap.isOpened()
        while   not stop_button_pressed and  x:
                ret, frame = cap.read()
                if not ret:
                    st.write("The video capture has")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = model.predict(frame, imgsz=(640,640))[0]
                detections = sv.Detections.from_yolov8(results)
                detections = detections[(detections.confidence > 0.5)]
                frame = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB" ) 
                
                for detection_idx in range(len(detections)):
                    
                    if detections.class_id[detection_idx]==0:
                    #shwo women ads
                        frame_placeholder.video('/Users/khawlahd/Desktop/cv/my-venv/Perfume.mp4')
                        x=False
                        break
                    else:
                        #show men ads /Users/khawlahd/Desktop/cv/my-venv/men.mp4
                        frame_placeholder.video("/Users/khawlahd/Desktop/cv/my-venv/men.mp4")
                        x=False
                        break
                if cv2.waitKey (1) & 0b11111111 == ord("q") or stop_button_pressed:
                    break
    cap. release ()
    cv2. destroyAllWindows () 


            