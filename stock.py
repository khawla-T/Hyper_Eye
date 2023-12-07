
import supervision as sv
import streamlit as st
from ultralytics import YOLO
import numpy as np
import  cv2
from roboflow import Roboflow
import time

def app():
    rf = Roboflow(api_key=)
    project = rf.workspace().project("")
    model = project.version(1).model
   
    cap = cv2.VideoCapture(0) #0
    st.title("Shelves Monitoring")
    frame_placeholder = st.empty()
    #im=cv2.imread('shelf.jpeg')
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
                
                
                
                
                result=model.predict(frame, confidence=40, overlap=30).json()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB" ) 
                
                for i in result['predictions']:
                    for b in i:
                       
                        if i['class_id'] == 1:
                            st.toast('Out 0f stock!!!', icon='🏃')
                            x=False
                            break
    
                if cv2.waitKey (1) & 0b11111111 == ord("q") or stop_button_pressed:
                    break
    cap. release ()
    cv2. destroyAllWindows () 


                   
