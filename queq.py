import supervision as sv
import streamlit as st

import numpy as np
import  cv2
import time


def app():
    
    cap = cv2.VideoCapture('/Users/khawlahd/Desktop/cv/my-venv/MallMenandWom.mp4') #0
    st.title("Queuing Managemnt ")
    frame_placeholder = st.empty()
    count=0
    print("im here")
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

                
                frame = cv2.cvtColor (frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame, channels="RGB" ) 
                
                    
                if count >3:
                    #shwo women ads
                        st.toast('There is Queue Open a new casher!!!', icon='person-fill-exclamation')
                        x=False
                        break
                count+=1
                if cv2.waitKey (1) & 0b11111111 == ord("q") or stop_button_pressed:
                    break
    cap. release ()
    cv2. destroyAllWindows () 


            