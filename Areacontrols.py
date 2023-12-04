import numpy as np
import supervision as sv
import streamlit as st
import tempfile

from ultralytics import YOLO
import  cv2
import var
from streamlit_drawable_canvas import st_canvas
import poly
from PIL import Image
import pandas as pd

#only detect the square 

def app():
    trackers = []
    c=0
    bg_video = st.file_uploader("Upload video:", type=["mp4", "jpg"])
    if bg_video is not None:
        #tfile = tempfile.NamedTemporaryFile(delete=False) 
        vid =bg_video.name
        #tfile.write(bg_video.read())
        with open(vid, mode='wb') as f:
            f.write(bg_video.read()) # save video to disk
        cap = cv2.VideoCapture(vid)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('areacon.avi',fourcc, 5, (640,640))

        while True:
            ret, frame = cap.read()
            if ret == True:
                b = cv2.resize(frame,(640,640),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                if c==0:
                    cv2.imwrite('area.jpg',frame)
                    c+=1
                out.write(b)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # Taking image form the video to take the ROI
        generator = sv.get_video_frames_generator('areacon.avi')  
        
        #////////////Start of ROI//////////////
        # Specify canvas parameters in application
        
        start_button_pressed = st.button("Start The ROI")
        drawing_mode = "freedraw"
        stroke_color = st.color_picker("Stroke color hex: ")
        img='area.jpg'
        # Create a canvas component
        canvas_result = st_canvas(
            fill_color="rgba(640, 640, 0, 0.3)",  # Fixed fill color with some opacity
            stroke_width=1,
            stroke_color=stroke_color,
            #background_color=bg_color,
            background_image=Image.open(img) if img else None,
            update_streamlit=True,#realtime_update,
            height=640,
            width=640,
            drawing_mode=drawing_mode, )
    
        if start_button_pressed is not None:
            stop_button_pressed = st.button("End The ROI")
            while not stop_button_pressed:
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
                            
                        elif r == 'rect' : # left  top  width  height
                            var.x=objects['left'].item()
                            var.y=objects['top'].item()
                            var.high_=objects['height'].item()
                            var.weid_=objects['width'].item()
                    
        #///////////end of ROI//////////////
            
                model = YOLO('yolov8s.pt')
                #if var.poly is not None:
                if len(var.poly) >0:
                    colors = sv.ColorPalette.default()
                    polygons = [
                            np.array(var.poly
                            , np.int32)
                            
                        ]
                    s= len(polygons)
                    video_info = sv.VideoInfo.from_video_path('areacon.avi')
                    zones = [
                        sv.PolygonZone(
                            polygon=polygon, 
                            frame_resolution_wh=video_info.resolution_wh
                        )
                        for polygon
                        in polygons
                    ]
                    zone_annotators = [
                            sv.PolygonZoneAnnotator(
                                zone=zone, 
                                color=colors.by_idx(index), 
                                thickness=4,
                                text_thickness=8,
                                text_scale=4
                            )
                            for index, zone
                            in enumerate(zones)
                        ]
                    box_annotators = [
                    
                        sv.BoxAnnotator(
                            color=colors.by_idx(index), 
                            thickness=4, 
                            text_thickness=4, 
                            text_scale=2
                            )
                        for index
                        in range(len(polygons))
                    ]
                
                    # extract video frame
                    generator = sv.get_video_frames_generator('areacon.avi')  

                    #####Define the first /previous frame of ids and zones
                    # dic ({'zone1':[ids]})    
                    def process_frame(frame: np.ndarray, _) -> np.ndarray:
                            iterator = iter(generator)
                            frame = next(iterator)
                            
                            # detect
                            results = model.predict(frame, imgsz=(640,640))[0]
                            detections = sv.Detections.from_yolov8(results)
                        
                            detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
                    
                            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                                mask = zone.trigger(detections=detections)#all detected pepole within the zone
                                detections_filtered = detections[mask] 
                                
                                if detections_filtered is not None:
                                    st.toast('Someone is geting in!!!')
                                    return frame
                                
                                frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
                                frame = zone_annotator.annotate(scene=frame)
                                
                                
                            return frame

                    
                        #sv.show_frame_in_notebook(frame, (16, 16))
                    sv.process_video(source_path='areacon.avi', target_path=f"/Users/khawlahd/Desktop/cv/my-venv/area-result.mp4", callback=process_frame)
                    


                
                

                