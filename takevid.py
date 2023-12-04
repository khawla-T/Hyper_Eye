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
#import yolox


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
        out = cv2.VideoWriter('output.avi',fourcc, 5, (640,640))

        while True:
            ret, frame = cap.read()
            if ret == True:
                b = cv2.resize(frame,(640,640),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
                if c==0:
                    cv2.imwrite('anno.jpg',frame)
                    c+=1
                out.write(b)
            else:
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        # Taking image form the video to take the ROI
        generator = sv.get_video_frames_generator('output.avi')  
        iterator = iter(generator)
        frame = next(iterator)
        #////////////Start of ROI//////////////
        # Specify canvas parameters in application
        
        start_button_pressed = st.button("Start The ROI")
        drawing_mode = "freedraw"
        stroke_color = st.color_picker("Stroke color hex: ")
        img='anno.jpg'
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
                
                if len(var.poly) >0:
                    colors = sv.ColorPalette.default()
                    polygons = [
                            np.array(var.poly
                            , np.int32)
                            
                        ]
                    s= len(polygons)
                    video_info = sv.VideoInfo.from_video_path('output.avi')
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
                #total count for each zones:
                    total_preZone = {key:0 for key in zones}  
                    current_frame={key:[] for key in zones}
                    previous_frame={key:[] for key in zones}
                    # extract video frame
                    generator = sv.get_video_frames_generator('output.avi')  

                    #####Define the first /previous frame of ids and zones
                    # dic ({'zone1':[ids]})    
                    def process_frame(frame: np.ndarray, _) -> np.ndarray:
                            iterator = iter(generator)
                            frame = next(iterator)
                            
                            # detect
                            results = model.track(frame, imgsz=(640,640))[0]
                            detections = sv.Detections.from_yolov8(results)
                            if results.boxes.id is not None:
                                detections.tracker_id= results.boxes.id.cpu().numpy().astype(int)
                            #detections2 = sv.Detections.from_ultralytics(results)#
                            detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
                        
                            #new detect to count
                            if detections.tracker_id is not None:
                                trackers.append(detections.tracker_id)
                            print("total number to track",len(trackers))    
                        
                            for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
                                mask = zone.trigger(detections=detections)#all detected pepole within the zone
                                detections_filtered = detections[mask] 
                                ###my try to count..
                                #print("zone count",zones)   
                                count1=len(detections[mask])#this count
                                print("detected is ",count1)
                            
                                if len(detections[mask]) > 0: #1
                                    #2 assign for every zone the trakerid current
                                    #3 if the previus is empty fill with current else compare current with previus
                                    #           if not-found -> add to the counter & add to prevous
                                    #           else do nothing
                                    
                                    for detection_idx in range(len(detections_filtered)):
                                        tracker_id = int(detections_filtered.tracker_id[detection_idx])
                                        print("tracker_id",tracker_id)
                                        current_frame[zone].append(tracker_id)
                                    print(current_frame[zone])
                                    if previous_frame[zone] == []:
                                        previous_frame[zone]=current_frame[zone]
                                        total_preZone[zone]+=len(detections_filtered)
                                        print("I must appear once!")
                                    #previous_frame=trackers
                                    else:
                                        
                                        for i in  current_frame[zone]:
                                            if i in  previous_frame[zone]: 
                                                continue  
                                            else:
                                                previous_frame[zone].append(i)
                                                # update counter
                                                total_preZone[zone]+=1          
                                #dict1['X'][0]
                                
                                #########
                                frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
                                frame = zone_annotator.annotate(scene=frame)
                                
                                #clear the current dic
                                print("current total: ", total_preZone[zone])
                                current_frame[zone]=[]
                            return frame

                    
                        #sv.show_frame_in_notebook(frame, (16, 16))
                    sv.process_video(source_path='output.avi', target_path=f"/Users/khawlahd/Desktop/cv/my-venv/count-result.mp4", callback=process_frame)
                    total=0
                    # for i , zone in zip(range(len(zones)), zones)
                    for zone in zones:
                        print("zone = ",total_preZone[zone])
                        total+=total_preZone[zone]
                    print("total is : ",total)
                    
                    st.text_area(label ='The total of pepols entred=',value=total, height =100)


            
            

                