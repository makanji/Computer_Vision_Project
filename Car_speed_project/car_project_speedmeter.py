##
# Library Importation for the project at hand
import cv2
import os
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker
import time
# Model initialization and class list
model = YOLO('yolov8s.pt')
class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
              'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep','cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
              'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
              'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
              'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
              'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
              'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
              'teddy bear', 'hair drier', 'toothbrush']
# creating tracker class
tracker = Tracker()
# Count Recorder and Parameters
down = {}
up = {}
red_line_y = 198
blue_line_y = 268
offset = 6
counter_down = []
counter_up = []
count = 0
text_color = (200, 150, 255)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
yellow_color = (0, 255, 255)
white_color = (255, 255, 255)

# Video Capture and Processing
cap = cv2.VideoCapture('highway.mp4')

# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('The_Car_speed_estimation.mp4', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('speed_est_output.avi', fourcc, 20.0, (1020, 500))
# Create a folder to save frames
if not os.path.exists('detected_frames'):
    os.makedirs('detected_frames')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    frame = cv2.resize(frame, (1020, 500))
    result = model.predict(frame)
    # object detection and tracking
    a = result[0].boxes.data
    a = a.detach().cpu().numpy()
    px = pd.DataFrame(a).astype("float")

    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        c_i = int(row[5])
        cls = class_list[c_i]

        if 'car' in cls:
            list.append([x1, y1, x2, y2])

    bbox_id = tracker.update(list)
    for i in bbox_id:
        x3, y3, x4, y4, id = i
        cx = int( x3 + x4 ) // 2
        cy = int( y3 + y4 ) // 2

        # Creating area within which calculation can be calculated
        cv2.line(frame, (172, 198), (774, 198), red_color, 3)
        cv2.putText(frame, ('Red line'), (172, 198), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.line(frame, (8, 268), (927, 268), blue_color, 2)
        cv2.putText(frame, ('Blue line'), (8, 268), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)

        cv2.putText(frame, ('Cars moving Down - ' + str(len(counter_down))), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    text_color,
                    1, cv2.LINE_AA)
        cv2.putText(frame, ('Car moving Up - ' + str(len(counter_up))), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color,
                    1,
                    cv2.LINE_AA)

        # Speed calculation and Visualization

        #check if a car crosses the red line
        if red_line_y < (cy + offset) and red_line_y >(cy - offset):
            down[id] = time.time() #recored current time of the car
        if id in down: #checking if id in dictionary
            if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
                elapsed_time = time.time() - down[id]  # Calculate the elapsed time since the car crossed the first line
                if counter_down.count(id) == 0: # Check if the car ID is not already recorded
                    counter_down.append(id)  # Record the car ID
                    distance = 10  # Define the distance between the two lines in meters
                    a_speed_ms = distance / elapsed_time  # Calculate the speed in meters per second
                    a_speed_kh = a_speed_ms * 3.6  # Convert speed to kilometers per hour
                    # Visualize the car and its speed
                    #cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw a circle at the center of the car
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box around the car
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255),
                            1)  # Display car ID
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 255), 2)  # Display car speed

                elapsed_time = time.time() - down[id]  # Calculate the elapsed time since the car crossed the first line
                if counter_down.count(id) == 0:  # Check if the car ID is not already recorded
                    counter_down.append(id)  # Record the car ID
                    distance = 10  # Define the distance between the two lines in meters
                    a_speed_ms = distance / elapsed_time  # Calculate the speed in meters per second
                    a_speed_kh = a_speed_ms * 3.6  # Convert speed to kilometers per hour
                    # Visualize the car and its speed
                    #cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw a circle at the center of the car
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box around the car
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255),
                                1)  # Display car ID
                    cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                (0, 255, 255), 2) # Display car speed

        # Check if a car crosses the blue line
        if blue_line_y < (cy + offset) and blue_line_y > (cy - offset):
            up[id] = time.time()  # Record the current time when the car touches the second line
        if id in up:  # Check if the car ID is in the dictionary
            if red_line_y < (cy + offset) and red_line_y > (cy - offset):
                elapsed1_time = time.time() - up[id]  # Calculate the elapsed time since the car crossed the second line
                if counter_up.count(id) == 0:  # Check if the car ID is not already recorded
                    counter_up.append(id)  # Record the car ID
                    distance1 = 10  # Define the distance between the two lines in meters
                    a_speed_ms1 = distance1 / elapsed1_time  # Calculate the speed in meters per second
                    a_speed_kh1 = a_speed_ms1 * 3.6  # Convert speed to kilometers per hour
                    #print('########################################################')
                    #print(a_speed_kh)
                    # Visualize the car and its speed
                    #cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw a circle at the center of the car
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # Draw bounding box around the car
                    cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255),
                            1)  # Display car ID
                    cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                            (0, 255, 255), 2)  # Display car speed

    # Save frame
    frame_filename = f'detected_frames/frame_{count}.jpg'
    cv2.imwrite(frame_filename, frame)
    out.write(frame)
    cv2.imshow('frame_view', frame)  # Display the frame with visualizations

    # Check for key press to exit the loop
    if cv2.waitKey(1) & 0xFF == 27:
        break
# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()

##

