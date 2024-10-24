import cv2
#import numpy as np
import os
import platform

def beep_alarm():
    if platform.system() == "Windows":
        import winsound
        winsound.Beep(1000, 500)  
    elif platform.system() == "Darwin":  
        os.system('afplay /System/Library/Sounds/Glass.aiff')
    else:
        os.system('beep')  
        print('\a')  


cap = cv2.VideoCapture(0)

first_frame = None

# Define the region of interest (ROI) - adjust these values
roi_top_left_x = 200   
roi_top_left_y = 200   
roi_width = 200        
roi_height = 150       

print("Starting camera...")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    cv2.rectangle(frame, (roi_top_left_x, roi_top_left_y), 
                  (roi_top_left_x + roi_width, roi_top_left_y + roi_height), 
                  (255, 0, 0), 2) 
    
    roi = frame[roi_top_left_y:roi_top_left_y + roi_height, 
                roi_top_left_x:roi_top_left_x + roi_width]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if first_frame is None:
        first_frame = gray
        continue

    
    frame_diff = cv2.absdiff(first_frame, gray)

    # Apply a threshold to highlight significant differences
    thresh_frame = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    motion_detected = False  
    for contour in contours:
        if cv2.contourArea(contour) < 1000:  
            continue
        motion_detected = True
        break  

    if motion_detected:
        print("Motion detected!")
        beep_alarm()

    
    first_frame = gray

    cv2.imshow("Camera Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
