import cv2
from random import randrange


#load pre trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#choose an image to detect faces in
#img1 = cv2.imread('Face recognition/TomCruise.png') 

#to capture video (single frame) from webcam 
webcam = cv2.VideoCapture(0) ## 0/1 means the default webcam for the system, can also include a video file. 

# iterate forever over the frames from the webcam 
while True:
    
    # read the current frame
    success_frame_read, frame = webcam.read()
    
    #must convert to grayscale
    img1_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces of any scale and returns the rectangle coordinates for the face 
    faces_coordinates = trained_face_data.detectMultiScale(img1_grayscale)
    
    # #loops through the coordinates of each face found in faces_coordinates and draws a rectangle on them for detection
    for x, y, w, h in faces_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 3) ## draws the rectangle on top of the image
    
    #shows image on a new window with the text as window name 
    cv2.imshow("Clever Programmer For Face Detector", frame)
    key = cv2.waitKey(1) ## wait for a key press before the window closes otherwise closes instantly

    #print(success_frame_read)
    
    # stop if Q key is pressed
    if key == 81 or key == 113:
        break  
    
# release the video capture object
webcam.release()

''' 
Code for single image recognition
    can give file path to an image and can recognize it
'''

# #must convert to grayscale
# img1_grayscale = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# # Detect faces of any scale and returns the rectangle coordinates for the face 
# faces_coordinates = trained_face_data.detectMultiScale(img1_grayscale)
# #print(faces_coordinates) ## prints: [[779 234 602 602]]


# #draw each rectangle individually instead of using a for loop
# # (x, y, w, h) = faces_coordinates[2]
# # cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3) ## draws the rectangle on top of the image

# # (x, y, w, h) = faces_coordinates[1]
# # cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3) ## draws the rectangle on top of the image

# # (x, y, w, h) = faces_coordinates[0]
# # cv2.rectangle(img1, (x, y), (x+w, y+h), (0, 255, 0), 3) ## draws the rectangle on top of the image

# #loops through the coordinates of each face found in faces_coordinates and draws a rectangle on them for detection
# for x, y, w, h in faces_coordinates:
#     cv2.rectangle(img1, (x, y), (x+w, y+h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 3) ## draws the rectangle on top of the image

# #shows image on a new window with the text as window name 
# cv2.imshow("Clever Programmer For Face Detector", img1)
# cv2.waitKey() ## wait for a key press before the window closes otherwise closes instantly


print("Code Completed")

