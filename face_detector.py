from cv2 import cv2
from random import randrange

#load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data=cv2.CascadeClassifier(r'C:\Users\Nikita Sara Mathew\Desktop\haarcascade_frontalface_default.xml')

#choose an image to detect faces from
#img = cv2.imread(r'C:\Users\Nikita Sara Mathew\Desktop\people.jpg')
#img = cv2.imread(r'C:\Users\Nikita Sara Mathew\Desktop\RDJ.png')
webcam=cv2.VideoCapture(0)


while True:

    successful_frame_read, frame = webcam.read()

    #convert to grayscale
    grayscaled_img=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect faces
    face_coordinates=trained_face_data.detectMultiScale(grayscaled_img)

    #draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    #display the image with the faces
    cv2.imshow("Nikki's Face Detector", frame)
    key=cv2.waitKey(1)

    #break out of image if Q is pressed
    if key==81 or key==113:
        break

   












"""












print("Code Completed")
"""