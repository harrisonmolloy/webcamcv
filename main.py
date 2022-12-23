import cv2
import numpy as np

# load cascade
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)
while video.isOpened():
    ret, frame = video.read()
    if ret :

        #resize frames
        scale_percent = 50 # percent of original size
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        resized = cv2.resize(frame, (width, height))

        # convert color
        grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        eql = cv2.equalizeHist(grey)

        # Detect bodies
        bodies = body_cascade.detectMultiScale(eql, 1.1, 4)
        # Draw rectangle around the bodies
        for (x, y, w, h) in bodies:
            cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 0, 0), 1)

        # Detect faces
        faces = face_cascade.detectMultiScale(eql, 1.1, 4)
        # Draw rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 0, 0), 1)

        # show frame
        cv2.imshow('webcam', resized)

    if cv2.waitKey(1) & 0xFF == ord('q') :
        break

video.release()
cv2.destroyAllWindows()
exit()

def resize_frame(frm, per):
    scale_percent = per # percent of original size
    width = int(frm.shape[1] * scale_percent / 100)
    height = int(frm.shape[0] * scale_percent / 100)
    return cv2.resize(frm, (width, height))
