import cv2
import numpy as np


#TRAINED XML FILES FOR FACE AND EYE DETECTION

face_cascade= cv2.CascadeClassifier("C:/Users/dell/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/cv2/data/haarcascade_frontalface_default.XML")
eye_cascade = cv2.CascadeClassifier("C:/Users/dell/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/cv2/data/haarcascade_eye.XML")
smile_cascade = cv2.CascadeClassifier("C:/Users/dell/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/cv2/data/haarcascade_smile.XML")
video_cap= cv2.VideoCapture(0)
while True:
    ret,frame = video_cap.read()
    #img = cv2.imread("C:/Users/dell/OneDrive/Pictures/Camera Roll/papa3.jpg")
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,24,24),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_color = frame[y:y+h,x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        smiles = smile_cascade.detectMultiScale(roi_gray)
        for (sx,sy,sw,sh) in smiles:
            cv2.rectangle(roi_color , (sx,sy),(sx+sw,sy+sh),(100,100,100),1)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ew+ex,ey+eh),(34,34,255),2)
    cv2.imshow('appu',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
video_cap.release()
cv2.waitKey()
cv2.destroyAllWindows()


