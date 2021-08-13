
import cv2
import numpy as np

eye_detector=cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')
face_detector= cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')

img=cv2.imread("1.jpg")
gray=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
faces=face_detector.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0,0),2)
    roi_gry=gray[y:y+h, x:x+w]
    roi=img[y:y+h, x:x+w]
    eyes = eye_detector.detectMultiScale(roi_gry)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)


##testcommit

cv2.imshow("original image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()