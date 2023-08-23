import cv2 as cv
import numpy as np

haar_cascade = cv.CascadeClassifier(r'D:\Nitya\AImL_Projects\Face_recognition_Python\haar_face.xml')

people =['Ben Afflek','Elton John','Jerry Seinfield','Madonna','Mindy Kaling']

#features = np.load(r'D:\Nitya\WebDev_projects\features.npy')
#labels = np.load(r'D:\Nitya\WebDev_projects\labels.npy')

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(r'D:\Nitya\AImL_Projects\Face_recognition_Python\face_trained.yml')

img = cv.imread(r'D:\Nitya\AImL_Projects\Face_recognition_Python\Faces\val\elton_john\3.jpg')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Person',gray)

# Detect face in image

faces_rect = haar_cascade.detectMultiScale( gray, scaleFactor=1.1,minNeighbors=4)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+w]

    label, confidence = face_recognizer.predict(faces_roi)
    print(f'\nLabel = {people[label]} with a confidence of {confidence}\n')

    cv.putText(img, str(people[label]), (25,25), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Detected Face',img)

cv.waitKey(0)

