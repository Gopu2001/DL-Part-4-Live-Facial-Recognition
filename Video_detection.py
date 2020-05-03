import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import cv2
import numpy as np
import face_recognition as fr
from PIL import Image
sys.stderr = stderr

database = "../Database/"
known = []
for img in os.listdir(database):
    known.append(fr.face_encodings(fr.load_image_file(database+img))[0])

cv2.namedWindow("Camera")
vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

frame_num = 0
frame_pause_num = -1
try:
    while rval:
        for x,y,w,h in cv2.CascadeClassifier("haarcascade_frontalface_default.xml").detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5):
            rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            try:
                results = []
                results = fr.compare_faces(known, fr.face_encodings(frame[y:y+h,x:x+w])[0], tolerance=0.5)
            except IndexError:
                continue
            cv2.rectangle(frame, (x,y+h),(x+w,y+h+20),(255,0,0),cv2.FILLED)
            cv2.putText(rect, os.listdir(database)[results.index(True)].split(".")[0] if True in results else "Unknown", (x+5,y+h+16), 0, 0.7, (0,0,0))
            if True not in results:
                if frame_pause_num == -1:
                    frame_pause_num = frame_num
                if frame_num == frame_pause_num+2:
                    name = input("Who is this person's name? ")
                    Image.fromarray(frame[y:y+h,x:x+w][:,:,::-1]).save(database+name+".png")
                    known.append(fr.face_encodings(fr.load_image_file(database+name+".png"))[0])
                    frame_pause_num = -1
            if frame_num > frame_pause_num+2 or True in results:
                frame_pause_num = -1
        cv2.imshow("Camera", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:
            vc.release()
            cv2.destroyWindow("Camera")
            break
        frame_num+=1
except KeyboardInterrupt:
    vc.release()
    cv2.destroyWindow("Camera")
    sys.exit(0)
