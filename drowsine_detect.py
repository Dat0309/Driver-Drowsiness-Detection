import cv2 as cv
import os
from tensorflow.keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Import file sound
mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

label = ['Closed','Open']

model = load_model('eye_detection.h5')
path = os.getcwd()
cap = cv.VideoCapture(0)
font = cv.FONT_HERSHEY_COMPLEX
count = 0
score = 0
thick = 2
rightPred=[99]
leftPred=[99]

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    '''
    Để phát hiện khuôn mặt trong hình ảnh, trước tiên chúng ta cần chuyển đổi hình ảnh thành thang màu xám (gray) vì thang màu xám không phức tạp như RGB,
    nên OpenCV có thể dễ dàng thao tác trên hình ảnh màu xám.

    sử dụng phân loại tầng Haar để phát hiện khuôn mặt. detectMultiScale() trả về một mảng phát hiện với tọa độ x,y và chiều cao ư, rộng h của khuôn mặt
    Ta sẽ dùng các tọa độ đó để vẽ hình chữ nhật xác định khuôn mặt.

    Tương tự với mắt trái và mắt phải.
    '''
    faces = face.detectMultiScale(gray, minNeighbors = 5, scaleFactor= 1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw rectangle in face
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x +w, y +h), (100, 100, 100), 1)

    cv.rectangle(frame, (0, height - 50), (250, height) , (255, 255, 255), thickness=cv.FILLED)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y+h, x:x+w]
        count = count + 1
        r_eye = cv.cvtColor(r_eye, cv.COLOR_BGR2GRAY)
        r_eye = cv.resize(r_eye, (24, 24))
        r_eye = r_eye/255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rightPred = model.predict_classes(r_eye)

        if(rightPred[0] == 1):
            label='Open'
        if(rightPred[0] == 0):
            label = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        left_eye = frame[y:y+h, x:x+w]
        count = count + 1
        left_eye = cv.cvtColor(left_eye, cv.COLOR_BGR2GRAY)
        left_eye = cv.resize(left_eye, (24, 24))
        left_eye = left_eye/255
        left_eye = left_eye.reshape(24, 24, -1)
        left_eye = np.expand_dims(left_eye, axis=0)
        leftPred = model.predict_classes(left_eye)

        if(leftPred[0] == 1):
            label='Open'
        if(leftPred[0] == 0):
            label = 'Closed'
        break

    if(rightPred[0] == 0 and leftPred[0] == 0):
        score +=1
        cv.putText(frame, 'Closed',(10, height -20), font, 1,(255,0,0), 1, cv.LINE_AA)
    else:
        score-=1
        cv.putText(frame, 'Open',(10, height -20), font, 1,(0,255,0), 1, cv.LINE_AA)

    if(score < 0):
        score = 0

    cv.putText(frame, 'Score:' + str(score), (110, height - 20), font, 1,cv.LINE_AA )

    if(score>15):
        # Person is Sleep, and maybe has died. So, we BEEP the alarm!
        cv.imwrite('image.jpg', frame)
        try:
            sound.play()

        except:
            pass
        if(thick < 16):
            thick +=2
        else:
            thick -=2
            if(thick<2):
                thick = 2
        cv.rectangle(frame, (0,0), (width, height), (0, 0, 255), thick)
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()