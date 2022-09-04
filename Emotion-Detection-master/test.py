'''
PyPower Projects
Emotion Detection Using AI
'''

#USAGE : python test.py

from keras.models import load_model
from time import sleep
from tensorflow.keras.utils import img_to_array 
from statistics import mode
import webbrowser
from keras.preprocessing import image
import cv2
import numpy as np
import random


face_classifier = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
classifier =load_model('./Emotion_Detection.h5')

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
list = []
counter = 0


while True:
    # Grab a single frame of video
    counter += 1
    if counter > 100 :
        break
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

        # make a prediction on the ROI, then lookup the class

            preds = classifier.predict(roi)[0]
            print("\nprediction = ",preds)
            label=class_labels[preds.argmax()]
            print("\nprediction max = ",preds.argmax())
            print("\nlabel = ",label)
            list.append(label)
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        else:
            cv2.putText(frame,'No Face Found',(20,60),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)
        print("\n\n")
    cv2.imshow('Emotion Detector',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(list)
varRes = mode(list)
neutral = ['https://www.youtube.com/watch?v=5mFTXbZzOAE', 'https://www.youtube.com/watch?v=k5VUKozfNsc', 'https://www.youtube.com/watch?v=_ae2j9jZY_U', 'https://www.youtube.com/watch?v=xCatIOFua2E'];

happy = ['https://www.youtube.com/watch?v=AYcxiROIktI&ab_channel=T-Series', 'https://www.youtube.com/watch?v=ZbZSe6N_BXs&ab_channel=PharrellWilliamsVEVO', 'https://www.youtube.com/watch?v=ipii7KbbJLY&ab_channel=PopularMusic', 'https://www.youtube.com/watch?v=iPUmE-tne5U&ab_channel=KatrinaTheWavesVEVO',
         'https://www.youtube.com/watch?v=ru0K8uYEZWw&ab_channel=justintimberlakeVEVO', 'https://www.youtube.com/watch?v=09R8_2nJtjg&ab_channel=Maroon5VEVO',
         'https://www.youtube.com/watch?v=d-diB65scQU&ab_channel=BobbyMcFerrinVEVO', 'https://www.youtube.com/watch?v=6POZlJAZsok&ab_channel=GroverWashington%2CJr.-Topic', 'https://www.youtube.com/watch?v=HNBCVM4KbUM&ab_channel=BobMarleyVEVO'];

sad = ['https://www.youtube.com/watch?v=HiXx5JFRxb4', 'https://www.youtube.com/watch?v=cyEvAHP8_60', 'https://www.youtube.com/watch?v=YwgNpObouB0'
       'https://www.youtube.com/watch?v=xrcMgO2fgpA', 'https://www.youtube.com/watch?v=JNKjudIKkLg', 'https://www.youtube.com/watch?v=Z6L4u2i97Rw'
       'https://www.youtube.com/watch?v=s-bZD3O3P80', 'https://www.youtube.com/watch?v=-aQMjByEeo8', 'https://youtu.be/DQ4r7HegRQw'];

surprise = ['https://www.youtube.com/watch?v=ZbZSe6N_BXs&ab_channel=PharrellWilliamsVEVO', 'https://www.youtube.com/watch?v=21LGv8Cf0us&ab_channel=SCEntertainment', 'https://www.youtube.com/watch?v=ipii7KbbJLY&ab_channel=PopularMusic', 'https://www.youtube.com/watch?v=qK5KhQG06xU&ab_channel=Audioandlyrics',
            'https://www.youtube.com/watch?v=eYSbUOoq4Vg&ab_channel=pluisje666', 'https://www.youtube.com/watch?v=1We3b8V45Rg&ab_channel=Uknow', 'https://www.youtube.com/watch?v=ApXoWvfEYVU&ab_channel=PostMaloneVEVO', 'https://www.youtube.com/watch?v=mRD0-GxqHVo&ab_channel=GlassAnimalsVEVO', 'https://www.youtube.com/watch?v=jJPMnTXl63E&ab_channel=PowfuVEVO',
            'https://www.youtube.com/watch?v=kTJczUoc26U&ab_channel=TheKidLAROIVEVO'];

angry = ['https://www.youtube.com/watch?v=ETNRfcNIl2w&ab_channel=SofieSo', 'https://www.youtube.com/watch?v=mQvteoFiMlg&ab_channel=EminemVEVO', 'https://www.youtube.com/watch?v=S9bCLPwzSC0&ab_channel=EminemVEVO', 'https://www.youtube.com/watch?v=_Yhyp-_hX2s&ab_channel=msvogue23',
         'https://www.youtube.com/watch?v=WNeLUngb-Xg&ab_channel=TrapMusicHDTV', 'https://www.youtube.com/watch?v=WA4iX5D9Z64&ab_channel=TaylorSwiftVEVO', 'https://www.youtube.com/watch?v=xuhl6Ji5zHM&ab_channel=KanyeWestVEVO', 'https://www.youtube.com/watch?v=ndCI8DIM86w&list=RDGMEMHDXYb1_DDSgDsobPsOFxpA&start_radio=1&rv=xuhl6Ji5zHM&ab_channel=XXXTENTACION-Topic',
         'https://www.youtube.com/watch?v=P9L_ZWVPX4g&list=RDGMEMHDXYb1_DDSgDsobPsOFxpA&index=4&ab_channel=TrevorDaniel-Topic'];

print("Your expression as predicted is :- " + varRes);

if varRes == 'Neutral' :
    webbrowser.open(random.choice(neutral))
if varRes == 'Happy' :
    webbrowser.open(random.choice(happy))
if varRes == 'Sad' :
    webbrowser.open(random.choice(sad))
if varRes == 'Surprise' :
    webbrowser.open(random.choice(surprise))
if varRes == 'Angry' :
    webbrowser.open(random.choice(angry))

cap.release()
cv2.destroyAllWindows()