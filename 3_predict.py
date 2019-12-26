import cv2
from imutils import face_utils
import dlib
import numpy as np
import pickle
from keras.models import load_model

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

test_data=[[0 for i in range(28)]for j in range(1)]
image_shapes=['Diamond','Oblong','Oval','Round','Square','Triangle']
count=0
def append_data(points):

    def euclidian(x1,y1,x2,y2):
        z=0
        z = np.sqrt((np.square(x1-x2))+(np.square(y1-y2)))
        return z

    face_dic={'Diamond':0,'Oblong':1,'Oval':2,'Round':3,'Square':4,'Triangle':5}

    point14=[0 for i in range(28)]

    
    point14[0]=euclidian(points[17][0],points[17][1],points[21][0],points[21][1])
    point14[1]=euclidian(points[21][0],points[21][1],points[39][0],points[39][1])
    point14[2]=euclidian(points[39][0],points[39][1],points[38][0],points[38][1])
    point14[3]=euclidian(points[39][0],points[39][1],points[41][0],points[41][1])
    point14[4]=euclidian(points[38][0],points[38][1],points[36][0],points[36][1])
    point14[5]=euclidian(points[36][0],points[36][1],points[41][0],points[41][1])
    point14[6]=euclidian(points[36][0],points[36][1],points[17][0],points[17][1])
    point14[7]=euclidian(points[41][0],points[41][1],points[48][0],points[48][1])
    point14[8]=euclidian(points[48][0],points[48][1],points[31][0],points[31][1])
    point14[9]=euclidian(points[31][0],points[31][1],points[39][0],points[39][1])
    point14[10]=euclidian(points[39][0],points[39][1],points[33][0],points[33][1])
    point14[11]=euclidian(points[33][0],points[33][1],points[31][0],points[31][1])
    point14[12]=euclidian(points[33][0],points[33][1],points[48][0],points[48][1])
    point14[13]=euclidian(points[48][0],points[48][1],points[57][0],points[57][1])

    point14[14]=euclidian(points[22][0],points[22][1],points[26][0],points[26][1])
    point14[15]=euclidian(points[26][0],points[26][1],points[45][0],points[45][1])
    point14[16]=euclidian(points[45][0],points[45][1],points[43][0],points[43][1])
    point14[17]=euclidian(points[43][0],points[43][1],points[42][0],points[42][1])
    point14[18]=euclidian(points[42][0],points[42][1],points[22][0],points[22][1])
    point14[19]=euclidian(points[45][0],points[45][1],points[46][0],points[46][1])
    point14[20]=euclidian(points[46][0],points[46][1],points[42][0],points[42][1])
    point14[21]=euclidian(points[42][0],points[42][1],points[33][0],points[33][1])
    point14[22]=euclidian(points[42][0],points[42][1],points[35][0],points[35][1])
    point14[23]=euclidian(points[35][0],points[35][1],points[33][0],points[33][1])
    point14[24]=euclidian(points[33][0],points[33][1],points[54][0],points[54][1])
    point14[25]=euclidian(points[54][0],points[54][1],points[35][0],points[35][1])
    point14[26]=euclidian(points[46][0],points[46][1],points[54][0],points[54][1])
    point14[27]=euclidian(points[54][0],points[54][1],points[57][0],points[57][1])

    global count
    global test_data

    for i in range(28):
        test_data[0][i]=round(point14[i],4)

camera = cv2.VideoCapture(0)
j=0
b=True
while (b):

    
    ret,img = camera.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
    rect=face_detector(gray)

    try:

        x1 = rect[0].left()
        y1 = rect[0].top()
        x2 = rect[0].right()
        y2 = rect[0].bottom()

        points = landmark_detector(gray,rect[0])
        points = face_utils.shape_to_np(points)

        for (x,y) in points:
            cv2.circle(img,(x,y),1,(0,255,0),1)
    
        cv2.imshow('live',img)
        cv2.waitKey(10)

        append_data(points)
        test_data = np.array(test_data)

        new_model = load_model('model.h5')

        result = new_model.predict(test_data)

        if(np.max(result)>0.9):
            print(image_shapes[np.argmax(result)])
                    
    except Exception as e:
        print(e)



    

