import cv2
from imutils import face_utils
import dlib
import numpy as np
import pickle

face_detector = dlib.get_frontal_face_detector()
landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

##camera = cv2.VideoCapture(0)


##while (True):
##
##    ret,img = camera.read()
##    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##
##    rect = face_detector(gray)
##
##    try:
##
##        x1 = rect[0].left()
##        y1 = rect[0].top()
##        x2 = rect[0].right()
##        y2 = rect[0].bottom()
##
##        points = landmark_detector(gray,rect[0])
##        points = face_utils.shape_to_np(points)
##
##        for (x,y) in points:
##            cv2.circle(img,(x,y),1,(0,255,0),1)
##        
##        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2)
##        cv2.putText
##    
##        cv2.imshow('live',img)
##        cv2.waitKey(1)
##
##    except Exception as e:
##
##        print(e)

####img = cv2.imread('face3.jpg')
####gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
####
####rect = face_detector(gray)
####
####x1 = rect[0].left()
####y1 = rect[0].top()
####x2 = rect[0].right()
####y2 = rect[0].bottom()
####
####cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,225),2)
####points = landmark_detector(gray,rect[0])
####points=face_utils.shape_to_np(points)   #convert points to numpy array

##k=0
        
##for (x,y) in points:
##
####    cv2.circle(img,(x,y),1,(0,225,225),1)
##    cv2.putText(img,str(k),(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.3,(0,225,225),1)
##    k+=1


train_data=[[0 for i in range(28)]for j in range(6)]
train_target=[0 for x in range(6)]

names=['Diamond','Oblong','Oval','Round','Square','Triangle']

count=0

def append_data(points,names):

    def euclidian(x1,y1,x2,y2):
        z=0
        z = np.sqrt((np.square(x1-x2))+(np.square(y1-y2)))
        return z


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
    global train_data
    global train_target

    train_data[count][0]=round(point14[0],4)
    train_data[count][1]=round(point14[1],4)
    train_data[count][2]=round(point14[2],4)
    train_data[count][3]=round(point14[3],4)
    train_data[count][4]=round(point14[4],4)
    train_data[count][5]=round(point14[5],4)
    train_data[count][6]=round(point14[6],4)
    train_data[count][7]=round(point14[7],4)
    train_data[count][8]=round(point14[8],4)
    train_data[count][9]=round(point14[9],4)
    train_data[count][10]=round(point14[10],4)
    train_data[count][11]=round(point14[11],4)
    train_data[count][12]=round(point14[12],4)
    train_data[count][13]=round(point14[13],4)
    train_data[count][14]=round(point14[14],4)
    train_data[count][15]=round(point14[15],4)
    train_data[count][16]=round(point14[16],4)
    train_data[count][17]=round(point14[17],4)
    train_data[count][18]=round(point14[18],4)
    train_data[count][19]=round(point14[19],4)
    train_data[count][20]=round(point14[20],4)
    train_data[count][21]=round(point14[21],4)
    train_data[count][22]=round(point14[22],4)
    train_data[count][23]=round(point14[23],4)
    train_data[count][24]=round(point14[24],4)
    train_data[count][25]=round(point14[25],4)
    train_data[count][26]=round(point14[26],4)
    train_data[count][27]=round(point14[27],4)

    train_target[count]=names

    count=count+1

num = 0
for name in names:
        
    img=cv2.imread('Name/'+str(num+1)+'.jpeg')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        

    rect=face_detector(gray)

    points=landmark_detector(gray,rect[0])

    points=face_utils.shape_to_np(points)

    append_data(points,name)
    num+=1

train_data = np.array(train_data)
train_target = np.array(train_target)

train_data_file = open('train_data.pickle','wb')
train_target_file = open('train_target.pickle','wb')

pickle.dump(train_data,train_data_file)
pickle.dump(train_target,train_target_file)

train_data_file.close()
train_target_file.close()
