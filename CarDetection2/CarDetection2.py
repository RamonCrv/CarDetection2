import cv2
#from google.colab.patches import cv2_imshow
from time import sleep
import numpy as np
import imutils 
import logging  

cars_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
body_cascade = cv2.CascadeClassifier('haarcascade_upperbody.xml')





padding = 10 
hog = cv2.HOGDescriptor() 
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector()) 
   
image = cv2.imread('testeImg.jpg') 
   
image = imutils.resize(image, width=min(800, image.shape[1])) 
   
(regions, _) = hog.detectMultiScale(image, winStride=(16, 16), padding=(4, 4), scale=1.21, useMeanshiftGrouping = 0) 
logging.warning("---------> qtd humanos: %.2f" % len(regions))
#(regions, _) = hog.detectMultiScale(image,  scale=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

for (x, y, w, h) in regions: 
    (startX,startY) = max(0, x-padding), max(0, y-h-padding)
    (endX,endY) = min(image.shape[1]-1, x+int(w * 1.5)+padding), min(image.shape[0]-1, y+h+padding)

    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) 

cv2.imshow("Image", image) 
cv2.waitKey(0) 
   
cv2.destroyAllWindows() 




def chop_cars(frame, imgName):    
    cars = cars_cascade.detectMultiScale(frame, 1.225, 1, minSize=(220,220))
    carsChopped = []  
    imgName2 = 0
    for (x, y, w, h) in cars:
        imgName2 = imgName + 1
        carsChopped = frame[y:y + h, x:x + w]
        
        
        cv2.imwrite(str(imgName)+str(imgName2)+'.jpg', carsChopped)
    
    

def detect_cars(frame):
    cars = cars_cascade.detectMultiScale(frame, 1.225, 1, minSize=(220,220))
    
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame
    


def detect_bodys(frame):
    bodys = body_cascade.detectMultiScale(frame, 1.05, 4)
    #bodys = body_cascade.detectMultiScale(frame, scaleFactor= 1.05, minNeighbors= 5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in bodys:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame    

def detect_bodys2(frame):
    #(bodys,_) = hog.detectMultiScale(frame, winStride=(8,8), padding=(16, 16), scale=1.05, useMeanshiftGrouping = -1) 
    (bodys,_) = hog.detectMultiScale(frame, winStride=(1,20), padding=(20, 12), scale=1.2225)  
    #(bodys,_) = hog.detectMultiScale(frame, winStride=(4,4), padding=(4, 4), scale=1.05) 
    #(bodys,_) = hog.detectMultiScale(frame, winStride=(5,5), padding=(3, 3), scale=1.21)   
    for (x, y, w, h) in bodys:
        #cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
         cv2.rectangle(frame, (x, y),  
                  (x + w, y + h),  
                  (0, 0, 255), 2) 
    return frame 

   



def SimulatorVideo():
    CarVideo = cv2.VideoCapture('cars2.mp4')
    imgName = 0;
    
    while CarVideo.isOpened():
        tempo = float(1/15)
        sleep(tempo)
        ret, frame = CarVideo.read()
        controlkey = cv2.waitKey(1)
        if ret:
            
            imgName = imgName + 1
            chop_cars(frame, imgName)
            cars_frame = detect_cars(frame)
            cv2.imshow('frame', cars_frame)
            
           
            
            
        else:
            break
        if controlkey == ord('q'):
            break

def SimulatorImagem():
    bodyImage = cv2.imread('pessoas.jpg')
    #image = imutils.resize(bodyImage, width=min(400, bodyImage.shape[1])) 
    bodys_frame = detect_bodys2(image)
    cv2.imshow('frame', bodys_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
           



def DetectCarInImage():
    carImage = cv2.imread('carroteste.jpg')
    cars_frame = detect_cars(carImage)
    cv2.imshow('frame', cars_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#bodyImage = cv2.imread('Imagem2.jpg')
#DetectCarInImage()

#SimulatorImagem()


    




