import cv2
if cv2.__version__.startswith('3'):
    print("Version 2 of opencv required")
    
import numpy as np

    
def threshold(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,img=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.putText(cimg,"threshold {}".format(threshold),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("Test",cimg)

def threshold2(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    # uses the gaussian weighted mean over 11x11 pixels, threshold is calcuated at each pixel and is 2 below this mean
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.putText(cimg,"adaptive threshold ",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("Test",cimg)

cap=cv2.VideoCapture(0)

myFuncs=[threshold, threshold2]
ind=0
threshold=127
while(True):
    ret,frame=cap.read()
    
    myFuncs[ind](frame)
    cv2.imshow("Live",frame)
    key= cv2.waitKey(10) & 0xFF 
    if key == ord("q"):
        break
    if key == ord("n"):
        ind+=1
        ind %=len(myFuncs)
    if key ==ord('u'):
        threshold+=1
        if threshold>255:
            threshold = 255
    if key ==ord('d'):
        threshold-=1
        if threshold<0:
            threshold = 0
    

cap.release()


