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

def threshold3(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,img=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    kernel = np.ones((ksize,ksize),np.uint8)
    img = cv2.erode(img,kernel, iterations=1);
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.putText(cimg,"threshold {} erode {}".format(threshold,ksize),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("Test",cimg)

def threshold4(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,img=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    kernel = np.ones((ksize,ksize),np.uint8)
    img = cv2.dilate(img,kernel, iterations=1);
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.putText(cimg,"threshold {} dilate {}".format(threshold,ksize),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("Test",cimg)

def threshold5(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,img=cv2.threshold(img,threshold,255,cv2.THRESH_BINARY)
    kernel = np.ones((ksize,ksize),np.uint8)
    img = cv2.dilate(img,kernel, iterations=1);
    img = cv2.erode(img,kernel, iterations=1);
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    cv2.putText(cimg,"threshold {} both {}".format(threshold,ksize),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("Test",cimg)


cap=cv2.VideoCapture(0)

ksize =5
myFuncs=[threshold, threshold3,threshold4, threshold5]
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
    
    if key ==ord('w'):
        ksize+=2
        if ksize>255:
            ksize = 255
        
    if key ==ord('s'):
         ksize-=2
         if ksize<1:
            ksize = 1

cap.release()


