import cv2
if cv2.__version__.startswith('3'):
    print("Version 2 of opencv required")
    
import numpy as np

def grey(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.putText(img,"grey scale",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)
    
def threshold(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    cv2.putText(img,"threshold",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)

def noise(image):
    #add some noise
    #print(image.size)
    img=np.zeros(image.shape,dtype=np.uint8)
    cv2.randn(img,(0,0,0),(20,20,20))
    image+=img
    cv2.putText(image,"Noise",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",image)

def denoise1(image):
    #add some noise
    #print(image.size)
    img=np.zeros(image.shape,dtype=np.uint8)
    cv2.randn(img,(0,0,0),(20,20,20))
    image+=img
    kern=np.ones((5,5),np.float)/25
    img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst=cv2.filter2D(img2,-1,kern)
    cv2.putText(dst,"De Noise mean filter",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",dst)
def sobelx(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img=cv2.Sobel(img,cv2.CV_16S,1,0,3,scale=1)
    img=cv2.convertScaleAbs(img)
    cv2.putText(img,"sobelx",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)
def sobely(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img=cv2.Sobel(img,cv2.CV_16S,0,1,3,scale=1)
    img=cv2.convertScaleAbs(img)
    cv2.putText(img,"sobely",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)
def sobelxy(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgx=cv2.Sobel(img,cv2.CV_16S,1,0,3,scale=1)
    imgx=cv2.convertScaleAbs(imgx)
    imgy=cv2.Sobel(img,cv2.CV_16S,0,1,3,scale=1)
    imgy=cv2.convertScaleAbs(imgy)
    img=cv2.addWeighted(imgx,0.5,imgy,0.5,0)
    cv2.putText(img,"sobelxy",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)

def face(img):
    rects = cascade.detectMultiScale(img, 1.3, 5)
    if len(rects) > 0:
        rects[:, 2:] += rects[:, :2]
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)
    cv2.putText(img,"face",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)


cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")   
cap=cv2.VideoCapture(0)

myFuncs=[grey,face,noise,denoise1,threshold,sobelx, sobely, sobelxy]
ind=0
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


cap.release()


