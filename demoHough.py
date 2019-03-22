import cv2
if cv2.__version__.startswith('2'):
    print("Version 3 of opencv required")
    
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
def canny(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    img = cv2.Canny(img,100,200,apertureSize=3)
    cv2.putText(img,"Canny",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)

def hough(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    
    img = cv2.Canny(img,100,200,apertureSize=3)
    
    lines=cv2.HoughLinesP(img,1,np.pi/180,threshold=100, minLineLength=30,maxLineGap=10)
    if lines is not None:
        for x in range(0 , len(lines)):
            for x1,y1,x2,y2 in lines[x]:
                cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)

    cv2.putText(img,"Canny",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)
cap=cv2.VideoCapture(0)

myFuncs=[grey,canny,hough]
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
cv2.destroyAllWindows()


