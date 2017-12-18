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
def robertx(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    k=np.array([[-1,0],[0,1]])
    img=cv2.filter2D(img,cv2.CV_16S,k)
    img=cv2.convertScaleAbs(img)
    cv2.putText(img,"robertx",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)
def roberty(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    k=np.array([[0,1],[-1,0]])
    img=cv2.filter2D(img,cv2.CV_16S,k)
    img=cv2.convertScaleAbs(img)
    cv2.putText(img,"roberty",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)
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
def robertxy(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    k=np.array([[-1,0],[0,1]])
    imgx=cv2.filter2D(img,cv2.CV_16S,k)
    imgx=cv2.convertScaleAbs(imgx)
    k=np.array([[0,1],[-1,0]])
    imgy=cv2.filter2D(img,cv2.CV_16S,k)
    imgy=cv2.convertScaleAbs(imgy)
    img=cv2.addWeighted(imgx,0.5,imgy,0.5,0)
    cv2.putText(img,"robertxy",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
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
def laplcian(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img=img.astype(np.float64)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    
    img = cv2.convertScaleAbs(laplacian)
    cv2.putText(img,"Laplacian",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)

cap=cv2.VideoCapture(0)

myFuncs=[grey,robertx,sobelx,roberty, sobely,robertxy, sobelxy,laplcian]
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


