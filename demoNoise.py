import cv2
if cv2.__version__.startswith('3'):
    print("Version 2 of opencv required")
    
import numpy as np

def add_sp(img):
    sp = 0.5
    amount=0.04
    num_salt=np.ceil(amount*img.size*sp)
    cood=[np.random.randint(0,i-1,int(num_salt)) for i in img.shape]
    img[cood]=1


    num_pepper=np.ceil(amount*img.size*(1.0-sp))
    cood=[np.random.randint(0,i-1,int(num_pepper)) for i in img.shape]
    img[cood]=0
def noise(image):
    #add some noise
    #print(image.size)
    img=np.zeros(image.shape,dtype=np.uint8)
    cv2.randn(img,(0,0,0),(20,20,20))
    image+=img
    cv2.putText(image," G Noise",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",image)


def noise2(image):
    #add some noise
    #print(image.size)
    add_sp(image)
    cv2.putText(image," SP Noise",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)

    cv2.imshow("Test",image)


def denoise1(image):
    #add some noise
    #print(image.size)
    img=np.zeros(image.shape,dtype=np.uint8)
    cv2.randn(img,(0,0,0),(20,20,20))
    image+=img
    kern=np.ones((ksize,ksize),np.float)/(ksize*ksize)
    img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst=cv2.filter2D(img2,-1,kern)
    cv2.putText(dst,"G Noise mean filter {}".format(ksize),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",dst)


def denoise2(image):
    #add some noise
    #print(image.size)
    img=np.zeros(image.shape,dtype=np.uint8)
    cv2.randn(img,(0,0,0),(20,20,20))
    image+=img
    kern=np.ones((ksize,ksize),np.float)/(ksize*ksize)
    img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst=cv2.medianBlur(img2,ksize)
    cv2.putText(dst,"G Noise median filter {}".format(ksize),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",dst)

def denoise3(image):
    #add some noise
    #print(image.size)

    add_sp(image)
    kern=np.ones((ksize,ksize),np.float)/(ksize*ksize)
    img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst=cv2.filter2D(img2,-1,kern)
    cv2.putText(dst," SP Noise mean filter",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",dst)


def denoise4(image):
    #add some noise
    add_sp(image)
    kern=np.ones((ksize,ksize),np.float)/(ksize*ksize)
    img2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst=cv2.medianBlur(img2,ksize)
    cv2.putText(dst,"SP Noise median filter {}".format(ksize),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",dst)
cap=cv2.VideoCapture(0)

ksize = 5
myFuncs=[noise,noise2,denoise1,denoise2,denoise3,denoise4]
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
    if key ==ord('w'):
        ksize+=2
        if ksize>255:
            ksize = 255
        
    if key ==ord('s'):
         ksize-=2
         if ksize<1:
            ksize = 1


cap.release()


