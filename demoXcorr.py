import cv2
if cv2.__version__.startswith('2'):
    print("Version 3 of opencv required")
    
import numpy as np

def grey(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    cv2.putText(img,"grey scale",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)
    
def xcorr(image):
    if target.size==0:
        return
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    res=cv2.matchTemplate(img,target,cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    res=(res/max_val)
    #resi=res.astype(np.uint8)
    #cv2.putText(image,"xcorr",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.putText(image,"xcorr Max val {:.2f}".format(max_val),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(1),2)
    cv2.imshow("Test",res)


def laplcian(image):
    img=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    img=img.astype(np.float64)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    
    img = cv2.convertScaleAbs(laplacian)
    cv2.putText(img,"Laplacian",(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255),2)
    cv2.imshow("Test",img)

cap=cv2.VideoCapture(0)

myFuncs=[grey,xcorr,laplcian]
ind=0
target=np.array([])
while(True):
    ret,frame=cap.read()
    
    myFuncs[ind](frame)
    cv2.imshow("Live",frame)
    key= cv2.waitKey(10) & 0xFF 
    if key == ord("q"):
        break
    if key == ord("g"):
        target=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        target=target[200:280,280:360]
        cv2.imshow("Target",target)
    if key == ord("n"):
        ind+=1
        ind %=len(myFuncs)


cap.release()
cv2.destroyAllWindows()


