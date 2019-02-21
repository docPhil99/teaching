import cv2
import numpy as np
first_frame=None
threshold = 25
subtractor = cv2.createBackgroundSubtractorMOG2(history=threshold, varThreshold=25, detectShadows=True)

def first_frame_method(gray_frame,grab_new): 
    if grab_new: #store first frame
        global first_frame
        first_frame = gray_frame.copy()

    difference = cv2.absdiff(first_frame, gray_frame)
    _, difference = cv2.threshold(difference, threshold, 255, cv2.THRESH_BINARY)
 
    cv2.imshow("First frame", first_frame)
    cimg = cv2.cvtColor(difference,cv2.COLOR_GRAY2BGR)
    cv2.putText(cimg,"first frame threshold {}".format(threshold),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("difference", cimg)
 
def MOG2(frame,_):
    subtractor.setHistory(threshold)
    mask = subtractor.apply(frame)
    cimg = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    cv2.putText(cimg,"MOG frame history {}".format(threshold),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    cv2.imshow("difference", cimg)

cap = cv2.VideoCapture(0)
myFuncs=[first_frame_method,MOG2]
ind=0
grab_new = True
while(True):
    ret,frame=cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    myFuncs[ind](frame,grab_new)
    grab_new=False
    cv2.imshow("Live",frame)
    key= cv2.waitKey(10) & 0xFF 
    if key == ord("q"):
        break
    if key == ord("n"):
        ind+=1
        ind %=len(myFuncs)
    if key == ord('g'):
        grab_new=True
    if key ==ord('u'):
        threshold+=1
        if threshold>255:
            threshold = 255
    if key ==ord('d'):
        threshold-=1
        if threshold<0:
            threshold = 0
    
cap.release()
cv2.destroyAllWindows()
