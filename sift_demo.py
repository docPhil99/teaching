import cv2
import sys


# function to detect the features by finding key points and descriptors from the image
def detector(image1, image2):
    # creating ORB detector
    detect = cv2.SIFT_create() #cv2.ORB_create()

    # finding key points and descriptors of both images using detectAndCompute() function
    key_point1, descrip1 = detect.detectAndCompute(image1, None)
    key_point2, descrip2 = detect.detectAndCompute(image2, None)
    return (key_point1, descrip1, key_point2, descrip2)


def BF_FeatureMatcher(des1, des2):
    brute_force = cv2.BFMatcher(cv2.NORM_L2)# cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    no_of_matches = brute_force.match(des1, des2)

    # finding the humming distance of the matches and sorting them
    no_of_matches = sorted(no_of_matches, key=lambda x: x.distance)
    return no_of_matches


# function displaying the output image with the feature matching
def display_output(pic1, kpt1, pic2, kpt2, best_match):
    # drawing the feature matches using drawMatches() function
    output_image = cv2.drawMatches(pic1, kpt1, pic2, kpt2, best_match, None, flags=2)
    cv2.imshow('Output image', output_image)


target_image = cv2.imread('target_image.png')
target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)


cap = cv2.VideoCapture(0) #webcam
if not cap.isOpened():
    print("Error opening video stream or file")
    sys.exit(1)
print('Press q quit')
while True:
    ret, frame = cap.read()
    if not ret:
        print('bad frame')
        break
    frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame2)
    key_pt1, descrip1, key_pt2, descrip2 = detector(target_image, frame2)

    # sorting the number of best matches obtained from brute force matcher
    number_of_matches = BF_FeatureMatcher(descrip1, descrip2)
    tot_feature_matches = len(number_of_matches)

    # printing total number of feature matches found
    print(f'Total Number of Features matches found are {tot_feature_matches}')

    # after drawing the feature matches displaying the output image
    display_output(target_image, key_pt1, frame2, key_pt2, number_of_matches)

    res =  cv2.waitKey(10)
    if res ==ord('q'):
        break
    if res == ord('s'):
        cv2.imwrite('target_image.png',frame)


