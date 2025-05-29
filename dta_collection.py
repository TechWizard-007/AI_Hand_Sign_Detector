import cv2                                   #p1 video access
from cvzone.HandTrackingModule import HandDetector #p2 detecting hand
import numpy as np #p4 'sqarification and white bg' for making a square matrix
import math #p6 
import time #p7 #time module will be used so that image captured once could not have same name as other
#p5 'putting croped image on the white image'
#p6 'putting imgcrop at the centre of the white bg for making it easy for the classifier to detect the hand' solution: 'if height is bigger than width, will make the height 300 if width is bigger than height then width will be stretched out till 300., now after stretching the hight, how much we have to stretch the width, that we'll calculate, and once we get those values, we'll put those values at the centre.


cap = cv2.VideoCapture(0) #p1 video access
detector = HandDetector(maxHands=1) #p2 detecting hand
offset = 20 # for p3 'croping hands' and giving a clear crop with some space. #p3 'croping the hands'
imgsize = 300

folder = r"college_proj_1_aiml\dta\washroom" #p7 saving image data whenever 's' key is pressed" 
counter = 0 #p7 #for checking number of images saved

while True: #p1 video access
    success,img = cap.read() #p1 video access
    hands, img = detector.findHands(img) #p2 detecting hand

    if hands: #p3 'croping the hands'
        hand = hands[0] #since only one hand is there in the list. #p3 'croping the hands'
        x, y, w, h = hand['bbox'] #p3 'croping the hands'

        imgWhite = np.ones((imgsize,imgsize,3),np.uint8)*255 #p4 #keeping it a square matrix (imgsize,imgsize,3), np.unit8 because values will be between 0 to 255 so it'll not show correct parameters, multiplied by 255 because pxiel values are 1 and if not multiplied by 255 then it'll only show a black image

        imgCrop = img[y-offset:y + h+offset , x-offset:x + w+offset ] # y = starting of the image, y+h means ending of image in the height direction same way for x and w #p3 'croping the hands'
        imgCropShape = imgCrop.shape #returns the height and width to the imgWhite for p5 


        aspectRatio = h/w #p6

        if aspectRatio>1: #p6 #if the height greater than width
            k = imgsize/h #p6 #stretching the height #const required to calculate the width
            width_calculated = math.ceil(k*w) #p6 #if val is 3.2 or 3.5 it'll always go to 4, only the upper value
            imgResize = cv2.resize(imgCrop,(width_calculated,imgsize))
            imgResizeShape = imgResize.shape
            width_gap = math.ceil((imgsize-width_calculated)/2) #p6 #finding the width gap to bring image in the centre
            imgWhite[:,width_gap:width_calculated+width_gap] = imgResize  #p5 #putting image crop values at the imageWhite values at the given points/values imgcropshape[0] -> height , imgcropshape[1] -> width #p6 due to p6, height is retyped as ':' so don't mind, here we're just fixing the image
# in case width > height
        else:
            k = imgsize/w #p6 #stretching the height #const required to calculate the width
            height_calculated = math.ceil(k*h) #p6 #if val is 3.2 or 3.5 it'll always go to 4, only the upper value
            imgResize = cv2.resize(imgCrop,(imgsize,height_calculated))
            imgResizeShape = imgResize.shape
            height_gap = math.ceil((imgsize-height_calculated)/2) #p6 #finding the width gap to bring image in the centre
            imgWhite[height_gap:height_calculated+height_gap,:] = imgResize  #p5 #putting image crop values at the imageWhite values at the given points/values imgcropshape[0] -> height , imgcropshape[1] -> width #p6 due to p6, height is retyped as ':' so don't mind, here we're just fixing the image

        cv2.imshow("croped image",imgCrop) #p3 'croping the hands'
        cv2.imshow("Imagewhite",imgWhite) #p4

    cv2.imshow("Image",img) #p1 video access
    key = cv2.waitKey(1) #1 millisecond delay #p1 video access #p7

    if key == ord("s"):
        counter += 1 #p7 #increments when an image saved
        cv2.imwrite(f'{folder}\Image_{time.time()}.jpg',imgWhite) #p7 #works like where you wanna save and then what you wanna save
        print(counter) #p7 #prints number of images saved