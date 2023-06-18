import cv2
import numpy as np
import math

from os import path


lower_thresh1 = 129
upper_thresh1 = 255
co_ordinates_for_CONVEX_HULL = [[]]
abc = []

# Open Camera
capture = cv2.VideoCapture(0)

while capture.isOpened():
    #capturing the video frames
    ret,frame = capture.read()

    #get hand data from circle
    cv2.rectangle(frame, (60, 60), (300, 300), (255, 255, 2),4)  # outer most rectangle
    crop_image = frame[70:300, 70:300]




    #Applying gaussian blur
    blur = cv2.GaussianBlur(crop_image,(3,3),0)

    grey = cv2.cvtColor(crop_image,cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)



    #creating a binary image or mask
    mask = cv2.inRange(hsv,np.array([2,0,0]),np.array([20,255,255]))

    #res = cv2.bitwise_and(frame,frame, mask = mask)

    value = (35,35)
    kernel = np.ones((5,5))
    dialation = cv2.dilate(mask,kernel,iterations=1)
    erosion = cv2.erode(dialation,kernel,iterations=1)
    filtered = cv2.GaussianBlur(erosion,(3,3),0)
    ret, thresh = cv2.threshold(filtered,127,255,0)


    #finding contours
    contours,hierarchy =cv2.findContours(thresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #finding contour with max area
    contour = max(contours,key= lambda x:cv2.contourArea(x))
    #creatina a bounding area around the contour

    area_of_contour = cv2.contourArea(contour)

    x,y,w,h =cv2.boundingRect(contour)
    cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,0,255),0)

    #finding convex hull

    hull = cv2.convexHull(contour)
    drawing = np.zeros(crop_image.shape,np.uint8)
    cv2.drawContours(drawing, [contour], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 255, 255), 0)

    hull = cv2.convexHull(contour,returnPoints=False)

    defects = cv2.convexityDefects(contour,hull)

    count_defects = 0

    cv2.drawContours(thresh,contours,-1,(0,255,0),3)
    #usnig cosine rule to find anfle of far point from the start and end point i.e the convex points(finger tips)
    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(contour[s][0])
        end = tuple(contour[e][0])
        far = tuple(contour[f][0])

        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 60
        #red circle
        cv2.circle(crop_image, far, 4, [0, 0, 255], -1)

        if angle <= 90:
            count_defects = count_defects+1
        cv2.line(crop_image,start,end,[255,0],3)

    moment = cv2.moments(contour)
    perimeter = cv2.arcLength(contour,True)
    area = cv2.contourArea(contour)

    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    cv2.circle(crop_image, center, radius, (255, 0, 0), 2)

    area_of_circle = math.pi*radius*radius

    hull_test =cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull_test)
    solidity = float(area)/hull_area

    aspect_ratio = float(w)/h

    rect_area = w*h
    extent = float(area)/rect_area

    (x, y), (MA, ma), angle_t = cv2.fitEllipse(contour)

    if area_of_circle - area <5000:

        #letter_correspond = "A.txt"
        #destination = 'Letters_stash_for_sounds/'
        #createFile(destination)

        #output = "A"
        #outFile = open('Letters_stash_for_sounds/A.txt', 'w')
        #outFile.write(output)

        cv2.putText(frame, "A", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)
    elif count_defects == 1:

        if angle_t < 10:


            cv2.putText(frame, "V", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        elif 40 < angle_t < 66:



            cv2.putText(frame, "C", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        elif 20 < angle_t < 35:


            cv2.putText(frame, "L", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        else:


            cv2.putText(frame, "Y", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

    elif count_defects == 2:  # Its either W or F

        if angle_t > 100:



            cv2.putText(frame, "F", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)

        else:



            cv2.putText(frame, "W", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

    elif count_defects == 4:



        cv2.putText(frame, "A", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

    else:
        if area > 12000:



            cv2.putText(frame, "B", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        else:

            if solidity < 0.85:

                if aspect_ratio < 1:

                    if angle_t < 20:



                        cv2.putText(frame, "D", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)

                    elif 169 < angle_t < 180:



                        cv2.putText(frame, "I", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

                    elif angle_t < 168:



                        cv2.putText(frame, "J", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)







                elif aspect_ratio > 1.01:



                    cv2.putText(frame, "Y", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)

            else:

                if angle_t > 30 and angle_t < 100:



                    cv2.putText(frame, "H", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)





                elif angle_t > 120:



                    cv2.putText(frame, "I", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)



                else:



                    cv2.putText(frame, "U", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)


    cv2.imshow('Gesture',frame)
    cv2.imshow('Contours',drawing)
    cv2.imshow('Defects',crop_image)
    cv2.imshow('Binary IMage',thresh)























    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()