import cv2
import numpy as np
import math




lower_thresh1 = 129
upper_thresh1 = 255
co_ordinates_for_CONVEX_HULL = [[]]
abc = []

# Open Camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    #capturing the video imgs
    ret,img = cap.read()

    #get hand data from circle
    cv2.rectangle(img, (60, 60), (300, 300), (255, 255, 2), 4)  # outer most rectangle
    crop_image = img[70:300, 70:300]

    grey = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #Applying gaussian blur


    



    lower_red = np.array([0,150,150])
    upper_red = np.array([195,255,255])

    #creating a binary image or mask
    mask = cv2.inRange(hsv,lower_red,upper_red)

    res = cv2.bitwise_and(img,img, mask = mask)

    value = (35,35)
    blur = cv2.GaussianBlur(grey,value, 0)
    _,thresh = cv2.threshold(blur,lower_thresh1,upper_thresh1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

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

        letter_correspond = "A.txt"
        destination = 'Letters_stash_for_sounds/'
        

        output = "A"
        outFile = open('Letters_stash_for_sounds/A.txt', 'w')
        outFile.write(output)

        cv2.putText(img, "A", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)
    elif count_defects == 1:

        if angle_t < 10:
            letter_correspond = "V.txt"

            destination = 'Letters_stash_for_sounds/'
            
            # raw_input("done!!!")

            output = "V"
            outFile = open('Letters_stash_for_sounds/V.txt', 'w')
            outFile.write(output)

            # letter_correspond = "V.txt"

            cv2.putText(img, "V", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        elif 40 < angle_t < 66:

            letter_correspond = "C.txt"

            destination = 'Letters_stash_for_sounds/'
            
            # raw_input("done!!!")

            output = "C"
            outFile = open('Letters_stash_for_sounds/C.txt', 'w')
            outFile.write(output)

            cv2.putText(img, "C", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        elif 20 < angle_t < 35:

            letter_correspond = "L.txt"

            destination = 'Letters_stash_for_sounds/'
            
            # raw_input("done!!!")

            output = "L"
            outFile = open('Letters_stash_for_sounds/L.txt', 'w')
            outFile.write(output)

            cv2.putText(img, "L", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        else:

            letter_correspond = "Y.txt"

            destination = 'Letters_stash_for_sounds/'
            
            # raw_input("done!!!")

            output = "Y"
            outFile = open('Letters_stash_for_sounds/Y.txt', 'w')
            outFile.write(output)

            cv2.putText(img, "Y", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

    elif count_defects == 2:  # Its either W or F

        if angle_t > 100:

            letter_correspond = "F.txt"

            destination = 'Letters_stash_for_sounds/'
            
            # raw_input("done!!!")

            output = "F"
            outFile = open('Letters_stash_for_sounds/F.txt', 'w')
            outFile.write(output)

            cv2.putText(img, "F", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)

        else:

            letter_correspond = "W.txt"

            destination = 'Letters_stash_for_sounds/'
            
            # raw_input("done!!!")

            output = "W"
            outFile = open('Letters_stash_for_sounds/W.txt', 'w')
            outFile.write(output)

            cv2.putText(img, "W", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

    elif count_defects == 4:

        letter_correspond = "CALIBRATE.txt"

        destination = 'Letters_stash_for_sounds/'
        
        # raw_input("done!!!")

        output = "CALIBRATE"
        outFile = open('Letters_stash_for_sounds/CALIBRATE.txt', 'w')
        outFile.write(output)

        cv2.putText(img, "A", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

    else:
        if area > 12000:

            letter_correspond = " B.txt"

            destination = 'Letters_stash_for_sounds/'
            
            # raw_input("done!!!")

            output = "B"
            outFile = open('Letters_stash_for_sounds/B.txt', 'w')
            outFile.write(output)

            cv2.putText(img, "B", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

        else:

            if solidity < 0.85:

                if aspect_ratio < 1:

                    if angle_t < 20:

                        letter_correspond = " D.txt"

                        destination = 'Letters_stash_for_sounds/'
                        
                        # raw_input("done!!!")

                        output = "D"
                        outFile = open('Letters_stash_for_sounds/D.txt', 'w')
                        outFile.write(output)

                        cv2.putText(img, "D", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)

                    elif 169 < angle_t < 180:

                        letter_correspond = " I.txt"

                        destination = 'Letters_stash_for_sounds/'
                        
                        # raw_input("done!!!")

                        output = "I"
                        outFile = open('Letters_stash_for_sounds/I.txt', 'w')
                        outFile.write(output)

                        cv2.putText(img, "I", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)

                    elif angle_t < 168:

                        letter_correspond = " J.txt"

                        destination = 'Letters_stash_for_sounds/'
                        
                        # raw_input("done!!!")

                        output = "J"
                        outFile = open('Letters_stash_for_sounds/J.txt', 'w')
                        outFile.write(output)

                        cv2.putText(img, "J", (320,55),cv2.FONT_HERSHEY_SIMPLEX ,2 , (50,100,190), 3, cv2.LINE_AA)







                elif aspect_ratio > 1.01:

                    letter_correspond = " Y.txt"

                    destination = 'Letters_stash_for_sounds/'
                    
                    # raw_input("done!!!")

                    output = "Y"
                    outFile = open('Letters_stash_for_sounds/Y.txt', 'w')
                    outFile.write(output)

                    cv2.putText(img, "Y", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)

            else:

                if angle_t > 30 and angle_t < 100:

                    letter_correspond = " H.txt"

                    destination = 'Letters_stash_for_sounds/'
                    
                    # raw_input("done!!!")

                    output = "H"
                    outFile = open('Letters_stash_for_sounds/H.txt', 'w')
                    outFile.write(output)

                    cv2.putText(img, "H", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)





                elif angle_t > 120:

                    letter_correspond = " I.txt"

                    destination = 'Letters_stash_for_sounds/'
                    
                    # raw_input("done!!!")

                    output = "I"
                    outFile = open('Letters_stash_for_sounds/I.txt', 'w')
                    outFile.write(output)

                    cv2.putText(img, "I", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)



                else:

                    letter_correspond = " U.txt"

                    destination = 'Letters_stash_for_sounds/'
                    
                    # raw_input("done!!!")

                    output = "U"
                    outFile = open('Letters_stash_for_sounds/U.txt', 'w')
                    outFile.write(output)

                    cv2.putText(img, "U", (320, 55), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 100, 190), 3, cv2.LINE_AA)


    cv2.imshow('Gesture',img)
    cv2.imshow('Contours',drawing)
    cv2.imshow('Defects',crop_image)
    cv2.imshow('Binary IMage',thresh)

    if cv2.waitKey(1)==ord('q'):
        break

cv2.destroyAllWindows()