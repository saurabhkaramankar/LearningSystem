import cv2
import time
import os
import HandTrackingModule as htm

wCam, hCam = 648, 488

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

path = "fingerImages"
myList = os.listdir(path)
pathList = []
for imgPath in myList:
    image = cv2.imread(f'{path}/{imgPath}')
    pathList.append(image)

pTime = 0

detector = htm.handDetector(detectionCon=0.75)

tipId = [4,8,12,16,20]

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    pointList = detector.findPosition(img, draw=False)
    # print(pointList)
    if(len(pointList)!=0):
        fingers = []
        if (pointList[tipId[0]][1] < pointList[tipId[0]-1][1]):
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if(pointList[tipId[id]][2] < pointList[tipId[id]-2][2]):
                fingers.append(1)
            else:
                fingers.append(0)
        totalFingers = fingers.count(1)

        h,w,c = pathList[0].shape
        img[0:h,0:w] = pathList[totalFingers-1]
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}',(500,50), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(5)