import cv2
import numpy as np
from numpy.lib.type_check import imag
import handDetector as hd
import time
import autopy

wCam, hCam = 640, 480
frameR = 100
smothening = 7

pTime = 0
plockX, plocY = 0, 0
clocX, clocY = 0, 0

cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)
detector = hd.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()

while True:
    success, image = cam.read()
    image = detector.findHands(image)
    lmList, bbox = detector.findPosition(image)

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        fingers = detector.fingersUp()

        cv2.rectangle(image, (frameR, frameR),
                      (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

        if fingers[1] == 1 and fingers[2] == 0:
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScr))

            clocX = plockX + (x3 - plockX) / smothening
            clocY = clocY + (y3 - plocY) / smothening

            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(image, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plockX, plocY = clocX, clocY

        if fingers[1] == 1 and fingers[2] == 1:
            length, image, lineInfo = detector.findDistance(8, 12, image)
            print(length)

            if length < 40:
                cv2.circle(
                    image, (lineInfo[4], lineInfo[5]), 4, (0, 255, 9), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(image, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12. Display
    cv2.imshow("Image", image)
    cv2.waitKey(1)
