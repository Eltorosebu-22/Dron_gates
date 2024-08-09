import cv2
import numpy as np
from djitellopy import Tello
import time
width = 640  # WIDTH OF THE IMAGE
height = 480  # HEIGHT OF THE IMAGE
deadZone =100
Start=0
sensors = (2, 3)

me = Tello()
me.connect()
print(me.get_battery())

me.streamoff()
me.streamon()

cvalue=[]

frameWidth = 640
frameHeight = 480

frameWidth_Half=640
frameHeight_Half=480
"""cap = cv2.VideoCapture(1)
cap.set(3, frameWidth)
cap.set(4, frameHeight)"""

deadZone=100
global cronologia
global ajustarvuelta
global modoturn
global areashow
global area
global imgContour
global imgContourHalf
global movedrone
global cxf
global cyf
global moveflag
global movex
global izquierda
global derecha
global arriba
global abajo
global AAB2
global biggest
biggest=0
izquierda=-10
derecha=10
arriba=10
abajo=-10
ajustarvuelta=0
cronologia=1
modoturn=0
areashow=0
area=0
moveflag=False
movex=0 #movimiento adelante atras
cxf = 0 #movimiento izquierda derecha
cyf = 0 #movimiento arriba abajo
senstivity = 4  # if number is high less sensitive
movedrone=0
AAB2=0
startCounter=0
def empty(a):
    pass
hsvVals=[
    [29, 130, 78, 35, 157, 93],
    [30, 124, 92, 35, 146, 104],
    [31, 122, 93, 35, 141, 98],
    [31, 133, 89, 35, 146, 102],
    [30, 111, 168, 32, 144, 205],
    [29, 115, 151, 32, 153, 196],
    [29, 119, 117, 33, 177, 170],
    [29, 118, 116, 33, 170, 171],
    [31, 133, 66, 36, 145, 75],
    [27, 126, 69, 35, 168, 95],
    [27, 92, 112, 39, 135, 151],
    [27, 131, 64, 34, 157, 73],
    [37, 124, 134, 40, 146, 149],
    [37, 125, 135, 40, 151, 146],
    [36, 126, 135, 40, 145, 151],
    [36, 126, 134, 40, 150, 146],
    [41, 60, 121, 54, 120, 162],
    [34, 102, 103, 43, 132, 119],
    [36, 112, 105, 41, 139, 116]
]
hsv=[2, 76, 166, 179, 140, 255]
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",166,255,empty)
cv2.createTrackbar("Threshold2","Parameters",171,255,empty)
cv2.createTrackbar("Area","Parameters",1500,30000,empty)

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver
def thresholding(img_half, hsv_vals):
    hsv = cv2.cvtColor(img_half, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_vals[:3])
    upper = np.array(hsv_vals[3:])
    mask_half2 = cv2.inRange(hsv, lower, upper)
    result_half2 = cv2.bitwise_and(img_half, img_half, mask=mask_half2)
    mask_half = cv2.cvtColor(mask_half2, cv2.COLOR_GRAY2BGR)
    return result_half2
def hsv(*args):
    combined_hsv = np.zeros_like(args[0])
    for img in args:
        combined_hsv = np.where(img != 0, img, combined_hsv)
    return combined_hsv
def fill_closed_areas(mask):
    # Encuentra los contornos de los objetos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Rellena los contornos cerrados con puntos blancos
    filled_mask = np.zeros_like(mask)
    cv2.drawContours(filled_mask, contours, -1, (255), thickness=cv2.FILLED)
    return filled_mask
def getContoursHalf(img_half,imgContourHalf):
    global movedrone
    global cvalue
    global cxf
    global cyf
    global moveflag
    global areashow
    global area
    contours, hierarchy = cv2.findContours(img_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if area > areaMin :
            moveflag=True
            areashow=area
            cv2.drawContours(imgContourHalf, cnt, -1, (255, 0, 255), 7)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContourHalf, (x, y), (x + w, y + h), (0, 255, 0), 5)

            cv2.putText(imgContourHalf, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
                        (0, 255, 0), 2)
            cv2.putText(imgContourHalf, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                        (0, 255, 0), 2)
            cv2.putText(imgContourHalf, " " + str(int(x)) + " " + str(int(y)), (x - 20, y - 45), cv2.FONT_HERSHEY_COMPLEX,
                        0.7,
                        (0, 255, 0), 2)

            cx = int(x + (w / 2))
            cy = int(y + (h / 2))
            cxf=cx
            cyf=cy

            if (cx < int(frameWidth_Half / 2) - deadZone):
                cv2.putText(imgContourHalf, " GO LEFT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
               # cv2.rectangle(imgContourHalf, (0, int(frameHeight_Half / 2 - deadZone)),
                             # (int(frameWidth_Half / 2) - deadZone, int(frameHeight_Half / 2) + deadZone), (0, 0, 255),
                            #  cv2.FILLED)
                movedrone=1
            elif (cx > int(frameWidth_Half / 2) + deadZone):
                cv2.putText(imgContourHalf, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                #cv2.rectangle(imgContourHalf, (int(frameWidth_Half / 2 + deadZone), int(frameHeight_Half / 2 - deadZone)),
                              #(frameWidth_Half, int(frameHeight_Half / 2) + deadZone), (0, 0, 255), cv2.FILLED)
                movedrone=2
            elif (cy < int(frameHeight_Half / 2) - deadZone):
                cv2.putText(imgContourHalf, " GO UP", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
               # cv2.rectangle(imgContourHalf, (int(frameWidth_Half / 2 - deadZone), 0),
                             # (int(frameWidth_Half / 2 + deadZone), int(frameHeight_Half / 2) - deadZone), (0, 0, 255),
                             # cv2.FILLED)
                movedrone=3
            elif (cy > int(frameHeight_Half / 2) + deadZone):
                cv2.putText(imgContourHalf, " GO DOWN", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
                #cv2.rectangle(imgContourHalf, (int(frameWidth_Half / 2 - deadZone), int(frameHeight_Half / 2) + deadZone),
                              #(int(frameWidth_Half / 2 + deadZone), frameHeight_Half), (0, 0, 255), cv2.FILLED)
                movedrone=4
            else:
                movedrone=0

            cv2.line(imgContourHalf, (int(frameWidth_Half / 2), int(frameHeight_Half / 2)), (cx, cy),
                     (0, 0, 255), 3)
def displayhalf(img_half):
    cv2.line(img_half, (int(frameWidth_Half / 2) - deadZone, 0), (int(frameWidth_Half / 2) - deadZone, frameHeight_Half), (255, 255, 0), 3)
    cv2.line(img_half, (int(frameWidth_Half / 2) + deadZone, 0), (int(frameWidth_Half / 2) + deadZone, frameHeight_Half), (255, 255, 0), 3)

    cv2.circle(img_half, (int(frameWidth_Half / 2), int(frameHeight_Half / 2)), 5, (0, 0, 255), 5)
    cv2.line(img_half, (0, int(frameHeight_Half / 2) - deadZone), (frameWidth_Half, int(frameHeight_Half / 2) - deadZone), (255, 255, 0), 3)
    cv2.line(img_half, (0, int(frameHeight_Half / 2) + deadZone), (frameWidth_Half, int(frameHeight_Half / 2) + deadZone), (255, 255, 0), 3)
def sensorOutput():
    global cxf
    global cyf
    global movedrone
    global moveflag
    global areashow
    global movex
    global modoturn
    global cronologia
    global izquierda
    global derecha
    global arriba
    global abajo
    global AAB
    global area
    global AAB2
    # Traslación
    lr = (cxf - width // 2) // senstivity
    lr = int(np.clip(lr, -15, 15))
    if 5 > lr > -5: lr = 0
    # Arriba abajo
    AAB=(cyf - height // 2) // senstivity*-1
    AAB=int(np.clip(AAB, -15, 15))
    if 5 > AAB > -5: AAB= 0
    print("esto es areshoooooooooooow",areashow)
    #Desplazamiento
    if areashow<92000 and areashow!=0 and areashow<30000:
        movex=8
    elif areashow<92000 and areashow!=0 and areashow<30000:
        movex=10
    else:
        movex=0
    #arriba abajo 2
    if movedrone==3:
        AAB2=10
    elif movedrone==4:
        AAB2=-10
    #movimiento comandoss#
    if areashow>60000:
        modoturn=1
    if areashow<1400:
        me.send_rc_control(0,0,0,0)
    if modoturn==0 and moveflag==True and area>1500:
        #print("si entroooo")
        me.send_rc_control(lr, movex, AAB, 0)
    if modoturn==1 and cronologia==1:
        me.send_rc_control(0,0,0,0)
        time.sleep(1)
        me.send_rc_control(20, 0, 0, 0) #se mueve 60 cm
        time.sleep(2.2)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        me.send_rc_control(derecha,0,0,0)
        time.sleep(3)
        me.send_rc_control(0,20,0,0)
        time.sleep(5)
        me.rotate_clockwise(-90)
        time.sleep(1)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        me.send_rc_control(0,20,0,0)
        time.sleep(2)
        cronologia=2
        modoturn=0
    if modoturn == 1 and cronologia == 2:
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(0.5)
        me.send_rc_control(20, 0, 0, 0) ##se mueve 60 cm
        time.sleep(2.2)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        me.send_rc_control(0, 0,arriba, 0)
        time.sleep(3)
        me.send_rc_control(0, 20, 0, 0)
        time.sleep(6)
        me.rotate_clockwise(-90)
        time.sleep(1)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        cronologia = 2
        modoturn = 0
    print(me)
while True:
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))
    #mitad_superior = imghalf[0:height // 2, 0:width]
   # thrirdheight=height//3
   # imghalf = img[height // 2:height, 0:width]
   # imghalf=img[0:height,0:width//3*2]
    imghalf=img
    img_half=cv2.resize(imghalf,(width,height))
    imgHsvHalf = cv2.cvtColor(img_half, cv2.COLOR_BGR2HSV)
    if img_half is not None:
        hsv_images = []
        for hsv_vals in hsvVals:
            imgThres = thresholding(img_half, hsv_vals)
            # hsv_image = fill_closed_areas(imgThres)
            hsv_images.append(imgThres)
        result_half = hsv(*hsv_images)
    imgContourHalf = img_half.copy()
    threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
    threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
    ###################################################################33
    imgBlur_half = cv2.GaussianBlur(result_half, (7, 7), 1)
    imgGray_half = cv2.cvtColor(imgBlur_half, cv2.COLOR_BGR2GRAY)
    imgCanny_half = cv2.Canny(imgGray_half, threshold1, threshold2)
    kernel = np.ones((5, 5))
    imgDil_half = cv2.dilate(imgCanny_half, kernel, iterations=1)
    getContoursHalf(imgDil_half, imgContourHalf)
    displayhalf(imgContourHalf)
    sensorOutput()

    stack = stackImages(0.7, ([imgContourHalf, result_half], [imgDil_half, img_half]))

    cv2.imshow('Horizontal Stacking', stack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
me.land()
cap.release()
cv2.destroyAllWindows()
