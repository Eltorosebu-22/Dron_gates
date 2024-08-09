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
global areaMin
global shape
triangle=1 #0 abajo 1 arriba
rombo=1#0 izquierda 1 derecha
circulo=1
hexagono=0
areaMin=0
biggest=0
izquierda=-10
derecha=10
arriba=10
abajo=-10
cronologia=1
modoturn=0
areashow=0
area=0
shape="romboide"
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
    [31, 132, 52, 39, 171, 59],
    [32, 136, 56, 35, 163, 63],
    [25, 114, 54, 41, 169, 65],
    [33, 119, 76, 42, 157, 83],
    [35, 104, 157, 43, 137, 178],
    [38, 134, 73, 38, 145, 76],
    [29, 106, 63, 41, 154, 74],
    [37, 105, 165, 41, 134, 186],
    [35, 104, 157, 43, 137, 178],
    [35, 119, 110, 41, 156, 120],
    [34, 118, 77, 40, 137, 81],
    [36, 121, 82, 39, 149, 86],
    [35, 121, 157, 40, 139, 170],
    [37, 100, 185, 40, 123, 195],
    [38, 111, 176, 40, 134, 186],
    [36, 129, 62, 42, 155, 66],
    [36, 119, 82, 41, 143, 90],
    [32, 115, 62, 40, 162, 69],
    [36, 126, 98, 41, 159, 111],
################################
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
    global areamin
    global biggest
    global shape
    global cronologia
    x1, y1, x2, y2 = 0, 0, 640, 480
    roi = imgContourHalf[y1:y2, x1:x2]
    contours, hieracrhy = cv2.findContours(img_half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) != 0:
        biggest = max(contours, key=cv2.contourArea)
        # Obtiene las coordenadas del contorno en la imagen de la ROI
        x, y, w, h = cv2.boundingRect(biggest)
        # Ajusta las coordenadas del contorno para la imagen original
        cx = x + w // 2
        cy = y + h // 2
        cxf = cx
        cyf = cy
        # Dibuja el contorno en la imagen original
        cv2.rectangle(imgContourHalf, (x, y), (x + w, y + h), (0, 255, 0), 5)
        cv2.circle(imgContourHalf, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
        cv2.rectangle(imgContourHalf, (x1, y1), (x2, y2), (0, 0, 0), 2)
        # Dibuja líneas horizontales y verticales desde el punto (cx, cy)
        cv2.line(imgContourHalf, (cx, 0), (cx, img.shape[0]), (0, 0, 255), 2)  # Línea vertical
        cv2.line(imgContourHalf, (0, cy), (img.shape[1], cy), (0, 0, 255), 2)  # Línea horizontal
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = cv2.getTrackbarPos("Area", "Parameters")
        if cronologia==1 or cronologia==3 or cronologia==4:
            areaMin = cv2.getTrackbarPos("Area", "Parameters")
        elif cronologia==2:
            areaMin=2500
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # Determina el tipo de forma
        num_sides = len(approx)
        if area > areaMin:
            moveflag = True
            areashow = area
        shape = "Desconocido"
        if num_sides == 3:
            shape = "Triangulo"
        elif num_sides == 4:
            # Comprueba si es un rombo
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Rombo"
            else:
                shape = "Rombo"  # También podría ser un rombo
        elif num_sides == 6:
            shape = "Hexagono"
        elif num_sides > 6:
            shape = "Circulo"

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
    global areaMin
    print("areaminnn",areaMin)
    print("figura detectadaaaaa",shape)
    # Traslación
    lr = (cxf - width // 2) // senstivity
    lr = int(np.clip(lr, -15, 15))
    if 5 > lr > -5: lr = 0
    # Arriba abajo
    AAB=(cyf - height // 2) // senstivity*-1
    AAB=int(np.clip(AAB, -15, 15))
    if 5 > AAB > -5: AAB= 0
    print("esto es areshoooooooooooow",areashow)
    print("areaaaaaaaaaaa", area)
    print()
    #Desplazamiento
    if areashow<92000 and areashow!=0 and areashow<5000:
        movex=15
    elif areashow<92000 and areashow!=0 and areashow>3000:
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
    if area<areaMin and area!=0:
        me.send_rc_control(0,0,0,0)
    elif modoturn==0 and area > areaMin:
        #print("si entroooo")
        me.send_rc_control(lr, movex, AAB, 0)
    if modoturn==1 and cronologia==1:
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        me.send_rc_control(20, 0, 0, 0) ##se mueve 60 cm
        time.sleep(3)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        me.send_rc_control(derecha, 0,0, 0)
        time.sleep(2)
        me.send_rc_control(0,20,0,0)
        time.sleep(5)
        me.rotate_clockwise(-90)
        time.sleep(1)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        me.send_rc_control(0,20,0,0)
        time.sleep(1)
        cronologia=2
        modoturn=0
    elif modoturn == 1 and cronologia == 2:
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        me.send_rc_control(20, 0, 0, 0) ##se mueve 60 cm
        time.sleep(2.2)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        me.send_rc_control(0, 0,abajo, 0)
        time.sleep(5)
        me.send_rc_control(0, 20, 0, 0)
        time.sleep(4)
        me.rotate_clockwise(-90)
        time.sleep(1)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        cronologia = 3
        modoturn = 0
    elif modoturn == 1 and cronologia == 3:
        me.send_rc_control(0,0,0,0)
        time.sleep(2)
        me.send_rc_control(20, 0, 0, 0) #se mueve 60 cm
        time.sleep(3)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        me.send_rc_control(izquierda,0,0,0)
        time.sleep(2)
        me.send_rc_control(0, 20, 0, 0)
        time.sleep(6)
        me.rotate_clockwise(-90)
        time.sleep(1)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        cronologia = 4
        modoturn = 0
    elif modoturn == 1 and cronologia == 4:
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        me.send_rc_control(20, 0, 0, 0) ##se mueve 60 cm
        time.sleep(2.2)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(2)
        me.send_rc_control(0, 0,arriba, 0)
        time.sleep(3)
        me.send_rc_control(0, 20, 0, 0)
        time.sleep(8)
        me.rotate_clockwise(-90)
        time.sleep(1)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        cronologia = 1
        modoturn = 0
        me.land()
    print(me)
while True:
    frame_read = me.get_frame_read()
    myFrame = frame_read.frame
    img = cv2.resize(myFrame, (width, height))
    if Start==0:
        me.send_rc_control(0, 0, 0, 0)
        me.takeoff()
        time.sleep(2)
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        me.send_rc_control(0, 0, 20, 0)  # Comando para descender
        time.sleep(8)  # Ajusta el tiempo de espera según la altura
        me.send_rc_control(0, 0, 0, 0)
        time.sleep(1)
        Start = 1
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
