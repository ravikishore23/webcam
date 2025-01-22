import cv2
import numpy as np
import mediapipe as mp

# Set width and height of the output screen
frameWidth = 640
frameHeight = 480

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

# Print default properties for debugging
print(f"Default frame width: {cap.get(3)}")
print(f"Default frame height: {cap.get(4)}")

# Optionally set frame width and height
cap.set(3, frameWidth)  # Property ID 3 is frame width
cap.set(4, frameHeight)  # Property ID 4 is frame height

# Set brightness, Property ID 10
cap.set(10, 150)

# Initialize Mediapipe Hand Tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Object color values in HSV
myColors = [[5, 107, 0, 19, 255, 255], 
            [133, 56, 0, 159, 156, 255], 
            [57, 76, 0, 100, 255, 255], 
            [90, 48, 0, 118, 255, 255]]

# Color values in BGR for painting
myColorValues = [[51, 153, 255], 
                 [255, 0, 255], 
                 [0, 255, 0], 
                 [255, 0, 0]]

# List to store points [x, y, colorId]
myPoints = []

# Function to detect and find the object's color
def findColor(img, myColors, myColorValues):
    global imgResult
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    count = 0
    newPoints = []
    
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        x, y = getContours(mask)
        
        if x != 0 and y != 0:
            newPoints.append([x, y, count])
        
        # Draw detected object on the result image
        cv2.circle(imgResult, (x, y), 15, myColorValues[count], cv2.FILLED)
        count += 1
    
    return newPoints

# Function to get contours for accurate detection
def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x, y, w, h = 0, 0, 0, 0
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w // 2, y

# Function to draw points on the virtual canvas
def drawOnCanvas(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgResult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)

# Main loop to capture video and process it
while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture video.")
        break
    
    imgResult = img.copy()
    
    # Detect and draw hands
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(imgResult, handLms, mpHands.HAND_CONNECTIONS)
            
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Example: Mark the tip of the index finger (ID 8)
                if id == 8:
                    cv2.circle(imgResult, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
                    myPoints.append([cx, cy, 0])  # Using colorId 0 for drawing with the first color

    # Finding object colors
    newPoints = findColor(img, myColors, myColorValues)
    if len(newPoints) != 0:
        for newP in newPoints:
            myPoints.append(newP)
    
    if len(myPoints) != 0:
        drawOnCanvas(myPoints, myColorValues)
    
    # Display the result
    cv2.imshow("Result", imgResult)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
