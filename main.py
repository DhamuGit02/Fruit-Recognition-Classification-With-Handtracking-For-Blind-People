from HandDetector import HandDetector
import cv2


def fruit_detector(frame):
    return detection_model.detect(frame, confThreshold=0.7)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
hd = HandDetector(max_hands=1)
while cap.isOpened():
    _, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    frame, landmarks = hd.detectHands(frame, draw=True, )
    label, conf, bbox = fruit_detector(frame)
    # print(None if len(bbox) == 0 else bbox[0])
    frame, direction = hd.withinRegionAndHandNavigator(frame, landmarks[0], landmarks[5], landmarks[17], [0, 0, 0, 0] if len(bbox) == 0 else bbox[0], draw=True)
    print(direction)
    cv2.imshow('Webcame', frame)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()
