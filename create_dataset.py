import cv2
import mediapipe as mp
import time
import os


label = 'Delete'
save_path = f'data_more(1)/{label}/'
os.makedirs(save_path, exist_ok=True)

roisize = 224
limit = 500

myhands = mp.solutions.hands
hands = myhands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

start_time = time.time()
while time.time() - start_time < 5:
    ret, frame = cap.read()
    frame=cv2.flip(frame,1)
    cv2.putText(frame, "Starting in 5 seconds...", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow(" Capture", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

count = 0
while count < limit:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            xcoordinate = []
            ycordinate=[]
            for lm in handLms.landmark:
                xcoordinate.append(lm.x)
            for lm in handLms.landmark:
                ycordinate.append(lm.y)

            cx = int(sum(xcoordinate) / len(xcoordinate) * w)
            cy = int(sum(ycordinate) / len(ycordinate) * h)

            top_left = (max(cx - roisize//2, 0), max(cy - roisize//2, 0))
            bottom_right = (min(cx + roisize//2, w), min(cy + roisize//2, h))
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            roi = frame[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

            if roi.shape[0] == roisize and roi.shape[1] == roisize:
                filename = f"{save_path}/{count}.jpg"
                cv2.imwrite(filename, roi)
                count += 1

            mp_draw.draw_landmarks(frame, handLms, myhands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Images Captured: {count}/{limit}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow(" Capture", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
