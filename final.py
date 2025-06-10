import cv2
import numpy as np
import mediapipe as mp
import time
from tensorflow.keras.models import load_model
import pyttsx3
import threading
engine = pyttsx3.init()
engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

from suggestion import suggestion


def speak(text):
    def run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=run, daemon=True).start()


model = load_model("model.h5")
time_start =time.time()

cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,800)
pred = ''
myhand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = myhand.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
class_labels = []
for i in range(65,65+26):

    if ( chr(i) != 'J' and chr(i) !='Z'):
        class_labels.append(chr(i))


ptime = 0
no_hand_time = None
pred=''
while True:

    string = ''
    ret, frame = cap.read()
    if not ret:
        break

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime


    frame = cv2.flip(frame, 1)  
    cv2.putText(frame, f'FPS: {int(fps)}', (20-4, 50-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0, 0), 3)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    cv2.putText(frame, f'prediction = {pred}', (100-90, 200-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            landmark_array = []
            for lm in hand_landmarks.landmark:
                x = int(lm.x*w)
                y=int(lm.y*h)
                landmark_array.append((x,y))

            x_coords = [pt[0] for pt in landmark_array]
            y_coords = [pt[1] for pt in landmark_array]
            x_min, x_max = max(min(x_coords) - 30, 0), min(max(x_coords) + 30, w)
            y_min, y_max = max(min(y_coords) - 30, 0), min(max(y_coords) + 30, h)

            hand_img = frame[y_min:y_max, x_min:x_max]

            hand_img = cv2.resize(hand_img, (224, 224))
            hand_img = hand_img / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)


            prediction = model.predict(hand_img)


            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            string = class_labels[class_index]
            label = f"{class_labels[class_index]} ({confidence:.2f})"
            time_new = time.time()
            if confidence>0.4 and (time.time()-time_start)>=5.0:
                pred += class_labels[class_index]
                time_start=time.time()
  

            cv2.putText(frame, f'prediction = {pred}', (100-90, 200-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            print(pred)
            words = pred.strip().split()
            if words:
                last_word = words[-1].lower()
                if last_word in suggestion:
                    cv2.putText(frame, f'suggestion = {suggestion[last_word]}', (100-90, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)



            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
            cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

            mp_drawing.draw_landmarks(frame, hand_landmarks, myhand.HAND_CONNECTIONS)

    else:
        if no_hand_time is None:
            no_hand_time = time.time()
        elif time.time() - no_hand_time >= 5:
            if pred:
                words = pred.strip().split()
                if words:
                    last_word = words[-1].lower()
                    if last_word in suggestion:
                        words[-1] = suggestion[last_word]
                        pred = ' '.join(words) + ' '
                        cv2.putText(frame, f'prediction = {pred}', (100-90, 200-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        speak(suggestion[last_word])


            no_hand_time = None


    cv2.imshow("Sign Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()




