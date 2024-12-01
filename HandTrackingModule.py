import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)


with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Tidak dapat mengakses kamera.")
            break

        
        image = cv2.flip(image, 1)

        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

       
        results = hands.process(image_rgb)

       
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]

                
                h, w, _ = image.shape
                thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
                index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)

                
                cv2.circle(image, (thumb_x, thumb_y), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(image, (index_x, index_y), 10, (255, 0, 0), cv2.FILLED)

                
                cv2.line(image, (thumb_x, thumb_y), (index_x, index_y), (0, 255, 0), 3)

                
                distance = np.sqrt((index_x - thumb_x) ** 2 + (index_y - thumb_y) ** 2)

               
                volume = np.interp(distance, [30, 200], [0, 100])

                
                cv2.putText(image, f'Volume: {int(volume)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        
        cv2.imshow('Hand Volume Control', image)

        
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Lepaskan capture dan tutup jendela
cap.release()
cv2.destroyAllWindows()