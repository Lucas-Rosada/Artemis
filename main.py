import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import mediapipe as mp

model = tf.keras.models.load_model('C:/Users/Lucas/Desktop/sinais/keras_model.h5')  

# Inicializar o MediaPipe (JANELA DO ZOOM)
mp_hands = mp.solutions.handsa
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sign_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'I', 6: 'L',
    7: 'M', 8: 'N', 9: 'O', 10: 'R', 11: 'S', 12: 'U', 
    13: 'V', 14: 'W'
}

#BUFF PRECIZAOOOAO
buffer_size = 5
prediction_buffer = deque(maxlen=buffer_size)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao capturar o vídeo.")
        break

    #PROCESSA TUA MÃO
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # DESENHA O ESQUELETO NA MÃO
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # PEGA AS COORDENADAS DA MÃO E DESENHA
            h, w, c = frame.shape
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * w
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * h
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * w
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * h

            # PEGA A REGIÃO ONDE A MÃO ESTA E DA ZOOM EXTRAINDO
            hand_img = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

            # EXIBE A MÃO EM UMA JANELA SEPARADA
            if hand_img.size > 0:
                cv2.imshow('Zoom', cv2.resize(hand_img, (250, 250)))

            # CIRCULA A MÃO
            cx, cy = int((x_min + x_max) // 2), int((y_min + y_max) // 2)
            radius = int(max(x_max - x_min, y_max - y_min) // 2)
            cv2.circle(frame, (cx, cy), radius, (0, 255, 0), 2)

            # PREPROCESSA A IMAGEM DE DETECÇÃP
            img = cv2.resize(frame, (224, 224))  
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
            img = np.expand_dims(img, axis=0) / 255.0  

            # FAZZ A PREVISÃO
            predictions = model.predict(img)
            sign_class = np.argmax(predictions)

            # ADD A PREVISAO AO BUFF
            prediction_buffer.append(sign_class)
            sign_class = max(set(prediction_buffer), key=prediction_buffer.count)

            # PEGA O SINAL Q CORRESPONDE
            sign = sign_dict.get(sign_class, "Desconhecido")

            # EXIBE A PREVISÃO
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.putText(frame, f'Sinal: {sign}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            print(f'Detecção: {sign}')

    cv2.imshow('Artemis', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()