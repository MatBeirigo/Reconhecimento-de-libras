import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = tf.keras.models.load_model('modelo_libras.h5')

# classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Space', 'Nothing', 'Del']

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    roi = frame[100:300, 100:300]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 2)

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, (64, 64))
    roi_normalized = roi_resized / 255.0
    roi_reshaped = np.reshape(roi_normalized, (1, 64, 64, 1))

    prediction = model.predict(roi_reshaped)
    print(prediction)
    predicted_class = classes[np.argmax(prediction)] 

    cv2.putText(frame, f'Predicao: {predicted_class}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow('Frame', frame)
    cv2.imshow('ROI', roi_gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()