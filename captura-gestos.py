import cv2
import os

save_dir = "data/A"
os.makedirs(save_dir, exist_ok=True)

count = 0

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    roi = frame[100:300, 100:300]
    cv2.imshow('ROI', roi)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        filename = os.path.join(save_dir, f"img_{count}.jpg")
        cv2.imwrite(filename, roi)
        print(f"Imagem salva: {filename}")
        count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()