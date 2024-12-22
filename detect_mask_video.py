import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Load pre-trained model and cascade
model = load_model("mask_detector.keras")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_and_predict_mask(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_list = []
    preds = []

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = np.array(face, dtype="float32") / 255.0
        face = np.expand_dims(face, axis=0)
        faces_list.append((x, y, w, h))
        preds.append(model.predict(face)[0])

    return faces_list, preds

# Start video feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces, preds = detect_and_predict_mask(frame)

    for (box, pred) in zip(faces, preds):
        (x, y, w, h) = box
        (mask, without_mask) = pred
        label = "Mask" if mask > without_mask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    cv2.imshow("Face Mask Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()