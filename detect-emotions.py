import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# ✅ Load face detection model
face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(face_cascade_path)

if face_classifier.empty():
    print("Error: Haar cascade model not found! Check OpenCV installation.")
    exit()

# ✅ Load emotion detection model
emotion_model = load_model("emotion_detection_model_100epochs.h5")

# Emotion labels
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ✅ Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale (faster face detection)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]  # Extract face region
        roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to match model input size

        # Normalize & preprocess image for the model
        roi = roi_gray.astype("float") / 255.0  # Scale pixel values
        roi = img_to_array(roi)  # Convert to array
        roi = np.expand_dims(roi, axis=0)  # Reshape for model input (1, 48, 48, 1)

        # ✅ Predict emotion
        preds = emotion_model.predict(roi)[0]
        label = class_labels[np.argmax(preds)]  # Get highest probability class

        # ✅ Draw rectangle around face & label emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # ✅ Show real-time video
    cv2.imshow("Real-Time Emotion Detector", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
