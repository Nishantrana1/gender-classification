import cv2
from deepface import DeepFace
import mediapipe as mp

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

# Initialize MediaPipe face detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

# Process every 5th frame to improve FPS
frame_skip = 5
frame_count = 0

# Total male and female counts in the current frame
male_count = 0
female_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    results = face_detection.process(rgb_frame)

    # Reset counts every frame
    male_count = 0
    female_count = 0

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            ih, iw, _ = frame.shape
            x = int(bbox.xmin * iw)
            y = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)

            face_crop = frame[y:y + h, x:x + w]

            try:
                # Analyze face for gender
                analysis = DeepFace.analyze(face_crop, actions=['gender'], enforce_detection=False, detector_backend='opencv')

                if isinstance(analysis, dict):
                    analysis = [analysis]  # Ensure it's a list

                gender = 'Unknown'
                for person in analysis:
                    if 'gender' in person:
                        if person['gender']['Man'] > 70:
                            gender = 'Man'
                            male_count += 1
                        elif person['gender']['Woman'] > 70:
                            gender = 'Woman'
                            female_count += 1

                # Draw rectangle and label
                color = (255, 0, 0) if gender == "Man" else (255, 20, 147)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

            except Exception as e:
                print(f"⚠️ DeepFace Error: {e}")
                cv2.putText(frame, "Face detected, gender unknown", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display total male & female counts
    cv2.putText(frame, f"Male: {male_count}", (frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, f"Female: {female_count}", (frame.shape[1] - 150, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 20, 147), 2)

    # Show frame
    cv2.imshow("Gender Detection with Count", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
