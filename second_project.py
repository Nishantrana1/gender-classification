import cv2
from deepface import DeepFace

# Path to the video file
video_path = r"D:\Nisha\programing\python\yolo_second\video.mp4"

# Open video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Error: Could not open video file {video_path}")
    exit()

frame_skip = 10  # Process every 10th frame (change if needed)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Video finished

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip frames to save processing time

    try:
        # Detect gender for all faces in the frame
        analysis = DeepFace.analyze(frame, 
                                    actions=['gender'], 
                                    enforce_detection=False, 
                                    detector_backend='retinaface')  # RetinaFace is faster

        # Loop through all detected faces and draw gender on each face
        for person in analysis:
            gender = person['dominant_gender']
            region = person['region']  # Face location

            x, y, w, h = region['x'], region['y'], region['w'], region['h']

            # Choose color for gender label
            color = (255, 0, 0) if gender == "Man" else (255, 20, 147)

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw gender label
            cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    except Exception as e:
        print(f"⚠️ Detection Error: {e}")
        cv2.putText(frame, "No Face Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show frame with gender info
    cv2.imshow("Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
