import dlib
import cv2
import numpy as np
import os

# Load the pre-trained face detection model (MMOD)
detector = dlib.cnn_face_detection_model_v1("Model/mmod_human_face_detector.dat")

# Load the pre-trained shape predictor model
shape_predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")

# Load the pre-trained face recognition model (ResNet)
face_recognizer = dlib.face_recognition_model_v1("Model/dlib_face_recognition_resnet_model_v1.dat")

# Directory path of reference face images
reference_dir = "Images"

# Load reference face images and compute face descriptors
reference_faces = {}
for filename in os.listdir(reference_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(reference_dir, filename)
        reference_image = cv2.imread(image_path)
        reference_faces[filename] = {}
        reference_faces[filename]["image"] = reference_image
        reference_faces[filename]["faces"] = detector(reference_image, 1)
        if len(reference_faces[filename]["faces"]) > 0:
            reference_face = reference_faces[filename]["faces"][0].rect
            reference_landmarks = shape_predictor(reference_image, reference_face)
            reference_features = face_recognizer.compute_face_descriptor(reference_image, reference_landmarks)
            reference_faces[filename]["features"] = reference_features

# Open the camera
camera = cv2.VideoCapture(0)

while True:
    # Capture a video frame from the camera
    ret, frame = camera.read()

    # Detect faces in the captured frame
    faces = detector(frame, 1)

    if len(faces) > 0:
        for face in faces:
            detected_face = face.rect
            detected_landmarks = shape_predictor(frame, detected_face)
            detected_features = face_recognizer.compute_face_descriptor(frame, detected_landmarks)

            verified = False
            for ref_name, ref_data in reference_faces.items():
                distance = np.linalg.norm(np.array(ref_data["features"]) - np.array(detected_features))
                threshold = 0.6
                if distance < threshold:
                    verified = True
                    cv2.putText(frame, "Face verified: " + ref_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    break

            if not verified:
                cv2.putText(frame, "Face verification failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame in a window
    cv2.imshow("Camera", frame)

    # Wait for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
