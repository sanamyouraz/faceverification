import dlib
import cv2
import numpy as np

# Load the pre-trained face detection model (MMOD)
detector = dlib.cnn_face_detection_model_v1("Model/mmod_human_face_detector.dat")

# Load the pre-trained shape predictor model
shape_predictor = dlib.shape_predictor("Model/shape_predictor_68_face_landmarks.dat")

# Load the pre-trained face recognition model (ResNet)
face_recognizer = dlib.face_recognition_model_v1("Model/dlib_face_recognition_resnet_model_v1.dat")

# Load the reference face image
reference_image = cv2.imread("Images/sanam.jpg")

# Detect faces in the reference image
reference_faces = detector(reference_image, 1)

if len(reference_faces) > 0:
    # Get the first detected face
    reference_face = reference_faces[0].rect

    # Perform face landmark detection on the reference face
    reference_landmarks = shape_predictor(reference_image, reference_face)

    # Calculate the face descriptor for the reference face
    reference_features = face_recognizer.compute_face_descriptor(reference_image, reference_landmarks)

    # Open the camera
    camera = cv2.VideoCapture(0)

    while True:
        # Capture a video frame from the camera
        ret, frame = camera.read()

        # Detect faces in the captured frame
        faces = detector(frame, 1)

        if len(faces) > 0:
            # Get the first detected face
            detected_face = faces[0].rect

            # Perform face landmark detection on the detected face
            detected_landmarks = shape_predictor(frame, detected_face)

            # Calculate the face descriptor for the detected face
            detected_features = face_recognizer.compute_face_descriptor(frame, detected_landmarks)

            # Compare the features of the detected face with the reference face
            distance = np.linalg.norm(np.array(reference_features) - np.array(detected_features))

            # Set a threshold for similarity/distance
            threshold = 0.6

            if distance < threshold:
                cv2.putText(frame, "Face verified successfully", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Face verification failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame in a window
        cv2.imshow("Camera", frame)

        # Wait for the 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    camera.release()
    cv2.destroyAllWindows()
else:
    print("No faces found in the reference image.")
