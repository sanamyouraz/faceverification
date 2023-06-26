import cv2
import face_recognition


capture = cv2.VideoCapture(0)  # Open the camera
while True:
    ret, frame = capture.read()

    # Detect faces in the frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if len(face_encodings) > 0:
        # Select the first detected face
        face_encoding = face_encodings[0]

        # Compare the face encoding with the reference encoding
        matches = face_recognition.compare_faces([reference_encoding], face_encoding)
        is_verified = all(matches)

        # Draw a rectangle around the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Display verification result
        text = "Verified" if is_verified else "Not Verified"
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Face Verification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
