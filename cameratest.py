import cv2

# Open the camera
camera = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = camera.read()

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Display the frame in a window
    cv2.imshow("Camera", frame)

    # Wait for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
camera.release()
cv2.destroyAllWindows()
