import cv2
import mediapipe as mp
import pyautogui

# Initialize the hand tracking module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize the camera
cap = cv2.VideoCapture(0)

# Get screen width and height
screen_width, screen_height = pyautogui.size()

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            for landmark in landmarks.landmark:
                # Extract and print the x, y, z coordinates of each hand landmark
                x, y, z = landmark.x, landmark.y, landmark.z
                print(f"Landmark: X: {x}, Y: {y}, Z: {z}")

            # Get the coordinates of the index finger tip (landmark 8)
            index_finger_tip = landmarks.landmark[8]
            index_finger_x = int(index_finger_tip.x * screen_width)
            index_finger_y = int(index_finger_tip.y * screen_height)

            # Move the cursor to the position of the index finger tip
            pyautogui.moveTo(index_finger_x, index_finger_y)

    # Display the frame with hand landmarks
    cv2.imshow('Hand Tracking', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()

