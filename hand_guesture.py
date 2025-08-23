import cv2
import mediapipe as mp
import numpy as np
from collections import deque, Counter
import os

# Suppress unnecessary warnings and logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)


def detect_gesture(hand_landmarks):
    """Detects hand gesture based on landmark positions"""
    landmarks = hand_landmarks.landmark

    # Thumb (check if it's extended)
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    # Other fingers
    index_tip = landmarks[8]
    index_pip = landmarks[6]

    middle_tip = landmarks[12]
    middle_pip = landmarks[10]

    ring_tip = landmarks[16]
    ring_pip = landmarks[14]

    pinky_tip = landmarks[20]
    pinky_pip = landmarks[18]

    # Check finger states (extended or bent)
    thumb_extended = thumb_tip.y < thumb_ip.y
    index_extended = index_tip.y < index_pip.y
    middle_extended = middle_tip.y < middle_pip.y
    ring_extended = ring_tip.y < ring_pip.y
    pinky_extended = pinky_tip.y < pinky_pip.y

    # Detect gestures
    if (index_extended and middle_extended and
            not ring_extended and not pinky_extended and
            not thumb_extended):
        return "VICTORY"
    elif thumb_extended and not any([index_extended, middle_extended, ring_extended, pinky_extended]):
        return "THUMBS_UP"
    elif (index_extended and thumb_extended and
          not middle_extended and not ring_extended and not pinky_extended):
        return "POINTING"
    elif (not index_extended and not middle_extended and
          not ring_extended and not pinky_extended and not thumb_extended):
        return "FIST"
    elif (index_extended and middle_extended and
          ring_extended and pinky_extended and thumb_extended):
        return "OPEN_PALM"
    elif (thumb_tip.x > landmarks[5].x and thumb_tip.y < landmarks[5].y and
          index_tip.y < landmarks[5].y and not middle_extended and
          not ring_extended and not pinky_extended):
        return "OK_SIGN"
    else:
        return "UNKNOWN"


def count_fingers(hand_landmarks):
    """Counts number of extended fingers"""
    landmarks = hand_landmarks.landmark
    thumb_extended = landmarks[4].y < landmarks[3].y
    index_extended = landmarks[8].y < landmarks[7].y
    middle_extended = landmarks[12].y < landmarks[11].y
    ring_extended = landmarks[16].y < landmarks[15].y
    pinky_extended = landmarks[20].y < landmarks[19].y

    return sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])


def main():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # For gesture smoothing
    gesture_history = deque(maxlen=5)

    # Instructions to display
    instructions = [
        "Hand Gesture Detection",
        "Gestures:",
        "- Open Palm (all fingers extended)",
        "- Fist (all fingers closed)",
        "- Thumbs Up (only thumb extended)",
        "- Victory (index+middle fingers up)",
        "- Pointing (index finger extended)",
        "- OK (thumb+index making circle)",
        "",
        "Press 'q' to quit"
    ]

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Convert the BGR image to RGB and process it
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Detect gesture
                    gesture = detect_gesture(hand_landmarks)
                    gesture_history.append(gesture)

                    # Get stable gesture (most common in history)
                    if gesture_history:
                        stable_gesture = Counter(gesture_history).most_common(1)[0][0]

                    # Count fingers
                    finger_count = count_fingers(hand_landmarks)

                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get wrist position for text
                    wrist = hand_landmarks.landmark[0]
                    text_x = int(wrist.x * image.shape[1])
                    text_y = int(wrist.y * image.shape[0]) - 50

                    # Display gesture and finger count
                    cv2.putText(image, f"{stable_gesture}", (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f"Fingers: {finger_count}", (text_x, text_y + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Display instructions
            y_offset = 20
            for line in instructions:
                cv2.putText(image, line, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20

            # Display the image
            cv2.imshow('Hand Gesture Detection', image)

            # Exit when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Program stopped by user")

    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()