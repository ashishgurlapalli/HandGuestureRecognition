import cv2
import mediapipe as mp
import os
import numpy as np
from collections import deque
import argparse  # Added for command-line arguments
import sys

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


def get_landmarks(image, results):
    """Extract hand landmarks in a consistent format"""
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    return None


def collect_gesture_data(gesture_name):
    """Main function to collect gesture data - can be called from other scripts"""
    # Create data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')

    cap = cv2.VideoCapture(0)
    samples_collected = 0
    buffer = deque(maxlen=5)  # For stable gesture collection

    # Create directory for this gesture
    gesture_dir = os.path.join('data', gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    print(f"Collecting samples for '{gesture_name}'. Press 's' to save, 'q' to quit")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # Process image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get and display landmarks
            landmarks = get_landmarks(image, results)
            if landmarks:
                buffer.append(landmarks)
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Display instructions
            cv2.putText(image, f"Gesture: {gesture_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, f"Samples: {samples_collected}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Press 's' to save sample", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(image, "Press 'q' to quit", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(f"Collecting {gesture_name}", image)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('s') and buffer:
                # Save the most stable sample (average of buffer)
                avg_landmarks = np.mean(buffer, axis=0)
                np.save(os.path.join(gesture_dir, f'sample_{samples_collected}.npy'), avg_landmarks)
                samples_collected += 1
                print(f"Saved sample {samples_collected}")
                buffer.clear()
            elif key & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"Collected {samples_collected} samples for {gesture_name}")
        return samples_collected


def main():
    """Main function that handles command-line arguments"""
    parser = argparse.ArgumentParser(description='Collect hand gesture data')
    parser.add_argument('gesture_name', nargs='?', help='Name of the gesture to collect')
    args = parser.parse_args()

    if args.gesture_name:
        gesture_name = args.gesture_name
    else:
        # Fallback to input if no command-line argument provided
        gesture_name = input("Enter gesture name to collect (e.g., 'thumbs_up'): ")

    collect_gesture_data(gesture_name)


if __name__ == "__main__":
    main()