import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import os
import sys


def run_recognition():
    """Main function to run gesture recognition - can be called from other scripts"""

    # Check if model files exist
    if not os.path.exists('model/gesture_model.pkl') or not os.path.exists('model/label_encoder.pkl'):
        print("Error: Model files not found. Please train the model first.")
        return False

    try:
        # Load trained model
        model = joblib.load('model/gesture_model.pkl')
        le = joblib.load('model/label_encoder.pkl')
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils

    def get_landmarks(image, results):
        """Extract hand landmarks in same format as training"""
        if results.multi_hand_landmarks:
            landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
            return np.array(landmarks).reshape(1, -1)
        return None

    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return False

    gesture_history = deque(maxlen=5)  # For stable prediction
    last_gesture = None
    frame_count = 0

    print("Starting gesture recognition. Press 'q' to quit.")

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture frame. Exiting...")
                break

            # Process image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Get landmarks and predict
            landmarks = get_landmarks(image, results)
            if landmarks is not None:
                try:
                    # Predict
                    proba = model.predict_proba(landmarks)[0]
                    top_idx = np.argmax(proba)
                    gesture = le.inverse_transform([top_idx])[0]
                    confidence = proba[top_idx]

                    # Only consider confident predictions
                    if confidence > 0.6:
                        gesture_history.append((gesture, confidence))

                    # Get most common gesture from history
                    if gesture_history:
                        # Count occurrences of each gesture
                        gesture_counts = {}
                        for g, c in gesture_history:
                            if g in gesture_counts:
                                gesture_counts[g] += 1
                            else:
                                gesture_counts[g] = 1

                        # Get gesture with highest count and confidence
                        stable_gesture = max(gesture_counts.items(), key=lambda x: x[1])
                        gesture = stable_gesture[0]

                        # Calculate average confidence for this gesture
                        confidences = [c for g, c in gesture_history if g == gesture]
                        confidence = np.mean(confidences) if confidences else 0

                    # Only update if gesture changed or confidence is high
                    if gesture != last_gesture or confidence > 0.8:
                        last_gesture = gesture

                        # Draw landmarks and prediction
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                        # Display prediction with confidence
                        cv2.putText(image, f"Gesture: {gesture}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        cv2.putText(image, f"Confidence: {confidence:.2%}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                        print(f"Detected: {gesture} ({confidence:.2%})")

                except Exception as e:
                    print(f"Prediction error: {e}")
            else:
                # No hand detected
                cv2.putText(image, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display instructions
            cv2.putText(image, "Press 'q' to quit", (10, image.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.imshow("Gesture Recognition", image)

            # Exit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recognition stopped by user.")
                break

    except Exception as e:
        print(f"Error during recognition: {e}")
        return False

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Gesture recognition stopped.")
        return True


def main():
    """Main function that can be called from command line"""
    run_recognition()


if __name__ == "__main__":
    main()