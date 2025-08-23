import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
from collections import defaultdict
import sys


def load_data(data_dir='data'):
    """Load collected gesture data with error handling"""
    X = []
    y = []

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return np.array(X), np.array(y)

    # Check if directory is empty
    if not os.listdir(data_dir):
        print("Error: Data directory is empty.")
        return np.array(X), np.array(y)

    gesture_count = 0
    total_samples = 0

    for gesture_name in os.listdir(data_dir):
        gesture_dir = os.path.join(data_dir, gesture_name)
        if os.path.isdir(gesture_dir):
            gesture_samples = 0
            for sample_file in os.listdir(gesture_dir):
                if sample_file.endswith('.npy'):
                    try:
                        sample_path = os.path.join(gesture_dir, sample_file)
                        data = np.load(sample_path)

                        # Check if data has correct shape (63 values)
                        if data.shape[0] == 63:  # 21 landmarks × 3 coordinates
                            X.append(data)
                            y.append(gesture_name)
                            gesture_samples += 1
                            total_samples += 1
                        else:
                            print(f"Warning: Invalid data shape in {sample_path}. Expected 63, got {data.shape[0]}")

                    except Exception as e:
                        print(f"Error loading {sample_path}: {e}")

            if gesture_samples > 0:
                gesture_count += 1
                print(f"Loaded {gesture_samples} samples for gesture '{gesture_name}'")
            else:
                print(f"Warning: No valid samples found for gesture '{gesture_name}'")

    print(f"\nTotal: {gesture_count} gestures, {total_samples} samples loaded")
    return np.array(X), np.array(y)


def train_model():
    """Train the gesture recognition model and return status"""
    try:
        # Load data
        print("Loading training data...")
        X, y = load_data()

        if len(X) == 0:
            error_msg = "No training data found. Please collect samples first."
            print(error_msg)
            return False, error_msg

        # Check if we have enough gestures
        unique_gestures = np.unique(y)
        if len(unique_gestures) < 2:
            error_msg = f"Need at least 2 different gestures. Found only {len(unique_gestures)}: {unique_gestures}"
            print(error_msg)
            return False, error_msg

        # Check if we have enough samples per gesture
        gesture_counts = {}
        for gesture in unique_gestures:
            count = np.sum(y == gesture)
            gesture_counts[gesture] = count
            if count < 5:
                print(f"Warning: Gesture '{gesture}' has only {count} samples (recommended: 10+)")

        print(f"\nTraining with {len(unique_gestures)} gestures: {list(unique_gestures)}")
        for gesture, count in gesture_counts.items():
            print(f"  {gesture}: {count} samples")

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42,
                                                            stratify=y_encoded)

        # Train model
        print("\nTraining SVM model...")
        model = SVC(kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, y_train)

        # Evaluate
        train_acc = model.score(X_train, y_train)
        test_acc = model.score(X_test, y_test)

        print(f"Training accuracy: {train_acc:.3f}")
        print(f"Testing accuracy: {test_acc:.3f}")

        # Additional metrics
        from sklearn.metrics import classification_report
        y_pred = model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        # Save model and label encoder
        if not os.path.exists('model'):
            os.makedirs('model')

        joblib.dump(model, 'model/gesture_model.pkl')
        joblib.dump(le, 'model/label_encoder.pkl')

        success_msg = f"Model trained successfully! Accuracy: {test_acc:.3f}"
        print(success_msg)

        return True, success_msg

    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        print(error_msg)
        return False, error_msg


def main():
    """Main function that can be called from command line"""
    success, message = train_model()
    if not success:
        # Use ASCII characters instead of Unicode for Windows compatibility
        print(f"\nERROR: {message}")
        sys.exit(1)
    else:
        # Use ASCII characters instead of Unicode for Windows compatibility
        print(f"\nSUCCESS: {message}")


if __name__ == "__main__":
    main()