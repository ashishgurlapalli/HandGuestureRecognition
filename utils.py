import os
import numpy as np

def get_available_gestures():
    """Get list of collected gestures from data folder"""
    if not os.path.exists('data'):
        return []
    gestures = []
    for item in os.listdir('data'):
        if os.path.isdir(os.path.join('data', item)):
            gestures.append(item)
    return gestures

def get_sample_count(gesture_name):
    """Count samples for a specific gesture"""
    gesture_path = os.path.join('data', gesture_name)
    if os.path.exists(gesture_path):
        samples = [f for f in os.listdir(gesture_path) if f.endswith('.npy')]
        return len(samples)
    return 0

def get_total_samples():
    """Get total number of samples across all gestures"""
    total = 0
    for gesture in get_available_gestures():
        total += get_sample_count(gesture)
    return total

def model_exists():
    """Check if trained model exists"""
    return os.path.exists('model/gesture_model.pkl') and os.path.exists('model/label_encoder.pkl')

def get_gesture_stats():
    """Get statistics about collected data"""
    gestures = get_available_gestures()
    stats = {}
    for gesture in gestures:
        stats[gesture] = get_sample_count(gesture)
    return stats