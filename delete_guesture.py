import os
import shutil
import argparse


def delete_gesture(gesture_name):
    gesture_path = os.path.join('data', gesture_name)

    if os.path.exists(gesture_path):
        shutil.rmtree(gesture_path)
        print(f"Deleted gesture: {gesture_name}")
    else:
        print(f"Gesture '{gesture_name}' not found")


def list_gestures():
    if os.path.exists('data'):
        gestures = [d for d in os.listdir('data') if os.path.isdir(os.path.join('data', d))]
        if gestures:
            print("Available gestures:")
            for gesture in gestures:
                count = len([f for f in os.listdir(os.path.join('data', gesture)) if f.endswith('.npy')])
                print(f"  {gesture} ({count} samples)")
        else:
            print("No gestures found")
    else:
        print("Data directory does not exist")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Manage gesture data')
    parser.add_argument('--list', action='store_true', help='List all gestures')
    parser.add_argument('--delete', type=str, help='Name of gesture to delete')

    args = parser.parse_args()

    if args.list:
        list_gestures()
    elif args.delete:
        delete_gesture(args.delete)
    else:
        print("Use --list to see gestures or --delete <name> to delete one")