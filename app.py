import streamlit as st
import subprocess
import os
import sys
from utils import *

# Page configuration
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="👋",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2563eb;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #2563eb;
        margin: 10px 0;
    }
    .stButton>button {
        width: 100%;
        margin: 10px 0;
        background-color: #2563eb;
        color: white;
    }
    .success-box {
        background-color: #d1fae5;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #10b981;
        margin: 10px 0;
    }
    .warning-box {
        background-color: #fef3c7;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #f59e0b;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.markdown('<h1 class="main-header">👋 Hand Gesture Recognition System</h1>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a page:",
                            ["Dashboard", "Collect Data", "Train Model", "Real-time Recognition"])

    # Dashboard page
    if page == "Dashboard":
        show_dashboard()

    # Collect Data page
    elif page == "Collect Data":
        show_collect_data()

    # Train Model page
    elif page == "Train Model":
        show_train_model()

    # Recognition page
    elif page == "Real-time Recognition":
        show_recognition()


def show_dashboard():
    st.header("📊 Dashboard Overview")

    # Create columns for metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        gesture_count = len(get_available_gestures())
        st.markdown(f"""
        <div class="metric-card">
            <h3>Gestures Collected</h3>
            <h2>{gesture_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        total_samples = get_total_samples()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Samples</h3>
            <h2>{total_samples}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        model_status = "✅ Trained" if model_exists() else "❌ Not Trained"
        st.markdown(f"""
        <div class="metric-card">
            <h3>Model Status</h3>
            <h2>{model_status}</h2>
        </div>
        """, unsafe_allow_html=True)

    # Show gesture statistics
    st.subheader("📋 Gesture Statistics")
    stats = get_gesture_stats()

    if stats:
        for gesture, count in stats.items():
            st.write(f"**{gesture}**: {count} samples")
    else:
        st.info("No gestures collected yet. Go to 'Collect Data' to get started!")

    # Quick actions
    st.subheader("⚡ Quick Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Check for Updates"):
            st.rerun()

    with col2:
        if st.button("📁 Open Data Folder"):
            os.startfile("data") if os.path.exists("data") else st.warning("Data folder doesn't exist")


def show_collect_data():
    st.header("🎥 Collect Gesture Data")

    col1, col2 = st.columns([2, 1])

    with col1:
        gesture_name = st.text_input("Enter Gesture Name:",
                                     placeholder="e.g., thumbs_up, victory, fist")

        target_samples = st.slider("Target Number of Samples:",
                                   min_value=10, max_value=100, value=30)

        if st.button("🎬 Start Data Collection", type="primary"):
            if gesture_name and gesture_name.strip():
                # Create data directory if it doesn't exist
                if not os.path.exists('data'):
                    os.makedirs('data')

                st.info(f"Starting data collection for '{gesture_name}'...")
                st.warning("⚠️ A new window will open. Press 's' to save samples and 'q' to quit.")

                # Run the collection script
                try:
                    process = subprocess.Popen([sys.executable, 'collect_data.py', gesture_name])
                    st.success("Data collection started successfully!")
                except Exception as e:
                    st.error(f"Error starting collection: {str(e)}")
            else:
                st.error("Please enter a valid gesture name")

    with col2:
        st.info("""
        **📝 Instructions:**
        1. Enter a descriptive gesture name
        2. Set target number of samples (20-30 recommended)
        3. Click 'Start Data Collection'
        4. Show your hand to the camera
        5. Press 's' to save each sample
        6. Press 'q' when finished

        **💡 Tips:**
        - Use consistent lighting
        - Vary hand position slightly
        - Collect samples from different angles
        """)

    # Show existing gestures
    st.subheader("📁 Existing Gestures")
    gestures = get_available_gestures()
    if gestures:
        for gesture in gestures:
            count = get_sample_count(gesture)
            st.write(f"**{gesture}**: {count} samples")
    else:
        st.info("No gestures collected yet.")


def show_train_model():
    st.header("🤖 Train Machine Learning Model")

    # Check if we have enough data
    gestures = get_available_gestures()
    total_samples = get_total_samples()

    if len(gestures) < 2:
        st.warning("""
        **⚠️ Not enough data!**
        You need at least 2 different gestures to train a model.
        Please collect more data first.
        """)
        return

    if total_samples < 20:
        st.warning(f"""
        **⚠️ Low sample count!**
        You only have {total_samples} total samples. 
        For better accuracy, collect at least 20 samples per gesture.
        """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Training Details")
        st.write(f"**Gestures available:** {len(gestures)}")
        st.write(f"**Total samples:** {total_samples}")

        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("🔄 Training model... This may take a few minutes"):
                try:
                    # Run training process
                    result = subprocess.run([sys.executable, 'train_model.py'],
                                            capture_output=True, text=True, timeout=120)

                    if result.returncode == 0:
                        st.success("✅ Model trained successfully!")
                        st.text_area("Training Output", result.stdout, height=200)
                    else:
                        st.error("❌ Training failed!")
                        st.text_area("Error Output", result.stderr, height=200)

                except subprocess.TimeoutExpired:
                    st.error("⏰ Training timed out. Please try again.")
                except Exception as e:
                    st.error(f"❌ Error during training: {str(e)}")

    with col2:
        st.info("""
        **🔧 Technical Details:**
        - **Algorithm:** Support Vector Machine (SVM)
        - **Input Features:** 63 values (21 landmarks × 3 coordinates)
        - **Output:** Gesture classification

        **📊 Expected Performance:**
        - 85-95% accuracy with good data
        - Real-time prediction (30+ FPS)

        **✅ Requirements:**
        - Minimum 2 different gestures
        - Minimum 10 samples per gesture
        - Consistent lighting conditions
        """)


def show_recognition():
    st.header("🔍 Real-time Gesture Recognition")

    # Check if model exists
    if not model_exists():
        st.warning("""
        **⚠️ No trained model found!**
        Please train a model first in the 'Train Model' section.
        """)
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Live Recognition")

        if st.button("👀 Start Recognition", type="primary"):
            st.info("""
            **🎥 Recognition Starting...**
            - A new window will open with camera feed
            - Show your hand gestures to the camera
            - Press 'q' to close the recognition window
            """)

            try:
                # Run recognition
                process = subprocess.Popen([sys.executable, 'recognize.py'])
                st.success("✅ Recognition started successfully!")
            except Exception as e:
                st.error(f"❌ Error starting recognition: {str(e)}")

    with col2:
        st.info("""
        **🎯 How to Use:**
        1. Click 'Start Recognition'
        2. Allow camera access if prompted
        3. Show hand gestures to the camera
        4. View real-time predictions
        5. Press 'q' to stop recognition

        **✨ Features:**
        - Real-time hand tracking
        - Instant gesture classification
        - Confidence score display
        - Smooth performance (30+ FPS)

        **📋 Supported Gestures:**
        """ + "\n".join([f"- {gesture}" for gesture in get_available_gestures()]))


if __name__ == "__main__":
    main()