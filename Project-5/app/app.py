import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import random
import pandas as pd

st.set_page_config(
    page_title="Aerial Object Intelligence Dashboard",
    page_icon="🦅",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATASET_PATH = r"C:\Users\ishan\OneDrive\College\Project\dataset"
MODEL_PATH_KERAS = r"models/transfer_bird_drone.keras"
MODEL_PATH_H5 = r"models/transfer_bird_drone.h5"

@st.cache_resource
def load_final_model():
    for path in [MODEL_PATH_KERAS, MODEL_PATH_H5]:
        if os.path.exists(path):
            try:
                return tf.keras.models.load_model(path, compile=False)
            except Exception:
                continue
    return None

def get_stats():
    stats = {}
    folders = ['train', 'valid', 'test']
    classes = ['bird', 'drone']
    
    if not os.path.exists(DATASET_PATH):
        return None
        
    for folder in folders:
        p = os.path.join(DATASET_PATH, folder)
        if os.path.exists(p):
            b = len([f for f in os.listdir(os.path.join(p, 'bird')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            d = len([f for f in os.listdir(os.path.join(p, 'drone')) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            stats[folder] = {"Bird": b, "Drone": d, "Total": b + d}
    return stats

model = load_final_model()
stats = get_stats()

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2563/2563413.png", width=100)
    st.title("Project Control")
    st.markdown("---")
    st.markdown("### 👨‍💻 Developer Info")
    st.text("Capstone Candidate")
    st.markdown("### 🛠️ Tech Stack")
    st.code("Python\nTensorFlow\nMobileNetV2\nStreamlit", language="text")
    st.markdown("---")
    st.caption("© 2024 Aerial Object Classification Project")

st.markdown("""
    <div style="background-color:#1E1E1E;padding:20px;border-radius:10px;border-left:8px solid #FF4B4B;">
        <h1 style="color:white;margin:0;">Aerial Object Classification & Detection</h1>
        <p style="color:#BCBCBC;font-size:18px;">Deep Learning for Autonomous Airspace Security</p>
    </div>
    """, unsafe_allow_html=True)

tab_home, tab_data, tab_model, tab_demo, tab_usecases = st.tabs([
    "🏠 Overview", "📊 Dataset Analytics", "🧠 Model Intelligence", "🚀 Live Prediction", "💼 Business Scope"
])

with tab_home:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.header("Project Vision")
        st.write("""
        This project addresses the critical challenge of distinguishing between natural wildlife (birds) and 
        man-made aerial vehicles (drones) in complex environments. Utilizing Transfer Learning and Computer Vision, 
        we provide a real-time identification system for security and safety applications.
        """)
        
        st.subheader("Core Objectives")
        st.markdown("""
        - **High Accuracy Detection:** Achieving >95% accuracy in distinguishing silhouettes.
        - **Low Latency Inference:** Optimization for near real-time processing.
        - **Airspace Monitoring:** Assisting airports and sensitive zones in early threat detection.
        """)
    
    with col2:
        st.markdown("### Workflow Status")
        st.success("✅ Dataset Collection")
        st.success("✅ Preprocessing & Augmentation")
        st.success("✅ Transfer Learning Training")
        st.success("✅ Model Evaluation")
        st.warning("🔄 Real-world Deployment")

with tab_data:
    st.header("Dataset Architecture")
    
    if stats:
        m1, m2, m3 = st.columns(3)
        m1.metric("Training Samples", stats['train']['Total'], "Total")
        m2.metric("Validation Split", stats['valid']['Total'], "Available")
        m3.metric("Unseen Test Set", stats['test']['Total'], "Final Evaluation")

        st.markdown("### Class Balance Visualization")
        df_chart = pd.DataFrame({
            "Category": ["Bird", "Drone"],
            "Training": [stats['train']['Bird'], stats['train']['Drone']],
            "Validation": [stats['valid']['Bird'], stats['valid']['Drone']],
            "Test": [stats['test']['Bird'], stats['test']['Drone']]
        })
        st.bar_chart(df_chart.set_index("Category"))

        st.markdown("### Dataset Sample Gallery")
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        for i, col in enumerate([g_col1, g_col2, g_col3, g_col4]):
            target_class = 'bird' if i < 2 else 'drone'
            sample_dir = os.path.join(DATASET_PATH, 'train', target_class)
            if os.path.exists(sample_dir):
                files = [f for f in os.listdir(sample_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if files:
                    sample_file = random.choice(files)
                    col.image(os.path.join(sample_dir, sample_file), caption=f"Sample {target_class.capitalize()}", width=300)
    else:
        st.error(f"Dataset not found at: {DATASET_PATH}. Please verify the path.")

with tab_model:
    col_m1, col_m2 = st.columns(2)
    
    with col_m1:
        st.header("Neural Network Design")
        st.info("**Backbone:** MobileNetV2 (Pre-trained on ImageNet)")
        st.write("""
        MobileNetV2 was selected for its inverted residual structure and depthwise separable convolutions, 
        offering an optimal balance between accuracy and computational efficiency for aerial monitoring.
        """)
        
        st.subheader("Training Parameters")
        params_df = pd.DataFrame({
            "Parameter": ["Input Shape", "Batch Size", "Learning Rate", "Epochs", "Optimizer"],
            "Value": ["(224, 224, 3)", "32", "0.001 (with Decay)", "15", "Adam"]
        })
        st.table(params_df)

    with col_m2:
        st.header("Evaluation Metrics")
        st.metric("Final Test Accuracy", "96.6%", "Industry Grade")
        
        report_data = {
            "Class": ["Bird", "Drone", "Macro Average"],
            "Precision": ["0.96", "0.97", "0.96"],
            "Recall": ["0.98", "0.95", "0.96"],
            "F1-Score": ["0.97", "0.96", "0.96"]
        }
        st.dataframe(pd.DataFrame(report_data), hide_index=True, width=600)
        st.caption("Performance evaluated on 215 unseen high-resolution samples.")


with tab_demo:
    st.header("Inference Engine")
    
    source = st.radio("Select Prediction Source:", ["Random Dataset Sample", "Manual File Upload"], horizontal=True)

    if model is None:
        st.error("Model weights not found. Please verify the 'models' folder.")
    else:
        current_image = None
        
        if source == "Random Dataset Sample":
            if st.button("🎲 Fetch Random Test Image"):
                try:
                    t_dir = os.path.join(DATASET_PATH, 'test')
                    cls_choice = random.choice(['bird', 'drone'])
                    cls_dir = os.path.join(t_dir, cls_choice)
                    f_list = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    if f_list:
                        selected_file = random.choice(f_list)
                        current_image = Image.open(os.path.join(cls_dir, selected_file))
                        st.session_state['active_img'] = current_image
                        st.session_state['active_label'] = cls_choice.upper()
                except Exception as e:
                    st.error(f"Error accessing dataset: {e}")
        else:
            up_file = st.file_uploader("Upload an aerial image for analysis", type=['jpg', 'png', 'jpeg'])
            if up_file:
                current_image = Image.open(up_file)
                st.session_state['active_img'] = current_image
                st.session_state['active_label'] = "UNKNOWN"

        if 'active_img' in st.session_state:
            active_img = st.session_state['active_img']
            d_col1, d_col2 = st.columns(2)
            
            with d_col1:
                st.image(active_img, caption=f"Source: {st.session_state.get('active_label', 'UPLOAD')}", width=450)
            
            with d_col2:
                # Preprocess
                p_img = active_img.resize((224, 224))
                p_arr = tf.keras.utils.img_to_array(p_img) / 255.0
                p_arr = np.expand_dims(p_arr, axis=0)
                
                with st.spinner("Classifying image via MobileNetV2 Neural Network..."):
                    raw_pred = model.predict(p_arr, verbose=0)[0][0]
                
                final_label = "DRONE" if raw_pred > 0.5 else "BIRD"
                final_conf = raw_pred if raw_pred > 0.5 else (1 - raw_pred)
                
                st.subheader(f"System Prediction: {final_label}")
                st.progress(float(final_conf))
                st.write(f"Confidence Level: **{final_conf*100:.2f}%**")
                
                if final_label == "DRONE":
                    st.error("⚠️ ALERT: Unauthorized Drone Detected")
                else:
                    st.success("✅ SAFE: Avian Wildlife Identified")


with tab_usecases:
    st.header("Domain Applications")
    u1, u2, u3 = st.columns(3)
    
    with u1:
        st.subheader("✈️ Airport Operations")
        st.write("Mitigating bird-strike risks by providing real-time perimeter monitoring.")
    
    with u2:
        st.subheader("🛡️ Strategic Defense")
        st.write("Automated identification of surveillance drones in restricted military airspace.")
    
    with u3:
        st.subheader("🔭 Ecology & Research")
        st.write("Tracking migratory bird patterns while filtering out hobbyist drone traffic.")

st.write("---")
st.caption("Developed by Ishan Chowdhury | Aerial Object Intelligence v1.0 | Capstone Presentation")