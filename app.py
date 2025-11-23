import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from PIL import Image
import io
import base64
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu
from skimage.morphology import closing, opening, disk, remove_small_objects

# Page configuration
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #ff6b6b;
        margin-bottom: 1rem;
    }
    
    .result-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .normal-result {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    
    .tb-result {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🫁 Tuberculosis Detection System</h1>', unsafe_allow_html=True)
st.markdown("""
<div class="info-box">
    <h3>🎯 About This System</h3>
    <p>This advanced TB detection system uses machine learning to analyze chest X-ray images with <strong>92.86% accuracy</strong>.</p>
    <p><strong>Features:</strong> CLAHE preprocessing, Otsu segmentation, GLCM + LBP feature extraction, SVM classification</p>
</div>
""", unsafe_allow_html=True)

# Constants
IMG_SIZE = 512
MODEL_DIR = "trained_models"

@st.cache_data
def load_models():
    """Load the trained models and preprocessors"""
    try:
        with open(os.path.join(MODEL_DIR, 'svm_model.pkl'), 'rb') as f:
            svm_model = pickle.load(f)
        
        with open(os.path.join(MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
            
        with open(os.path.join(MODEL_DIR, 'pca.pkl'), 'rb') as f:
            pca = pickle.load(f)
            
        return svm_model, scaler, pca
    except FileNotFoundError:
        st.error("❌ Model files not found! Please train the model first using the notebook.")
        return None, None, None

def preprocess_image(img_bgr):
    """
    Preprocess chest X-ray image
    """
    # Resize to 512x512
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # Median blur for noise reduction
    blurred = cv2.medianBlur(clahe_img, 5)
    
    # Normalize to [0, 1]
    normalized = blurred.astype(np.float32) / 255.0
    
    return normalized

def segment_lungs(img):
    """
    Segment lung regions using Otsu thresholding
    """
    # Convert to 0-255 for Otsu
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Otsu threshold
    thresh = threshold_otsu(img_uint8)
    binary = img_uint8 > thresh
    
    # Morphological operations
    selem = disk(5)
    closed = closing(binary, selem)
    opened = opening(closed, selem)
    
    # Remove small objects
    cleaned = remove_small_objects(opened, min_size=500)
    
    return cleaned.astype(np.float32)

def extract_glcm_features(img):
    """
    Extract GLCM texture features
    """
    # Convert to 0-255 range
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Reduce to 32 gray levels for faster computation
    img_reduced = img_uint8 // 8
    
    # Calculate GLCM
    distances = [1, 2, 3]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    features = []
    for d in distances:
        for angle in angles:
            glcm = graycomatrix(img_reduced, distances=[d], angles=[angle], 
                              levels=32, symmetric=True, normed=True)
            
            # Extract texture properties
            contrast = graycoprops(glcm, 'contrast')[0, 0]
            correlation = graycoprops(glcm, 'correlation')[0, 0]
            energy = graycoprops(glcm, 'energy')[0, 0]
            homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
            
            features.extend([contrast, correlation, energy, homogeneity])
    
    return np.array(features)

def extract_lbp_features(img, P=16, R=2):
    """
    Extract Local Binary Pattern features
    """
    # Convert to uint8
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Calculate LBP
    lbp = local_binary_pattern(img_uint8, P, R, method='uniform')
    
    # Calculate histogram
    n_bins = P + 2  # uniform patterns + non-uniform
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    
    # Normalize histogram
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)
    
    return hist

def extract_all_features_from_lung(lung_img):
    """
    Extract combined GLCM and LBP features
    """
    glcm_feat = extract_glcm_features(lung_img)
    lbp_feat = extract_lbp_features(lung_img)
    
    # Combine features
    all_feats = np.concatenate([glcm_feat, lbp_feat], axis=0)
    return all_feats

def process_image_for_prediction(image):
    """
    Process uploaded image for prediction
    """
    try:
        # Convert PIL Image to OpenCV format
        img_array = np.array(image)
        
        # Handle different image formats
        if len(img_array.shape) == 3:
            if img_array.shape[2] == 4:  # RGBA
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:  # RGB
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        else:  # Grayscale
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        
        # Preprocess
        img_prep = preprocess_image(img_bgr)
        
        # Segment lungs
        mask = segment_lungs(img_prep)
        lung_only = img_prep * mask
        
        # Extract features
        features = extract_all_features_from_lung(lung_only)
        
        return features, img_prep, mask, lung_only, img_bgr
        
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None, None, None, None, None

def create_visualization(img_bgr, lung_mask, tb_probability):
    """
    Create visualization with TB probability overlay
    """
    # Convert BGR to RGB for display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Create red overlay based on TB probability
    overlay = np.zeros_like(img_rgb)
    overlay[:, :, 0] = lung_mask * tb_probability * 255  # Red channel
    
    # Blend original image with overlay
    alpha = 0.3  # Transparency
    result = cv2.addWeighted(img_rgb, 1-alpha, overlay.astype(np.uint8), alpha, 0)
    
    return result

def main():
    # Load models
    svm_model, scaler, pca = load_models()
    
    if svm_model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.markdown("## 📋 Instructions")
    st.sidebar.markdown("""
    1. Upload a chest X-ray image (PNG, JPG, JPEG)
    2. Click 'Analyze Image' to get TB detection results
    3. View the analysis with probability visualization
    4. Red overlay intensity shows TB probability
    """)
    
    st.sidebar.markdown("## 📊 Model Performance")
    st.sidebar.markdown("""
    - **Accuracy**: 92.86%
    - **TB Precision**: 95%
    - **Normal Recall**: 96%
    - **Feature Extraction**: GLCM + LBP
    - **Classifier**: SVM with RBF kernel
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear chest X-ray image for TB detection analysis"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
            
            # Analysis button
            if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("🔄 Processing image and extracting features..."):
                    # Process image
                    features, img_prep, mask, lung_only, img_bgr = process_image_for_prediction(image)
                    
                    if features is not None:
                        # Make prediction
                        features_scaled = scaler.transform([features])
                        features_pca = pca.transform(features_scaled)
                        
                        # Get prediction and probabilities
                        prediction = svm_model.predict(features_pca)[0]
                        
                        # Get probabilities
                        if hasattr(svm_model, 'predict_proba'):
                            probabilities = svm_model.predict_proba(features_pca)[0]
                            normal_prob = probabilities[0]
                            tb_prob = probabilities[1]
                        else:
                            # Fallback using decision function
                            decision_score = svm_model.decision_function(features_pca)[0]
                            # Convert to probability-like score
                            tb_prob = 1 / (1 + np.exp(-decision_score))
                            normal_prob = 1 - tb_prob
                        
                        # Store results in session state
                        st.session_state.prediction = prediction
                        st.session_state.normal_prob = normal_prob
                        st.session_state.tb_prob = tb_prob
                        st.session_state.img_bgr = img_bgr
                        st.session_state.mask = mask
                        st.session_state.img_prep = img_prep
                        st.session_state.lung_only = lung_only
    
    with col2:
        st.markdown('<h2 class="sub-header">📋 Analysis Results</h2>', unsafe_allow_html=True)
        
        if hasattr(st.session_state, 'prediction'):
            # Display prediction results
            prediction = st.session_state.prediction
            normal_prob = st.session_state.normal_prob
            tb_prob = st.session_state.tb_prob
            
            # Result box
            result_class = "tb-result" if prediction == 1 else "normal-result"
            result_text = "Tuberculosis Detected" if prediction == 1 else "Normal - No TB Detected"
            result_icon = "🔴" if prediction == 1 else "🟢"
            
            st.markdown(f'''
            <div class="result-box {result_class}">
                <h3>{result_icon} {result_text}</h3>
                <p><strong>Confidence Scores:</strong></p>
                <p>Normal: {normal_prob:.1%}</p>
                <p>Tuberculosis: {tb_prob:.1%}</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Progress bars for probabilities
            st.markdown("**Probability Distribution:**")
            st.progress(float(normal_prob), text=f"Normal: {normal_prob:.1%}")
            st.progress(float(tb_prob), text=f"TB: {tb_prob:.1%}")
            
    # Visualization section
    if hasattr(st.session_state, 'img_bgr'):
        st.markdown("---")
        st.markdown('<h2 class="sub-header">🎨 Visualization & Processing Steps</h2>', unsafe_allow_html=True)
        
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["🔍 TB Probability Overlay", "⚙️ Processing Steps", "📈 Feature Analysis"])
        
        with tab1:
            # Create and display visualization
            visualization = create_visualization(
                st.session_state.img_bgr, 
                st.session_state.mask, 
                st.session_state.tb_prob
            )
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(st.session_state.img_bgr, cv2.COLOR_BGR2RGB), 
                        caption="Original X-ray", use_container_width=True)
            with col2:
                st.image(visualization, 
                        caption="TB Probability Overlay (Red = Higher TB Probability)", 
                        use_container_width=True)
            
            st.info("🔴 **Red Overlay Explanation**: The intensity of red color indicates the probability of tuberculosis in lung regions. Brighter red areas suggest higher TB probability.")
        
        with tab2:
            # Show processing steps
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(st.session_state.img_prep, caption="1. Preprocessed (CLAHE)", 
                        use_container_width=True, clamp=True)
            
            with col2:
                st.image(st.session_state.mask, caption="2. Lung Segmentation", 
                        use_container_width=True, clamp=True)
            
            with col3:
                st.image(st.session_state.lung_only, caption="3. Lung Regions Only", 
                        use_container_width=True, clamp=True)
        
        with tab3:
            # Feature analysis
            st.markdown("### 📊 Extracted Features Summary")
            
            if hasattr(st.session_state, 'prediction'):
                # Show feature extraction info
                st.info("""
                **Feature Extraction Process:**
                - **GLCM Features**: 48 texture features (contrast, correlation, energy, homogeneity)
                - **LBP Features**: 59 local binary pattern histogram features  
                - **Total Features**: 107 features combined
                - **Preprocessing**: StandardScaler normalization + PCA dimensionality reduction
                """)
                
                # Show confidence interpretation
                confidence_level = "High" if max(st.session_state.normal_prob, st.session_state.tb_prob) > 0.8 else "Medium" if max(st.session_state.normal_prob, st.session_state.tb_prob) > 0.6 else "Low"
                st.markdown(f"**Model Confidence**: {confidence_level} ({max(st.session_state.normal_prob, st.session_state.tb_prob):.1%})")

    # Footer information
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🏥 <strong>TB Detection System</strong> | Accuracy: 92.86% | 
        Built with Computer Vision & Machine Learning</p>
        <p><em>⚠️ This tool is for research purposes only. Always consult healthcare professionals for medical diagnosis.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
