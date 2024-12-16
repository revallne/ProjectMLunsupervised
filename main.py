import pickle
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Body Performance Analysis",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with enhanced styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
        background-color: #fafafa;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        margin-top: 20px;
        background-color: #4CAF50 !important;
        color: white !important;
        height: 3.5rem;
        font-size: 18px;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049 !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }
    
    /* Title container styling */
    .title-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Result container styling */
    .result-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Section styling */
    .section-container {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Input field styling */
    .stNumberInput div[data-baseweb="input"] {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
    }
    
    /* Metric styling */
    .stMetric {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: white;
        border-radius: 15px;
        padding: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 8px;
        padding: 0 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50 !important;
        color: white !important;
    }
    
    /* Success/Warning message styling */
    .stSuccess, .stWarning {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('Model_KMeans_Body_Performance.pkl', 'rb') as f:
        return pickle.load(f)

pipeline = load_model()
encoder = pipeline['encoder']
scaler = pipeline['scaler']
kmeans = pipeline['kmeans']

# Title section with enhanced design
st.markdown("""
    <div class="title-container">
        <h1 style="color: white; font-size: 2.5rem; margin-bottom: 1rem;">🏃‍♂️ Analisis Performa Tubuh</h1>
        <p style='font-size: 1.3rem; color: rgba(255,255,255,0.9);'>Sistem analisis clustering untuk menentukan tingkat performa tubuh Anda</p>
    </div>
""", unsafe_allow_html=True)

# Create tabs with enhanced styling
tab1, tab2 = st.tabs(["📝 Input Data", "ℹ️ Informasi"])

with tab1:
    # Data Pribadi section
    st.markdown("""
        <div class="section-container">
            <h3 style="color: #333; margin-bottom: 1.5rem;">📋 Data Pribadi</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        age = st.number_input('Umur (tahun)', 
                             min_value=0, 
                             max_value=100,
                             step=1, 
                             format="%d",
                             help="Masukkan umur Anda dalam tahun")
    
    with col2:
        gender = st.selectbox('Jenis Kelamin',
                            options=['Male', 'Female'],
                            help="Pilih jenis kelamin Anda")
    
    with col3:
        height = st.number_input('Tinggi Badan (cm)',
                               min_value=0.0,
                               max_value=250.0,
                               step=0.1,
                               help="Masukkan tinggi badan Anda dalam sentimeter")
    
    with col4:
        weight = st.number_input('Berat Badan (kg)',
                               min_value=0.0,
                               max_value=200.0,
                               step=0.1,
                               help="Masukkan berat badan Anda dalam kilogram")

    # Pengukuran Fisik section with enhanced design
    st.markdown("""
        <div class="section-container">
            <h3 style="color: #333; margin-bottom: 1.5rem;">📊 Pengukuran Fisik</h3>
        </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        body_fat = st.number_input('Persentase Lemak Tubuh (%)',
                                 min_value=0.0,
                                 max_value=100.0,
                                 step=0.1,
                                 help="Masukkan persentase lemak tubuh Anda")
        
        systolic = st.number_input('Tekanan Darah Sistolik (mmHg)',
                                 min_value=0,
                                 max_value=200,
                                 step=1,
                                 format="%d",
                                 help="Masukkan tekanan darah sistolik Anda")
    
    with col2:
        grip_force = st.number_input('Kekuatan Genggaman (kg)',
                                   min_value=0.0,
                                   max_value=100.0,
                                   step=0.1,
                                   help="Masukkan kekuatan genggaman Anda dalam kilogram")
        
        sit_and_bend_forward = st.number_input('Sit and Bend Forward (cm)',
                                             min_value=0.0,
                                             max_value=100.0,
                                             step=0.1,
                                             help="Masukkan hasil pengukuran sit and bend forward dalam sentimeter")
    
    with col3:
        broad_jump = st.number_input('Broad Jump (cm)',
                                   min_value=0.0,
                                   max_value=400.0,
                                   step=0.1,
                                   help="Masukkan hasil pengukuran broad jump dalam sentimeter")
        
        sit_ups = st.number_input('Jumlah Sit-Ups',
                                min_value=0,
                                max_value=100,
                                step=1,
                                format="%d",
                                help="Masukkan jumlah sit-ups yang dapat Anda lakukan")

    if st.button('🔍 Analisis Performa'):
        with st.spinner('Menganalisis data...'):
            gender_code = 'M' if gender == 'Male' else 'F'
            data_baru = pd.DataFrame([{
                'age': age,
                'gender': gender_code,
                'height_cm': height,
                'weight_kg': weight,
                'body fat_%': body_fat,
                'systolic': systolic,
                'gripForce': grip_force,
                'sit and bend forward_cm': sit_and_bend_forward,
                'sit-ups counts': sit_ups,
                'broad jump_cm': broad_jump
            }])

            # Transform categorical data
            for column in ['gender']:
                data_baru[column] = encoder[column].transform(data_baru[column])

            # Scale data
            data_baru_scaled = scaler.transform(data_baru)

            # Predict cluster
            cluster_pred = kmeans.predict(data_baru_scaled)
            jarak_cluster = kmeans.transform(data_baru_scaled)

            # Display results with enhanced styling
            st.markdown("""
                <div class="result-container">
                    <h2 style='color: #333; text-align: center; margin-bottom: 2rem; font-size: 2rem;'>📊 Hasil Analisis</h2>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="📈 Jarak ke Cluster Performa Tinggi",
                    value=f"{jarak_cluster[0][0]:.4f}"
                )
            
            with col2:
                st.metric(
                    label="📉 Jarak ke Cluster Performa Rendah",
                    value=f"{jarak_cluster[0][1]:.4f}"
                )

            # Final result with enhanced styling
            if cluster_pred[0] == 0:
                st.success("🌟 Anda termasuk dalam kelompok dengan **PERFORMA TINGGI**")
                st.markdown("""
                    <div style='background-color: #e8f5e9; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                        <p style='color: #2e7d32; font-size: 1.1rem;'>Pertahankan performa Anda dengan tetap konsisten dalam berolahraga dan menjaga pola hidup sehat!</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Anda termasuk dalam kelompok dengan **PERFORMA RENDAH**")
                st.markdown("""
                    <div style='background-color: #fff3e0; padding: 1rem; border-radius: 10px; margin-top: 1rem;'>
                        <p style='color: #ef6c00; font-size: 1.1rem;'>Tingkatkan performa Anda dengan rutin berolahraga dan mengonsumsi makanan bergizi!</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)