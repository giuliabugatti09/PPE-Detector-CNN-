# =================================================================
# 1. BOILERPLATE DE COMPATIBILIDADE (Obrigatório para Python 3.13)
# =================================================================
import os
import sys

# Ignora conflitos de versão do Protobuf (Erro Gencode/Runtime)
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Mock para evitar erro de inicialização do Keras/DTypePolicy
try:
    from google.protobuf import runtime_version
    runtime_version.ValidateProtobufRuntimeVersion = lambda *args, **kwargs: None
except:
    pass

# =================================================================
# 2. IMPORTS
# =================================================================
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# =================================================================
# 3. CONFIGURAÇÃO E CONSTANTES
# =================================================================
st.set_page_config(page_title="PPE Safety Detector", page_icon="⛑️", layout="wide")

IMG_SIZE = (128, 128)
# Certifique-se de que o nome do arquivo abaixo é o do seu modelo mais recente
MODEL_PATH = os.path.join("models", "best_epi_model.keras")

# =================================================================
# 4. CARREGAMENTO DO MODELO
# =================================================================
@st.cache_resource
def load_my_model():
    try:
        # Registro de objetos para compatibilidade entre versões do Keras
        from keras.src.dtype_policies import dtype_policy
        tf.keras.utils.get_custom_objects()['DTypePolicy'] = dtype_policy.DTypePolicy
        
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        return None

model = load_my_model()

# =================================================================
# 5. FUNÇÕES DE PROCESSAMENTO
# =================================================================
def preprocess(image_pil):
    # 1. Resize para o padrão do treino
    img = image_pil.resize(IMG_SIZE)
    # 2. Garantir 3 canais (RGB)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    # 3. Normalização (Crucial para evitar predições viciadas)
    img_array = np.array(img) / 255.0
    # 4. Expansão para formato de lote (Batch)
    return np.expand_dims(img_array, axis=0)

# =================================================================
# 6. INTERFACE STREAMLIT
# =================================================================
st.title("⛑️ Global Solution: PPE Detection System")
st.subheader("Safety Compliance Monitoring via Computer Vision")

st.info("""
**System Overview:** This CNN model identifies safety violations by distinguishing 
between workers with protective headgear and those at risk.
""")

col1, col2 = st.columns(2)

with col1:
    st.header("1. Image Capture/Upload")
    uploaded_file = st.file_uploader("Upload worker photo for analysis...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Target Image", use_container_width=True)

with col2:
    st.header("2. Safety Analysis")
    if uploaded_file and model:
        with st.spinner("Analyzing PPE compliance..."):
            processed_img = preprocess(img)
            prediction = model.predict(processed_img)
            score = prediction[0][0] # Saída Sigmoid (0.0 a 1.0)
            
            # --- LÓGICA DE DECISÃO (BASEADA NO CLASS MAPPING) ---
            # De acordo com seu treino: Class 0 = Head | Class 1 = Helmet
            
            if score < 0.5:
                # Valores próximos de 0 indicam 'Head' (Sem Capacete)
                label = "NO HELMET DETECTED (DANGER)"
                conf = (1 - score) * 100
                st.error(f"### Status: {label} ⚠️")
                st.warning("Immediate safety intervention required.")
            else:
                # Valores próximos de 1 indicam 'Helmet' (Protegido)
                label = "HELMET DETECTED (SAFE)"
                conf = score * 100
                st.success(f"### Status: {label} ✅")
                st.balloons()
            
            st.metric("Detection Confidence", f"{conf:.2f}%")
            
            with st.expander("Technical Logs"):
                st.write(f"Raw Model Probability: `{score:.4f}`")
                st.write(f"Class Mapping: `0: Head, 1: Helmet`")

st.sidebar.markdown("---")
st.sidebar.write("👤 **Developer:** Giulia Bugatti")
st.sidebar.write("🎓 **Institution:** FIAP")