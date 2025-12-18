# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import io

# --- Configuração da Página ---
st.set_page_config(
    page_title="Detector de EPI",
    page_icon="⛑️",
    layout="wide"
)

# --- Constantes ---
# O modelo foi treinado com estas dimensões
IMG_SIZE = (128, 128) 
MODEL_PATH = "meu_modelo_epi.keras" 

# --- Carregar o Modelo (com cache para otimização) ---
# A função cache_resource evita recarregar o modelo da memória a cada interação.
@st.cache_resource
def carregar_modelo(caminho_modelo):
    """Carrega o modelo Keras salvo."""
    try:
        model = tf.keras.models.load_model(caminho_modelo)
        return model
    except Exception as e:
        # Exibe um erro se o arquivo .keras não for encontrado
        st.error(f"Erro ao carregar o modelo: Certifique-se de que o arquivo '{MODEL_PATH}' está na mesma pasta. Detalhe: {e}")
        return None

model = carregar_modelo(MODEL_PATH)

# --- Função de Pré-processamento ---
def processar_imagem(imagem_pil):
    """Converte uma imagem PIL para o formato que o modelo espera."""
    # 1. Redimensionar para o tamanho do treino
    img = imagem_pil.resize(IMG_SIZE)
    
    # 2. Converter para um array NumPy
    img_array = np.array(img)
    
    # 3. Garantir que tem 3 canais (remover canal Alfa se for PNG)
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
        
    # 4. Normalizar os pixels (de [0, 255] para [0, 1]) - Passo crucial!
    img_array = img_array / 255.0
    
    # 5. Adicionar uma dimensão de "batch" (lote)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    
    return img_array_expanded

# --- Interface do Streamlit ---

st.title("⛑️ Detector de Equipamento de Proteção Individual (EPI)")
st.subheader("Solução de Visão Computacional para Detecção de Capacetes")

st.markdown("""
Esta aplicação usa uma Rede Neural Convolucional (CNN) treinada do zero para
classificar se uma imagem contém uma pessoa **sem capacete** ('head') ou **com capacete** ('helmet').
""")

# Dividir a interface em colunas para melhor organização
col1, col2 = st.columns(2)

with col1:
    st.header("1. Faça o Upload da Imagem")
    
    # Ficheiro de Upload
    uploaded_file = st.file_uploader(
        "Escolha uma imagem de teste (jpg, jpeg, png)...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        # Ler a imagem
        image = Image.open(uploaded_file)
        
        st.image(image, caption="Imagem Carregada", use_column_width=True)

with col2:
    st.header("2. Resultado da Predição")
    
    if uploaded_file is not None and model is not None:
        # Mensagem de carregamento visual
        with st.spinner("Analisando a imagem..."):
            
            imagem_processada = processar_imagem(image)
            predicao_bruta = model.predict(imagem_processada)
            confianca_bruta = predicao_bruta[0][0] # Valor entre 0.0 e 1.0
            
            # --- CORREÇÃO DA INVERSÃO DE RÓTULOS (CRÍTICO) ---
            # O modelo treinado inverteu as classes internamente (0 -> 'COM CAPACETE').
            # Ajustamos a lógica de exibição aqui:
            
            if confianca_bruta < 0.5: 
                # Se o valor é baixo (perto de 0.0), o modelo está a prever a classe 0 (SEM CAPACETE).
                # Mas, devido à inversão, a classe 0 REAL é COM CAPACETE.
                classe = "COM CAPACETE"
                # Usamos (1 - valor_bruto) para obter a confiança do oposto
                confianca_percentual = (1 - confianca_bruta) * 100 
                st.success(f"**Resultado:** {classe}")
            else:
                # Se o valor é alto (perto de 1.0), o modelo está a prever a classe 1 (COM CAPACETE).
                # Mas, devido à inversão, a classe 1 REAL é SEM CAPACETE.
                classe = "SEM CAPACETE"
                confianca_percentual = confianca_bruta * 100
                st.error(f"**Resultado:** {classe}")
                

            st.metric(
                label=f"Confiança em '{classe}'",
                value=f"{confianca_percentual:.2f} %"
            )
            
            st.info("""
            **Como funciona:**
            * O modelo binário (Sigmoid) retorna um valor entre 0.0 e 1.0.
            * **Observação:** Devido a um mapeamento interno durante o treino, os rótulos foram invertidos. A lógica de exibição foi corrigida para que a predição seja sempre intuitiva:
            * **Valores próximos de 0.0 (saída do modelo) agora significam 'COM CAPACETE'.**
            * **Valores próximos de 1.0 (saída do modelo) agora significam 'SEM CAPACETE'.**
            """)

    elif model is None:
        st.error("Modelo 'meu_modelo_epi.keras' não encontrado. Certifique-se de que está na mesma pasta que o app.py.")
    else:
        st.info("Aguardando o upload de uma imagem...")