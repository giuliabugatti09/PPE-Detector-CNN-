⛑️ Projeto Global Solution: Detector de EPIs (CNN)

Aluno: Giulia Bugatti Fonseca RM562675
Turma: 1tiapf

Projeto desenvolvido para a Global Solution da FIAP, sob o tema "Tecnologias Inteligentes para o Futuro do Trabalho".

1. 🎯 Objetivo do Projeto

O objetivo deste projeto foi desenvolver uma solução de Visão Computacional para melhorar a segurança no ambiente de trabalho. A missão era construir, treinar e avaliar uma Rede Neural Convolucional (CNN) "do zero" (sem transfer learning), capaz de identificar o uso correto de Equipamentos de Proteção Individual (EPI).

O caso de uso específico é a classificação de imagens para determinar se um trabalhador está "Com Capacete" (helmet) ou "Sem Capacete" (head).

2. 🛠️ Tecnologias Utilizadas

Python 3.10+

TensorFlow (Keras): Para a construção, treino e avaliação da CNN.

Streamlit: Para a criação da interface de demonstração interativa (app.py).

Pandas: Para a manipulação e limpeza do dataset (ficheiro .csv).

Scikit-learn: Para a divisão estratégica e balanceamento dos dados.

Matplotlib: Para a visualização das curvas de aprendizagem.

Numpy: Para manipulação de arrays.

3. 🗂️ Dataset e Pré-processamento

O sucesso deste projeto dependeu de um pré-processamento e limpeza de dados rigorosos.

Fonte: Hard Hat Workers Object Detection Dataset (Roboflow)

Dataset Original: O dataset público continha 7000+ imagens, mas era originalmente um dataset de Detecção de Objetos.

Processo de Limpeza e Balanceamento

O dataset bruto não podia ser usado diretamente e apresentava dois grandes problemas:

Filtragem de Ruído: Após a inspeção, foram removidas 531 imagens ambíguas que estavam marcadas como head = 1 E helmet = 1 ao mesmo tempo.

Balanceamento de Classes (Undersampling): O dataset estava severamente desbalanceado (4301 helmet vs. 437 head). Aplicámos uma técnica de Undersampling para reduzir aleatoriamente as amostras da classe majoritária, criando um dataset final perfeitamente balanceado de 874 imagens (437 vs 437).

Divisão: O dataset final foi dividido em 80% para treino (699 imagens) e 20% para validação (175 imagens) usando train_test_split.

4. 🏗️ Arquitetura da CNN (do Zero)

A arquitetura do modelo foi construída do zero, focando em estabilidade e eficiência.

Design: A rede utiliza blocos de Conv2D + BatchNormalization + MaxPooling2D. A BatchNormalization foi crucial para estabilizar o treino.

Camada Final: A camada Flatten foi substituída por GlobalAveragePooling2D para reduzir o número de parâmetros.

Saída: A rede termina com uma única camada Dense(1, activation='sigmoid') e usa a função de perda binary_crossentropy, característica de uma classificação binária.

Sumário do Modelo Final

--- Sumário do Modelo FINAL (Simples) ---
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv1_1 (Conv2D)                │ (None, 128, 128, 32)   │           896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 128, 128, 32)   │           128 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ pool1 (MaxPooling2D)            │ (None, 64, 64, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2_1 (Conv2D)                │ (None, 64, 64, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 64, 64, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ pool2 (MaxPooling2D)            │ (None, 32, 32, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_avg_pool                 │ (None, 64)             │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_layer_1 (Dense)           │ (None, 64)             │         4,160 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_layer (Dropout)         │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ output_layer (Dense)            │ (None, 1)              │            65 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 24,001 (93.75 KB)
 Trainable params: 23,809 (92.99 KB)
 Non-trainable params: 192 (768.00 B)


5. 📊 Resultados e Análise

Após 40 épocas de treino no dataset limpo e balanceado, o modelo atingiu um resultado robusto:

Acurácia de Treino: 83.19%

Acurácia de Validação Final: 84.57%

Curvas de Aprendizagem

Os gráficos demonstram que o modelo está aprendendo ativamente, com a Acurácia de Treino (azul) subindo consistentemente até ~91% e a Perda de Treino diminuindo. No entanto, o modelo apresenta overfitting significativo a partir da 30ª época, evidenciado pela linha da Perda de Validação (laranja) que se torna muito volátil e se afasta drasticamente da linha de treino, indicando que o modelo memorizou os dados de treinamento e está com dificuldades para generalizar. O ponto ideal de desempenho foi atingido por volta da 30ª a 35ª época.

6. ⚠️ Limitações e Próximos Passos

Apesar do sucesso (84.57% de acurácia), o projeto tem limitações que orientam futuros desenvolvimentos:

Dataset Pequeno: A necessidade de balanceamento resultou num conjunto de treino com menos de 700 imagens, o que limita a capacidade final do modelo.

Classificador vs. Detector: Este modelo é um classificador de imagens pré-recortadas. Ele não consegue localizar pessoas numa cena complexa, apenas classificar um crop fornecido.

Próximos Passos:

Data Augmentation: Reativar e treinar com mais épocas para extrair a performance máxima do modelo estável.

Modelo de Detecção (YOLO/SSD): O próximo passo lógico seria a implementação de Transfer Learning (e.g., MobileNetV2) para construir um verdadeiro Detector de Objetos capaz de desenhar caixas delimitadoras (bounding boxes) em tempo real.

7. 🚀 Demonstração (Streamlit)

O modelo treinado (meu_modelo_epi.keras) foi implementado numa aplicação web simples usando Streamlit para demonstração interativa (ficheiro app.py).

8. ⚙️ Como Executar o Projeto

Mantenha todos os ficheiros (notebook, app.py, requirements.txt, modelo) na mesma pasta.

Crie um ambiente virtual: python -m venv venv e ative-o.

Instale as dependências: pip install -r requirements.txt

Para executar a aplicação web:

streamlit run app.py


Para re-treinar o modelo, abra e execute o notebook treinamento_cnn.ipynb (recomenda-se o Google Colab).