# 🔬 Documentação Técnica - Classificação de Lixo Reciclável com Deep Learning

## 📋 Índice
1. [Visão Geral Arquitetural](#visão-geral-arquitetural)
2. [Modelos Implementados](#modelos-implementados)
3. [Técnicas de Otimização](#técnicas-de-otimização)
4. [Pipeline de Dados](#pipeline-de-dados)
5. [Análise Comparativa](#análise-comparativa)
6. [Implementação Técnica](#implementação-técnica)

---

##  Visão Geral Arquitetural

### Stack Tecnológico
```python
Framework Principal: TensorFlow 2.16.1
Linguagem: Python 3.11
Bibliotecas Auxiliares:
  - scikit-learn (métricas e avaliação)
  - pandas (manipulação de dados)
  - matplotlib (visualização)
  - numpy (operações numéricas)
```

### Arquitetura do Sistema
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Dataset   │───▶│  Data Pipeline   │───▶│  Model Training │
│   (TrashNet)    │    │ (Preprocessing)  │    │   & Evaluation  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │ Data Augmentation│    │   Results &     │
                    │   & Validation   │    │  Artifacts      │
                    └──────────────────┘    └─────────────────┘
```

---

## Modelos Implementados

### 1. CNN Baseline (Convolutional Neural Network do Zero)

#### **O que é:**
Uma rede neural convolucional construída from scratch, sem usar conhecimento pré-treinado.

#### **Arquitetura Detalhada:**
```python
Model: "cnn_baseline"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 160, 160, 32)     864       
 batch_normalization         (None, 160, 160, 32)     128       
 re_lu (ReLU)                (None, 160, 160, 32)     0         
 max_pooling2d               (None, 80, 80, 32)       0         
 dropout (Dropout)           (None, 80, 80, 32)       0         
                                                                 
 conv2d_1 (Conv2D)           (None, 80, 80, 64)       18432     
 batch_normalization_1       (None, 80, 80, 64)       256       
 re_lu_1 (ReLU)              (None, 80, 80, 64)       0         
 max_pooling2d_1             (None, 40, 40, 64)       0         
 dropout_1 (Dropout)         (None, 40, 40, 64)       0         
                                                                 
 conv2d_2 (Conv2D)           (None, 40, 40, 128)      73728     
 batch_normalization_2       (None, 40, 40, 128)      512       
 re_lu_2 (ReLU)              (None, 40, 40, 128)      0         
 max_pooling2d_2             (None, 20, 20, 128)      0         
 dropout_2 (Dropout)         (None, 20, 20, 128)      0         
                                                                 
 global_average_pooling2d    (None, 128)               0         
 dense (Dense)               (None, 256)               33024     
 dropout_3 (Dropout)         (None, 256)               0         
 dense_1 (Dense)             (None, 6)                 1542      
=================================================================
Total params: 128,486 (501.90 KB)
Trainable params: 128,038 (500.15 KB)
Non-trainable params: 448 (1.75 KB)
```

#### **Componentes Explicados:**

**1. Camadas Convolucionais (Conv2D):**
```python
layers.Conv2D(filters, kernel_size, padding="same", use_bias=False)
```
- **Função:** Extração de features espaciais (bordas, texturas, formas)
- **Filters:** 32 → 64 → 128 (progressão hierárquica)
- **Kernel 3x3:** Janela deslizante para detecção de padrões locais
- **use_bias=False:** Bias é manipulado pelo BatchNormalization

**2. Batch Normalization:**
```python
layers.BatchNormalization()
```
- **Função:** Normaliza entradas de cada camada
- **Benefícios:** Acelera convergência, reduz overfitting, estabiliza gradientes
- **Matemática:** `y = γ(x - μ)/σ + β` onde μ, σ são média e desvio do batch

**3. Função de Ativação ReLU:**
```python
layers.ReLU()
```
- **Função:** `f(x) = max(0, x)`
- **Vantagens:** Evita vanishing gradient, computacionalmente eficiente
- **Não-linearidade:** Permite aprender padrões complexos

**4. Max Pooling:**
```python
layers.MaxPooling2D(pool_size=(2,2))
```
- **Função:** Redução dimensional (downsampling)
- **Benefícios:** Reduz parâmetros, aumenta campo receptivo, invariância translacional

**5. Dropout:**
```python
layers.Dropout(rate)
```
- **Função:** Regularização através de "desligamento" aleatório de neurônios
- **Taxa 0.3:** 30% dos neurônios são zerados durante treino
- **Prevenção:** Overfitting e co-adaptação de features

**6. Global Average Pooling:**
```python
layers.GlobalAveragePooling2D()
```
- **Função:** Converte feature maps em vetores 1D
- **Vantagem vs Flatten:** Reduz drasticamente parâmetros, menos overfitting
- **Operação:** Média de todos valores em cada canal

### 2. MobileNetV2 Transfer Learning

#### **O que é:**
Aproveitamento de uma rede pré-treinada (MobileNetV2) para nova tarefa de classificação.

#### **MobileNetV2 - Conceitos Fundamentais:**

**1. Depthwise Separable Convolutions:**
```python
# Convolution tradicional: O(H × W × C_in × C_out × K²)
# Depthwise Separable: O(H × W × C_in × K²) + O(H × W × C_in × C_out)
```
- **Redução computacional:** ~8-9x menos operações
- **Mantém performance:** Através de separação de responsabilidades

**2. Inverted Residuals:**
```python
# Sequência: 1x1 expand → 3x3 depthwise → 1x1 project
input → [1×1 conv] → [3×3 depthwise] → [1×1 conv] → output
      (expand)     (spatial mixing)    (compress)
```

**3. Linear Bottlenecks:**
- Última camada sem ativação não-linear
- Preserva informação em espaços de baixa dimensão

#### **Estratégia de Transfer Learning:**

**Fase 1 - Feature Extraction (Frozen Base):**
```python
base.trainable = False  # Congela todos os pesos pré-treinados
```
- **Duração:** 8 épocas
- **Learning Rate:** 1e-4
- **Objetivo:** Adaptar apenas o classificador

**Fase 2 - Fine-tuning (Partial Unfreezing):**
```python
base.trainable = True
for layer in base.layers[:-40]:  # Descongela apenas últimas 40 camadas
    layer.trainable = False
```
- **Duração:** 7 épocas  
- **Learning Rate:** 1e-4 (mesmo valor, pois já está baixo)
- **Objetivo:** Ajuste fino das features de alto nível

---

##  Técnicas de Otimização

### 1. Otimizador AdamW

#### **O que é:**
Variante do Adam com weight decay desacoplado.

#### **Funcionamento:**
```python
# Adam tradicional
m_t = β₁ × m_{t-1} + (1-β₁) × g_t
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²
θ_{t+1} = θ_t - α × m̂_t / (√v̂_t + ε)

# AdamW adiciona
θ_{t+1} = θ_{t+1} - α × λ × θ_t  # weight decay desacoplado
```

#### **Vantagens sobre Adam:**
- **Weight decay real:** Não afetado pela adaptação da learning rate
- **Melhor generalização:** Regularização mais efetiva
- **Convergência superior:** Especialmente em tasks de visão computacional

### 2. Learning Rate Scheduling

#### **ReduceLROnPlateau:**
```python
keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss", 
    factor=0.5,        # Reduz LR pela metade
    patience=3,        # Após 3 épocas sem melhoria
    min_lr=1e-6       # LR mínimo
)
```

**Matemática:**
```
LR_new = LR_current × factor  se val_loss não melhorar por 'patience' épocas
```

### 3. Early Stopping

#### **Funcionamento:**
```python
keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True
)
```
- **Prevenção:** Overfitting
- **Eficiência:** Para treino quando não há mais melhoria
- **Restore:** Volta aos melhores pesos encontrados

### 4. Model Checkpointing

#### **Implementação:**
```python
keras.callbacks.ModelCheckpoint(
    "modelo_best.keras",
    monitor="val_accuracy",
    save_best_only=True
)
```
- **Função:** Salva apenas o melhor modelo durante treino
- **Critério:** Maior acurácia de validação

---

## Pipeline de Dados

### 1. Carregamento e Divisão

#### **Estratégia de Split:**
```python
# 80% treino, 20% validação/teste
train_ds = 80% (2022 imagens)
val_ds = 10% (253 imagens)  
test_ds = 10% (252 imagens)
```

#### **Implementação tf.data:**
```python
train_base = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2, 
    subset="training",
    seed=SEED,              # Reprodutibilidade
    image_size=(160, 160),  # Resize automático
    batch_size=16
)
```

### 2. Preprocessing Pipeline

#### **Normalização:**
```python
normalizer = tf.keras.layers.Rescaling(1./255)
# Converte [0, 255] → [0, 1]
```

#### **Data Augmentation:**
```python
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),    # Flip horizontal aleatório
    tf.keras.layers.RandomRotation(0.03),        # Rotação ±3°
    tf.keras.layers.RandomZoom(0.05),           # Zoom ±5%
])
```

**Por que essas transformações?**
- **RandomFlip:** Lixo pode aparecer em qualquer orientação
- **Rotação pequena:** Objetos podem estar ligeiramente inclinados
- **Zoom leve:** Simula diferentes distâncias da câmera

### 3. Otimização de Performance

#### **tf.data Pipeline:**
```python
dataset = dataset.map(prep_function, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

**Explicação:**
- **map + AUTOTUNE:** Paralelização automática do preprocessing
- **prefetch:** Carrega próximo batch enquanto treina o atual
- **Resultado:** Reduz tempo de I/O significativamente

---

## ⚖️ Análise Comparativa

### Performance Computacional

| Aspecto | CNN Baseline | MobileNetV2 TL |
|---------|--------------|----------------|
| **Parâmetros** | 128K | ~2.3M |
| **Tamanho Modelo** | 500KB | ~9MB |
| **Tempo/Época** | ~70s | ~30s |
| **Memória GPU** | Baixo | Médio |
| **Inferência** | Rápida | Muito Rápida |

### Complexidade Algoritmica

#### **CNN Baseline:**
```
Complexidade Treino: O(E × B × H × W × F × K²)
onde:
E = épocas, B = batch_size, H×W = dimensões imagem
F = filtros, K = kernel_size
```

#### **MobileNetV2:**
```
Complexidade Treino: O(E × B × H × W × (C + F))
onde C = canais entrada, F = fator expansão
Redução: ~8-9x devido às depthwise separable convolutions
```

### Trade-offs

| Modelo | Vantagens | Desvantagens |
|--------|-----------|--------------|
| **CNN Baseline** | • Controle total da arquitetura<br>• Modelo leve<br>• Interpretável | • Performance limitada<br>• Treino do zero<br>• Requer mais dados |
| **MobileNetV2** | • Alta performance<br>• Convergência rápida<br>• Features robustas | • Modelo maior<br>• Menos controle<br>• Dependência do pré-treino |

---

## 🔧 Implementação Técnica

### 1. Gerenciamento de Memória

#### **Estratégias Aplicadas:**
```python
# Batch size otimizado para CPU
BATCH_SIZE = 16  # Balanço entre convergência e memória

# Mixed precision (se GPU disponível)
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### 2. Reprodutibilidade

#### **Seeds Determinísticos:**
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
```

### 3. Tratamento de Desbalanceamento

#### **Class Weights Calculados:**
```python
class_weight = {
    cls: total_samples / (num_classes × samples_per_class)
    for cls, samples_per_class in class_counts.items()
}
```

**Matemática:**
```
weight_i = N / (C × N_i)
onde:
N = total de amostras
C = número de classes  
N_i = amostras da classe i
```

### 4. Métricas de Avaliação

#### **Implementação Completa:**
```python
# Métricas por classe
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, 
    labels=range(NUM_CLASSES),
    average=None
)

# Matriz de confusão
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
```

### 5. Salvamento e Serialização

#### **Formato Keras:**
```python
model.save("modelo.keras")  # Formato nativo TensorFlow 2.x
```

**Vantagens do .keras:**
- **Completo:** Arquitetura + pesos + otimizador
- **Compatibilidade:** Futuras versões TensorFlow
- **Portabilidade:** Entre diferentes plataformas

---

## 🎯 Conclusões Técnicas

### Lições Aprendidas

1. **Transfer Learning é Superior:**
   - Features pré-treinadas no ImageNet são transferíveis
   - Reduz significativamente tempo de treino
   - Melhora generalização em datasets pequenos

2. **Regularização é Crucial:**
   - Dropout previne overfitting efetivamente
   - BatchNormalization acelera convergência
   - Weight decay melhora generalização

3. **Pipeline Otimizado Importa:**
   - tf.data reduz gargalos de I/O
   - Data augmentation aumenta diversidade
   - Preprocessing eficiente acelera treino

### Recomendações para Produção

1. **Modelo Recomendado:** MobileNetV2 Transfer Learning
2. **Deployment:** TensorFlow Lite para mobile
3. **Monitoramento:** Drift detection nas predições
4. **Retreino:** Quando acurácia cair abaixo de threshold

---

**Autor:** Equipe de ML Engineering  
**Data:** Outubro 2025  
**Versão Técnica:** 1.0