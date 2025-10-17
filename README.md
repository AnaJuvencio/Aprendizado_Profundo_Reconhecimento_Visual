# Classificação de Lixo Reciclável com Deep Learning

## Descrição do Projeto

Este projeto implementa e compara dois modelos de aprendizado profundo para classificação automática de lixo reciclável utilizando o dataset **TrashNet**. O objetivo é desenvolver soluções robustas para auxiliar na separação automatizada de resíduos recicláveis, contribuindo para práticas ambientalmente sustentáveis.

## Objetivo

Desenvolver e comparar modelos de classificação de imagens para distinguir entre 6 categorias de resíduos:
- **Cardboard** (Papelão)
- **Glass** (Vidro) 
- **Metal** (Metal)
- **Paper** (Papel)
- **Plastic** (Plástico)
- **Trash** (Lixo comum)

## Dataset

**TrashNet Dataset:**
- **Total de imagens:** 2.527
- **Classes:** 6 categorias de resíduos
- **Resolução:** Redimensionada para 160×160 pixels
- **Divisão:**
  - Treino: 2.022 imagens (80%)
  - Validação: 253 imagens (10%)  
  - Teste: 252 imagens (10%)

### Distribuição por Classe:
| Classe | Quantidade | Percentual |
|--------|------------|------------|
| Paper | 594 | 23.5% |
| Glass | 501 | 19.8% |
| Plastic | 482 | 19.1% |
| Cardboard | 403 | 16.0% |
| Metal | 410 | 16.2% |
| Trash | 137 | 5.4% |

## Modelos Implementados

### 1. CNN Baseline (Arquitetura do Zero)

**Arquitetura:**
- 3 blocos convolucionais (32, 64, 128 filtros)
- BatchNormalization + ReLU + MaxPooling2D
- Dropout (0.3) após cada bloco convolucional
- GlobalAveragePooling2D
- Dense(256) + Dropout(0.5) + Dense(6)

**Configurações de Treino:**
- **Otimizador:** AdamW (learning_rate=1e-4, weight_decay=1e-4)
- **Loss:** Sparse Categorical Crossentropy
- **Épocas:** 20 (early stopping)
- **Callbacks:** ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
- **Parâmetros:** 128.486 (500KB)

### 2. MobileNetV2 Transfer Learning

**Arquitetura:**
- **Base:** MobileNetV2 pré-treinada (ImageNet)
- **Estratégia:** Transfer Learning em 2 fases
  - Fase 1: Base congelada (8 épocas)
  - Fase 2: Fine-tuning das últimas 40 camadas (7 épocas)
- **Cabeçalho:** GlobalAveragePooling2D + Dropout(0.2) + Dense(6)

**Configurações de Treino:**
- **Otimizador:** AdamW (1e-4, weight_decay=1e-4)
- **Loss:** Sparse Categorical Crossentropy
- **Callbacks:** EarlyStopping

## 🏆 Resultados dos Experimentos

### Comparação Geral dos Modelos

| Modelo | Acurácia Final | Parâmetros | Tempo/Época |
|--------|----------------|------------|-------------|
| **MobileNetV2 TL** | **78.71%** | ~2.3M | ~30s |
| **CNN Baseline** | 57.83% | 128K | ~70s |

** Diferença de Performance:** 20.88 pontos percentuais a favor do Transfer Learning

### Resultados Detalhados por Classe

#### MobileNetV2 Transfer Learning (Melhor Modelo)
| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Cardboard | 81.82% | 79.41% | 80.60% | 34 |
| Glass | 78.72% | 72.55% | 75.51% | 51 |
| Metal | 88.24% | 73.17% | 80.00% | 41 |
| Paper | 83.87% | 85.25% | 84.55% | 61 |
| Plastic | 70.59% | 80.00% | 75.00% | 45 |
| Trash | 54.55% | 70.59% | 61.54% | 17 |

#### CNN Baseline
| Classe | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| Cardboard | 61.36% | 79.41% | 69.23% | 34 |
| Glass | 61.76% | 44.68% | 51.85% | 47 |
| Metal | 75.76% | 59.52% | 66.67% | 42 |
| Paper | 49.02% | 83.33% | 61.73% | 60 |
| Plastic | 66.67% | 48.00% | 55.81% | 50 |
| Trash | 0.00% | 0.00% | 0.00% | 16 |

### Análise de Performance

#### ✅ Pontos Fortes do MobileNetV2:
- **Excelente performance geral** (78.71% acurácia)
- **Convergência rápida** devido ao Transfer Learning
- **Boa generalização** em todas as classes
- **Eficiência computacional** durante inferência

#### ⚠️ Limitações Identificadas:
- **Classe "Trash" mais desafiadora** (menor recall)
- **Desbalanceamento do dataset** afeta classes minoritárias
- **Paper vs Cardboard** apresentam alguma confusão

#### 📊 CNN Baseline:
- **Performance limitada** mas aceitável para arquitetura simples
- **Dificuldade com classe "Trash"** (0% precision/recall)
- **Overfitting mais pronunciado** devido ao tamanho limitado do dataset

## Técnicas de Otimização Aplicadas

### Data Augmentation:
- RandomFlip (horizontal)
- RandomRotation (±3°)
- RandomZoom (±5%)

### Regularização:
- Dropout nas camadas convolucionais (0.3)
- Dropout na camada densa (0.5)
- Weight Decay (AdamW)

### Estratégias de Treino:
- Early Stopping (patience=6)
- ReduceLROnPlateau
- ModelCheckpoint (best validation accuracy)

## 📁 Estrutura de Arquivos

```
📦 results/
 ┣ 🤖 models/
 ┃ ┗ cnn_baseline_best.keras          # Melhor modelo CNN baseline
 ┣ 📊 plots/
 ┃ ┣ accuracy/                        # Curvas de acurácia por modelo
 ┃ ┣ loss/                           # Curvas de loss por modelo  
 ┃ ┗ confusion_matrices/             # Matrizes de confusão
 ┣ 📈 history/
 ┃ ┣ cnn_baseline_history.csv        # Histórico treino CNN
 ┃ ┣ mobilenetv2_tl_freeze_history.csv    # Histórico fase frozen
 ┃ ┗ mobilenetv2_tl_finetune_history.csv  # Histórico fine-tuning
 ┗ 📋 reports/
   ┣ models_comparison.csv            # Comparação entre modelos
   ┣ class_report_cnn_baseline.csv    # Métricas detalhadas CNN
   ┗ class_report_mobilenetv2_tl.csv  # Métricas detalhadas MobileNetV2
```

### Organização dos Resultados:
- **`results/models/`** - Modelos treinados salvos (.keras)
- **`results/plots/`** - Todas as visualizações (gráficos e matrizes)
- **`results/history/`** - Históricos de treino em CSV
- **`results/reports/`** - Relatórios de métricas e comparações

## � Como Executar

### 1. **Clonar repositório:**
```bash
git clone https://github.com/AnaJuvencio/Aprendizado_Profundo_Reconhecimento_Visual.git
cd Aprendizado_Profundo_Reconhecimento_Visual
```

### 2. **Baixar o Dataset TrashNet:**
```bash
# Opção 1: Clonar repositório do TrashNet
git clone https://github.com/garythung/trashnet.git trashnet-master

# Opção 2: Download direto (ZIP)
# Acesse: https://github.com/garythung/trashnet
# Extraia para: trashnet-master/
```

**⚠️ Estrutura esperada:**
```
📦 Projeto/
 ┗ 📂 trashnet-master/
   ┗ 📂 data/
     ┗ 📂 dataset-resized/
       ┣ 📁 cardboard/
       ┣ 📁 glass/
       ┣ 📁 metal/
       ┣ 📁 paper/
       ┣ 📁 plastic/
       ┗ 📁 trash/
```

### 3. **Instalar dependências:**
```bash
pip install tensorflow scikit-learn pandas matplotlib
```

### 4. **Executar notebook:**
```bash
jupyter notebook Projeto_Aprendizado_Profundo.ipynb
```

### 5. **Explorar resultados:**
```bash
# Visualizar estrutura de resultados
ls results/

# Carregar modelo salvo
python -c "from tensorflow import keras; model = keras.models.load_model('results/models/cnn_baseline_best.keras')"
```

## 📥 Sobre o Dataset

**Por que o dataset não está no repositório?**
- **Tamanho:** ~500MB+ (excede limites do GitHub)
- **Boa prática:** Datasets grandes devem ser baixados separadamente  
- **Performance:** Mantém o repositório leve e clones rápidos

**TrashNet Dataset:**
- **Fonte original:** https://github.com/garythung/trashnet
- **Licença:** Consulte o repositório original
- **Tamanho:** 2.527 imagens (160x160px redimensionadas)

## �🔧 Configuração do Ambiente

### Dependências:
```python
tensorflow>=2.16.1
scikit-learn
pandas
matplotlib
numpy
```

### Hardware Utilizado:
- **CPU:** Processamento em CPU (sem GPU)
- **RAM:** Configuração padrão
- **Batch Size:** 16 (otimizado para CPU)

## 📈 Conclusões e Insights

### Principais Achados:

1. **Transfer Learning Superior:** MobileNetV2 supera significativamente a CNN baseline (+20.88%)

2. **Eficiência do Pré-treinamento:** Utilizar features pré-treinadas no ImageNet acelera convergência e melhora generalização

3. **Desafios do Desbalanceamento:** Classe "Trash" (5.4% do dataset) apresenta maior dificuldade

4. **Regularização Efetiva:** Dropout e weight decay preveniram overfitting severo

### Melhorias Futuras:

1. **Balanceamento de Classes:**
   - Implementar class weights
   - Técnicas de data augmentation específicas
   - Coleta de mais dados para classes minoritárias

2. **Arquiteturas Avançadas:**
   - EfficientNet
   - Vision Transformers
   - Ensemble methods

3. **Otimizações:**
   - Mixed precision training
   - Learning rate scheduling avançado
   - Cross-validation

4. **Aplicação Prática:**
   - Deploy em dispositivos mobile
   - API para classificação em tempo real
   - Interface web para usuários finais


## Métricas de Sucesso Alcançadas

- ✅ **Acurácia > 75%** com MobileNetV2 Transfer Learning
- ✅ **Modelo baseline funcional** com arquitetura simples
- ✅ **Convergência estável** sem overfitting severo
- ✅ **Documentação completa** e reprodutível
- ✅ **Análise comparativa detalhada** entre abordagens


## Como Obter o Dataset

**Download do TrashNet:**
   - Acesse: https://github.com/garythung/trashnet
   - Clone: `git clone https://github.com/garythung/trashnet.git`
   - Ou baixe o ZIP diretamente

---

**Desenvolvido por:** AnaJuvencio  
**Data:** Outubro 2025  
**Framework:** TensorFlow 2.16.1

