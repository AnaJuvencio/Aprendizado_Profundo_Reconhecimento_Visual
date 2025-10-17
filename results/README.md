# 📁 Estrutura de Resultados

Esta pasta contém todos os arquivos gerados durante os experimentos de classificação de lixo reciclável.

## 📂 Organização das Pastas

### 🤖 `models/`
Modelos treinados salvos em formato Keras (.keras):
- `cnn_baseline_best.keras` - Melhor modelo CNN baseline (57.83% acurácia)

### 📊 `plots/`
Visualizações geradas durante o treinamento e avaliação:

#### `accuracy/`
Gráficos de evolução da acurácia:
- `acc_cnn_baseline.png` - Curva de acurácia CNN baseline
- `acc_mobilenetv2_tl_freeze.png` - Curva de acurácia MobileNetV2 (fase frozen)
- `acc_mobilenetv2_tl_finetune.png` - Curva de acurácia MobileNetV2 (fine-tuning)

#### `loss/`
Gráficos de evolução do loss:
- `loss_cnn_baseline.png` - Curva de loss CNN baseline
- `loss_mobilenetv2_tl_freeze.png` - Curva de loss MobileNetV2 (fase frozen)
- `loss_mobilenetv2_tl_finetune.png` - Curva de loss MobileNetV2 (fine-tuning)

#### `confusion_matrices/`
Matrizes de confusão para análise de erros:
- `cm_abs_cnn_baseline.png` - Matriz de confusão absoluta CNN
- `cm_norm_cnn_baseline.png` - Matriz de confusão normalizada CNN
- `cm_abs_mobilenetv2_tl.png` - Matriz de confusão absoluta MobileNetV2
- `cm_norm_mobilenetv2_tl.png` - Matriz de confusão normalizada MobileNetV2

### 📈 `history/`
Históricos de treino em formato CSV:
- `cnn_baseline_history.csv` - Histórico completo treino CNN baseline
- `mobilenetv2_tl_freeze_history.csv` - Histórico fase frozen MobileNetV2
- `mobilenetv2_tl_finetune_history.csv` - Histórico fine-tuning MobileNetV2

### 📋 `reports/`
Relatórios e métricas de avaliação:
- `models_comparison.csv` - Comparação final entre modelos
- `class_report_cnn_baseline.csv` - Métricas detalhadas por classe (CNN)
- `class_report_mobilenetv2_tl.csv` - Métricas detalhadas por classe (MobileNetV2)

## 🏆 Resumo dos Melhores Resultados

**Melhor Modelo:** MobileNetV2 Transfer Learning
- **Acurácia:** 78.71%
- **Arquivo:** Não salvo (variável `tl` no notebook)
- **Localização das métricas:** `reports/class_report_mobilenetv2_tl.csv`

**Modelo Baseline:** CNN do Zero  
- **Acurácia:** 57.83%
- **Arquivo:** `models/cnn_baseline_best.keras`
- **Localização das métricas:** `reports/class_report_cnn_baseline.csv`

## 📝 Como Utilizar

1. **Carregar modelo salvo:**
```python
from tensorflow import keras
model = keras.models.load_model('results/models/cnn_baseline_best.keras')
```

2. **Analisar históricos:**
```python
import pandas as pd
history = pd.read_csv('results/history/cnn_baseline_history.csv')
```

3. **Visualizar métricas:**
```python
comparison = pd.read_csv('results/reports/models_comparison.csv')
print(comparison)
```

---
*Gerado automaticamente durante os experimentos de Deep Learning - Outubro 2025*