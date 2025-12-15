# YOLOv8n – Pipeline de Treinamento e Avaliação

O Código treino.py contém um **pipeline completo para treinamento do YOLOv8n**, cobrindo desde a obtenção do dataset até a análise dos resultados finais.  
O objetivo é permitir a **reprodução integral do experimento**, com o mínimo de ajustes manuais.

O **YOLOv8n** é a versão mais leve da família YOLOv8, indicado para aplicações em que **velocidade** e **baixo consumo de GPU** são mais importantes do que a precisão máxima.

---

## 📌 Sobre o YOLOv8n

- Detector *one-stage* rápido e compacto  
- Baixa demanda computacional  
- Adequado para ambientes com recursos limitados  
- Bom desempenho para tarefas gerais de detecção

---

## 🔁 Fluxo do Pipeline

O script automatiza todas as etapas necessárias para o treinamento e avaliação do modelo.

### Etapas Executadas

1. **Download do dataset**
   - Dataset obtido automaticamente via **Roboflow** no formato **YOLO**.

2. **Ajuste de rótulos**
   - Conversão dos rótulos da classe `1` para `0`.
   - Padronização para treinamento com **classe única**.

3. **Geração do arquivo YAML**
   - Criação automática do arquivo `data.yaml` utilizado pelo YOLOv8.

4. **Treinamento do modelo**
   - Treinamento do **YOLOv8n** por **100 épocas**.
   - Uso de parâmetros ajustados e *augmentations* básicas para manter estabilidade.

5. **Análise de métricas**
   - Leitura do arquivo CSV gerado durante o treinamento.
   - Geração de gráficos de:
     - mAP
     - Precision
     - Recall
     - Losses

6. **Exportação dos resultados**
   - Compactação de toda a pasta de resultados em um arquivo **ZIP**.

---

## 🧪 Observações

- O pipeline foi pensado para ser simples e reproduzível.
- O formato YOLO facilita a integração com o ecossistema Ultralytics.
- Os gráficos ajudam a acompanhar a convergência e o desempenho do modelo.
- Ao final do processo, todos os artefatos ficam organizados para análise posterior.

---

## 🚀 Tecnologias Utilizadas

- Python  
- YOLOv8 (Ultralytics)  
- YOLO Dataset Format  
- Roboflow  
- Pandas / Matplotlib  

---

## 📊 Métricas Avaliadas

- mAP  
- Precision  
- Recall  
- Losses  

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos e de pesquisa.

