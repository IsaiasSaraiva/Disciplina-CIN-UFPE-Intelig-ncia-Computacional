# YOLOv8s – Treinamento em Duas Etapas (Hybrid Beast Mode)

Este repositório apresenta um **pipeline de treinamento em duas fases para o YOLOv8s**, pensado para extrair o máximo desempenho do modelo em cenários com **bases pequenas** e **apenas uma classe**, como tarefas de detecção de danos em fuselagens.

O fluxo combina estabilidade inicial com refinamento agressivo, explorando o melhor dos pesos pré-treinados antes de liberar todo o modelo para *fine-tuning*.

---

## 📌 Ideia do Método

O treinamento é dividido em **duas etapas bem definidas**:

- **Etapa 1 – Estabilização**
  - Congelamento parcial do *backbone*
  - Ajustes mais conservadores
  - Foco em adaptar as camadas finais ao dataset

- **Etapa 2 – Refinamento completo**
  - Liberação de todos os parâmetros
  - *Fine-tuning* mais agressivo
  - Uso de *augmentations* fortes e otimizador **AdamW**

Essa abordagem tende a gerar resultados mais consistentes quando o conjunto de dados é limitado.

---

## 🔁 Fluxo do Pipeline

O script automatiza todo o processo, do preparo dos dados à exportação dos resultados.

### Etapas do Processo

1. **Download do dataset**
   - Dataset obtido automaticamente via **Roboflow** no formato **YOLO**.

2. **Correção de rótulos**
   - Ajuste automático dos rótulos para manter apenas uma classe.
   - Evita inconsistências durante o treinamento.

3. **Criação do arquivo YAML**
   - Geração automática do arquivo `data.yaml` utilizado pelo YOLOv8.

4. **Treinamento – Etapa 1 (Backbone congelado)**
   - Congelamento parcial do *backbone*.
   - Treinamento focado na adaptação inicial do modelo ao dataset.

5. **Treinamento – Etapa 2 (Fine-tuning completo)**
   - Liberação de todos os pesos do modelo.
   - Ajustes mais agressivos de hiperparâmetros.
   - Uso do otimizador **AdamW** e *augmentations* mais fortes.

6. **Análise das métricas**
   - Leitura do CSV gerado ao longo do treinamento.
   - Geração de gráficos de:
     - mAP
     - Precision
     - Recall
     - Losses

7. **Predição de teste**
   - Execução de uma predição utilizando imagens do conjunto de teste.
   - Salvamento das imagens anotadas para inspeção visual.

8. **Exportação dos resultados**
   - Compactação automática de toda a pasta do experimento em um arquivo **ZIP**.

---

## 🧪 Observações

- A estratégia em duas etapas ajuda a evitar *overfitting* em bases pequenas.
- O uso de *augmentations* fortes é mais seguro após a fase de estabilização.
- O pipeline deixa todos os artefatos organizados para análise posterior.

---

## 🚀 Tecnologias Utilizadas

- Python  
- YOLOv8 (Ultralytics)  
- YOLO Dataset Format  
- Roboflow  
- AdamW  
- Pandas / Matplotlib  

---

## 📊 Métricas Avaliadas

- mAP  
- Precision  
- Recall  
- Losses  

---

## 📄 Licença

Projeto desenvolvido para fins acadêmicos, experimentais e de pesquisa.

