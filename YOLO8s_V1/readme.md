# YOLOv8s – Pipeline de Treinamento, Avaliação e Predição

O Código treino.py, contém um **pipeline completo para treinamento do YOLOv8s**, indo desde a preparação do dataset até a geração de métricas, gráficos e uma predição final para conferência dos resultados.

O foco do script é facilitar a execução do experimento de ponta a ponta, mantendo o processo organizado e reproduzível.

---

## 📌 Sobre o YOLOv8s

O **YOLOv8s** é um modelo intermediário da família YOLOv8.  
Ele oferece **mais precisão que o YOLOv8n**, mantendo ainda um bom desempenho em termos de velocidade, o que o torna uma opção equilibrada para testes e comparações entre modelos.

Principais características:
- Detector *one-stage*
- Boa relação entre precisão e velocidade
- Adequado para experimentos comparativos e validações

---

## 🔁 Fluxo do Pipeline

O script automatiza todas as etapas necessárias para o treinamento e avaliação do modelo.

### Etapas do Processo

1. **Download do dataset**
   - Dataset baixado automaticamente via **Roboflow** no formato **YOLO**.

2. **Correção de rótulos**
   - Ajuste dos rótulos para manter apenas uma classe.
   - Padronização necessária para o treinamento correto do modelo.

3. **Criação do arquivo YAML**
   - Geração automática do arquivo `data.yaml` utilizado no treinamento.

4. **Treinamento do modelo**
   - Treinamento do **YOLOv8s** com:
     - Ajustes de hiperparâmetros
     - Resolução de entrada maior
     - *Data augmentations* mais fortes
   - Configuração pensada para melhorar a capacidade de generalização.

5. **Análise das métricas**
   - Leitura do arquivo CSV gerado ao final do treinamento.
   - Criação de gráficos de:
     - mAP@50
     - Precision
     - Recall
     - Losses

6. **Exportação dos resultados**
   - Compactação automática de todos os arquivos do experimento em um único arquivo **ZIP**.

7. **Predição final**
   - Execução de uma predição utilizando uma imagem do próprio conjunto de teste.
   - Salvamento da imagem anotada para verificação visual do desempenho do modelo.

---

## 🧪 Observações

- O pipeline foi organizado para facilitar a reprodução do experimento.
- O uso de *augmentations* mais fortes é compensado por ajustes de hiperparâmetros.
- Ao final da execução, o experimento gera:
  - Métricas quantitativas
  - Gráficos de desempenho
  - Imagens anotadas
  - Arquivo ZIP com todos os resultados

---

## 🚀 Tecnologias Utilizadas

- Python  
- YOLOv8 (Ultralytics)  
- YOLO Dataset Format  
- Roboflow  
- Pandas / Matplotlib  

---

## 📊 Métricas Avaliadas

- mAP@50  
- Precision  
- Recall  
- Losses  

---

## 📄 Licença

Projeto utilizado para fins acadêmicos e experimentais.

