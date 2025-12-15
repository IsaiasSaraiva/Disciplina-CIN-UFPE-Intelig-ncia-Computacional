# YOLOv10s – Pipeline Completo de Detecção de Objetos

O Código treino.py apresenta um **pipeline completo para treinamento, avaliação e visualização de resultados utilizando o YOLOv10s**, a versão mais leve da família YOLOv10.  
O foco é obter **alta velocidade de inferência**, **baixo consumo de GPU** e **boa precisão** para tarefas gerais de detecção de objetos.

---

## 📌 Visão Geral do YOLOv10s

O **YOLOv10s** é caracterizado por:

- Detector **one-stage** extremamente rápido.
- Arquitetura otimizada para **baixo custo computacional**.
- Ideal para aplicações em tempo real ou ambientes com recursos limitados.
- Boa relação entre **velocidade × precisão**.

---

## 🔁 Pipeline do Projeto

O script executa todo o fluxo necessário para treinar e validar o YOLOv10s utilizando o ecossistema **Ultralytics**.

### Passo a Passo

1. **Download do dataset**
   - Dataset obtido via **Roboflow** já no formato **YOLO**.

2. **Correção de rótulos**
   - Conversão automática dos rótulos:
     - Classe `1` → Classe `0`
   - Garante compatibilidade com treinamento de **classe única**.

3. **Geração do arquivo YAML**
   - Criação automática do arquivo `data.yaml` utilizado pelo Ultralytics.

4. **Treinamento do modelo**
   - Inicialização do **YOLOv10s**.
   - Treinamento com **140 épocas**, incluindo ajustes de:
     - *Learning rate*
     - *Momentum*
     - *Weight decay*
     - *Freeze* de camadas
     - *Data augmentations*

5. **Salvamento de métricas**
   - Registro automático de logs e métricas durante o treinamento.

6. **Geração de gráficos**
   - Leitura do arquivo CSV gerado pelo Ultralytics.
   - Criação de gráficos para:
     - Precision
     - Recall
     - mAP@50
     - Losses

7. **Exportação dos resultados**
   - Compactação de todos os arquivos do experimento em um único arquivo **ZIP**.

8. **Predição final**
   - Execução de uma predição de teste.
   - Salvamento da imagem anotada para validação visual do modelo.

---

## 🧪 Observações Importantes

- O YOLOv10s é ideal quando se busca **desempenho em tempo real**.
- O formato YOLO simplifica a integração com o pipeline Ultralytics.
- A correção automática dos rótulos evita inconsistências no treinamento.
- Ao final do pipeline são gerados:
  - Métricas quantitativas
  - Gráficos de desempenho
  - Imagens anotadas
  - Arquivo ZIP com todos os resultados

---

## 🚀 Tecnologias Utilizadas

- Python  
- YOLOv10 (Ultralytics)  
- YOLO Dataset Format  
- Roboflow  
- Pandas / Matplotlib (análise e gráficos)

---

## 📊 Métricas Avaliadas

- Precision  
- Recall  
- mAP@50  
- Losses  

---

## 📄 Licença

Este projeto é destinado a fins acadêmicos e de pesquisa.

