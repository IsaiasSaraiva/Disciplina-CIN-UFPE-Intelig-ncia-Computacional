# RetinaNet – Pipeline Completo de Detecção de Objetos

O Código treino.py, apresenta um **pipeline completo de detecção de objetos utilizando o RetinaNet**, cobrindo desde o download e preparo do dataset até o treinamento, avaliação, visualização das predições e exportação final dos resultados.

O **RetinaNet** é um detector *one-stage* conhecido por empregar a **Focal Loss**, que lida de forma eficiente com o desequilíbrio entre classes, melhorando a detecção de objetos pequenos ou menos frequentes.

---

## 📌 Visão Geral do RetinaNet

O **RetinaNet** possui as seguintes características principais:

- Detector de **uma única passada (one-stage)**.
- Uso de **Focal Loss** para reduzir o impacto de exemplos fáceis durante o treinamento.
- Bom equilíbrio entre desempenho e simplicidade de pipeline.
- Arquitetura baseada em **ResNet + FPN**.

Essa abordagem é especialmente eficaz em cenários com **datasets desbalanceados**.

---

## 🔁 Pipeline do Projeto

O script implementa um fluxo completo de treinamento e avaliação utilizando o **Detectron2**.

### Passo a Passo

1. **Download do dataset**
   - Dataset obtido via **Roboflow** no formato **COCO** (formato nativo do Detectron2).

2. **Conversão das anotações**
   - Conversão das anotações COCO para o formato interno esperado pelo Detectron2.

3. **Registro dos datasets**
   - Registro automático dos conjuntos de:
     - Treino (`train`)
     - Validação (`valid`)
     - Teste (`test`)
   - Uso do `DatasetCatalog` e `MetadataCatalog`.

4. **Configuração do modelo**
   - Carregamento do modelo **RetinaNet R50-FPN** pré-treinado no COCO.
   - Ajuste do modelo para **apenas uma classe**.

5. **Configuração de hiperparâmetros**
   - Definição de:
     - *Learning rate*
     - *Warmup*
     - *Steps*
     - Número de épocas
     - *Batch size*

6. **Treinamento**
   - Execução do treinamento completo.
   - Salvamento automático de *checkpoints* no diretório do experimento.

7. **Avaliação**
   - Avaliação no conjunto de validação utilizando o **COCOEvaluator**.

8. **Visualização das predições**
   - Geração de imagens anotadas com as predições no conjunto de teste.

9. **Exportação dos resultados**
   - Salvamento das métricas em formato **JSON**.
   - Compactação de todos os resultados em um arquivo **ZIP** para análise posterior.

---

## 🧪 Observações Importantes

- O RetinaNet é um detector **one-stage**, mais simples que modelos baseados em propostas.
- A **Focal Loss** evita que o modelo foque excessivamente em exemplos fáceis.
- O uso do formato COCO facilita a integração com o Detectron2.
- O pipeline gera automaticamente:
  - Métricas quantitativas
  - Imagens anotadas
  - Arquivo ZIP com todos os resultados do experimento

---

## 🚀 Tecnologias Utilizadas

- Python  
- Detectron2  
- RetinaNet (ResNet-50 + FPN)  
- COCO Dataset Format  
- Roboflow  

---

## 📊 Métricas Avaliadas

- Métricas COCO (AP, AP50, AP75, etc.)
- Precision  
- Recall  

---

## 📄 Licença

Este projeto é destinado a fins acadêmicos e de pesquisa.

