# Faster R-CNN – Pipeline Completo de Detecção de Objetos

O Código treino.py apresenta um **pipeline completo de detecção de objetos utilizando Faster R-CNN**, abrangendo desde a preparação dos dados até a avaliação final e visualização dos resultados.

O Faster R-CNN é um modelo de detecção de objetos em duas etapas, amplamente utilizado por oferecer **alta precisão** e bom equilíbrio entre desempenho e custo computacional.

---

## 📌 Visão Geral do Faster R-CNN

O **Faster R-CNN** é composto por duas partes principais:

1. **Region Proposal Network (RPN)**  
   Responsável por gerar regiões candidatas onde podem existir objetos.

2. **Rede de Classificação e Regressão**  
   Classifica as regiões propostas e ajusta as *bounding boxes*.

Essa abordagem permite maior precisão na localização e classificação dos objetos.

---

## 🔁 Pipeline do Projeto

O script implementa um fluxo completo de treinamento e avaliação do Faster R-CNN.

### Passo a Passo

1. **Configuração inicial**
   - Definição de hiperparâmetros.
   - Criação do diretório de saída para resultados e checkpoints.

2. **Download do dataset**
   - Dataset obtido via **Roboflow** no formato **COCO**.

3. **Mapper de dados**
   - Aplicação de *data augmentations* moderados (redimensionamento e *flips* leves).

4. **Processamento das anotações**
   - Conversão do formato COCO para o padrão do **Detectron2**.
   - Remoção de *bounding boxes* muito pequenas.
   - Uso apenas de imagens com anotações válidas.

5. **Registro dos datasets**
   - Registro dos conjuntos de **treino**, **validação** e **teste** no `DatasetCatalog` e `MetadataCatalog`.

6. **Configuração do modelo**
   - Modelo pré-treinado do **Detectron2 Model Zoo**:
     - Faster R-CNN com **ResNet-101 + FPN**
   - Ajuste fino de hiperparâmetros para estabilidade e desempenho.

7. **Trainer customizado**
   - Implementação de um `Trainer` customizado para utilizar o *mapper* com augmentations.

8. **Treinamento**
   - Execução do treinamento.
   - Salvamento automático dos *checkpoints*.

9. **Ajuste do threshold de confiança**
   - Teste de múltiplos valores de *confidence threshold*.
   - Seleção do melhor valor com base no **F1-score**.

10. **Avaliação final**
    - Avaliação utilizando o melhor threshold encontrado.
    - Registro das métricas padrão **COCO**.

11. **Visualização dos resultados**
    - Geração de imagens com predições no conjunto de teste.

12. **Relatório e exportação**
    - Geração de relatório resumido contendo:
      - Precision
      - Recall
      - F1-score
    - Compactação dos resultados finais.

---

## 🧪 Observações

- Augmentations leves foram escolhidas para manter a estabilidade do treinamento.
- Caixas muito pequenas são filtradas para reduzir ruído.
- Uso de *learning rate* baixo e *gradient clipping* para evitar instabilidade.
- O ajuste automático do threshold melhora o equilíbrio entre *precision* e *recall*.

---

## 🚀 Tecnologias Utilizadas

- Python  
- Detectron2  
- Faster R-CNN  
- COCO Dataset Format  
- Roboflow  

---

## 📊 Métricas Avaliadas

- Precision  
- Recall  
- F1-score  
- Métricas COCO (AP, AP50, AP75, etc.)

---

## 📄 Licença

Este projeto é destinado a fins acadêmicos e de pesquisa.

