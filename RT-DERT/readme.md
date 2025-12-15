# RT-DETR-X – Pipeline Completo de Detecção de Objetos

O Código treino.py contém um **pipeline completo para treinamento, avaliação e visualização de resultados utilizando o modelo RT-DETR-X**, incluindo download e correção de dataset, ajuste automático de rótulos, geração de gráficos, exportação dos resultados e criação de uma predição final para validação visual.

O **RT-DETR-X** é um detector baseado em *Transformers*, que utiliza **atenção direta para localizar objetos**, dispensando o uso de *region proposals*, o que o torna mais simples e eficiente em determinados cenários.

---

## 📌 Visão Geral do RT-DETR-X

Diferente de detectores baseados em propostas (como Faster R-CNN), o **RT-DETR-X**:

- Localiza objetos diretamente por mecanismos de atenção.
- Apresenta bom desempenho em tempo real.
- É sensível a *augmentations* agressivos, exigindo configurações mais suaves.

---

## 🔁 Pipeline do Projeto

O script implementa todas as etapas necessárias para executar o fluxo completo do RT-DETR-X.

### Passo a Passo

1. **Configuração inicial**
   - Carregamento das configurações básicas.
   - Definição do nome do experimento (*run name*).

2. **Download do dataset**
   - Dataset baixado via **Roboflow** no formato **YOLO**.
   - Organização automática da estrutura de diretórios.

3. **Correção de rótulos**
   - Conversão automática de rótulos incorretos:
     - Classe `"1"` → Classe `"0"`
   - Garantia de consistência para treinamento com classe única.

4. **Geração do arquivo YAML**
   - Criação automática do arquivo `data.yaml` do dataset YOLO.

5. **Treinamento do modelo**
   - Carregamento do modelo **RT-DETR-X pré-treinado**.
   - Ajuste de:
     - *Learning rate* baixo
     - Otimizador
     - *Augmentations* leves  
   - Evita transformações agressivas que prejudicam a convergência do DETR.

6. **Salvamento de métricas**
   - Histórico de treinamento salvo em:
     ```
     runs/detect/<RUN_NAME>
     ```

7. **Geração de gráficos**
   - Leitura automática do arquivo CSV de resultados.
   - Geração de gráficos de:
     - Loss
     - Precision
     - Recall
     - mAP@50

8. **Exportação dos resultados**
   - Compactação completa da pasta do experimento em um arquivo `.zip`.

9. **Predição final**
   - Execução de uma predição de teste.
   - Salvamento da imagem anotada para conferência visual do modelo.

---

## 🧪 Observações Importantes

- O RT-DETR-X utiliza **atenção direta**, sem *region proposals*.
- O modelo apresenta melhor estabilidade com:
  - *Learning rate* baixo
  - *Augmentations* suaves
- *Mosaic*, *shear* e *copy-paste* agressivos tendem a prejudicar a convergência.
- O script tenta adaptar automaticamente os nomes das colunas do CSV.
- Ao final do pipeline, são gerados:
  - Gráficos
  - Arquivo ZIP do experimento
  - Uma predição visual para validação.

---

## 🚀 Tecnologias Utilizadas

- Python  
- RT-DETR-X  
- YOLO Dataset Format  
- Roboflow  
- Matplotlib / Pandas (para análise e gráficos)

---

## 📊 Métricas Avaliadas

- Loss  
- Precision  
- Recall  
- mAP@50  

---

## 📄 Licença

Este projeto é destinado a fins acadêmicos e de pesquisa.

