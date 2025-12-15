# Disciplina Inteligência Computacional CIN-UFPE
Repositório referente ao projeto da disciplina de Inteligência Computacional do programa de pós graduação Scrictu Senso em computação do Centro de Informática da Universidade Federal de Pernambuco.

### Trabalho Final Inteligência Computacional

<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSX26f4IHVwUDcuLfH6zZRooqi7OVHeSDq1cA&s" width="300"/>

O aprendizado profundo é uma área-chave por trás de diferentes tecnologias de visão computacional, reconhecimento de imagem, reconhecimento de fala, processamento de linguagem natural, processamento de texto, recuperação de informações e processamento de informações multimodais. O aprendizado profundo está recebendo muita atenção da academia e da indústria. O curso começou em 2016 e é um mergulho profundo nas arquiteturas de aprendizagem profunda, com foco no aprendizado de modelos de ponta a ponta para essas tarefas. Temos uma abordagem teórica e prática para permitir que os alunos aprendam a implementar, treinar e depurar seus projetos.

https://deeplearning.cin.ufpe.br/



**Alunos: Isaias Abner Lima Saraiva, Sérgio Santana**

## Fonte dos Dados e Estrutura dos Dados

Para o treinamento dos modelos, será utilizado o *dataset* **"Aircraft Damage Detection"** (obtido via [Roboflow Universe](https://universe.roboflow.com/college-jcb9y/aircraft-damage-detection-a8z4k)).

### Detecção de Defeitos

O *dataset* original é rotulado para **Detecção de Objetos** (localização da falha via *bounding boxes*). O foco é treinar modelos para **detecção de defeitos em aeronaves**, ou seja, identificar a presença e a localização das anomalias.

| Categoria de Classe | Condição Original do Dataset | Rótulo de Detecção |
| :--- | :--- | :--- |
| **DEFECTO** (Classe 1) | Imagens que possuem **anotações de defeitos** (presença de *bounding boxes*). | A imagem **contém** um defeito. |
| **NORMAL** (Classe 0) | Imagens que **não possuem anotações** (imagens de fundo limpo). | A imagem está **sem falhas**. |

## Organização do Repositório

Este repositório está estruturado da seguinte forma:

- **Faster R-CNN**: contém o código-fonte utilizado para o treinamento do modelo Faster R-CNN.
- **RT-DETR**: inclui o arquivo `treino.py` e demais scripts necessários para o treinamento da arquitetura RT-DETR.
- **RetinaNet**: pasta destinada ao código de treinamento do modelo RetinaNet.
- **YOLO10s**: contém os arquivos referentes ao treinamento do modelo YOLOv10s.
- **YOLO8n**: código-fonte para o treinamento do YOLOv8n.
- **YOLO8s_V1 / YOLO8s_V2 / YOLO8s_V3**: versões alternativas de configuração e treinamento do modelo YOLOv8s.
Os resultados de treinamento, validação métricas de desempenho, encontram-se na pasta no drive:
https://drive.google.com/drive/folders/1Y1GMi5mJymhJywffSkHbCVtkZt8WDiuH?usp=drive_link

---

## Como Executar

Para utilizar qualquer um dos modelos:

1. Clone este repositório:

   ```bash
   git clone <URL-do-repositório>
   ```
2. Crie um ambiente virtual.

Instale as dependências listadas no arquivo requirements.txt:
   ```bash
    pip install -r requirements.txt
