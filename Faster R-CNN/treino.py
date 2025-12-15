# ==========================================
# FASTER R-CNN 
# ==========================================
# O Faster R-CNN é um modelo de detecção de objetos que usa uma rede para
# gerar propostas de regiões e outra para classificar e ajustar as caixas.
# É mais rápido que as versões anteriores da família R-CNN e costuma ter
# boa precisão em tarefas de detecção.

# ==========================================
# FASTER R-CNN (pipeline)
# ==========================================
# Script completo: prepara dados, treina e avalia um Faster R-CNN.
#
# PASSO A PASSO (resumido):
# 1) Configura parâmetros e cria pasta de saída.
# 2) Baixa o dataset do Roboflow no formato COCO.
# 3) Define um mapper com augmentations moderados para treino.
# 4) Converte as anotações COCO para o formato esperado pelo Detectron2,
#    filtrando boxes muito pequenas e mantendo apenas imagens com anotações válidas.
# 5) Registra os datasets (train, valid, test) no DatasetCatalog/MetadataCatalog.
# 6) Carrega uma configuração pré-treinada do model zoo (ResNet-101-FPN)
#    e ajusta hiperparâmetros para estabilidade e desempenho.
# 7) Define um trainer customizado para usar o mapper com augmentations.
# 8) Executa o treinamento e salva checkpoints na pasta de saída.
# 9) Após treinar, testa vários thresholds de confiança para encontrar
#    o melhor balanço (precision/recall) medido por F1.
# 10) Reexecuta avaliação final com o melhor threshold e registra métricas COCO.
# 11) Gera visualizações: desenha predições em algumas imagens de teste.
# 12) Salva um relatório resumido (precision, recall, F1) e compacta os resultados.
#
# Observações rápidas:
# - O mapper aplica redimensionamento e flips leves para manter estabilidade.
# - Filtramos caixas muito pequenas para reduzir ruído nas anotações.
# - Usamos clipping de gradiente e learning rate baixo para evitar instabilidade.
# - A etapa de avaliação testa múltiplos thresholds para escolher o ideal.


import os
import shutil
import json
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from roboflow import Roboflow
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import Visualizer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils

# 🔧 Evitar OOM
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# -----------------------------
# CONFIGS
# -----------------------------
API_KEY = "6yKQfUumfFPyQzjUodnU"
WORKSPACE = "college-jcb9y"
PROJECT_NAME = "aircraft-damage-detection-a8z4k"
VERSION = 1
FORMAT = "coco"
RUN_NAME = "faster_rcnn_aircraft_optimized_85"
OUTPUT_DIR = f"./output/{RUN_NAME}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# BAIXAR DATASET
# -----------------------------
print(f"\n🚀 Baixando dataset {PROJECT_NAME} no formato COCO...")

rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT_NAME)
dataset = project.version(VERSION).download(FORMAT)
dataset_path = dataset.location

print(f"✅ Dataset baixado em: {dataset_path}")

# -----------------------------
# DATA AUGMENTATION MODERADO E ESTÁVEL
# -----------------------------
class CustomMapper:
    """Mapper com data augmentation moderado para estabilidade"""
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.augmentations = [
            T.ResizeShortestEdge(
                short_edge_length=(640, 672, 704, 736, 768, 800),
                max_size=1333,
                sample_style="choice"
            ),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
            T.RandomBrightness(0.9, 1.1),
            T.RandomContrast(0.9, 1.1),
        ]
        
    def __call__(self, dataset_dict):
        dataset_dict = dataset_dict.copy()
        image = utils.read_image(dataset_dict["file_name"], format="BGR")
        
        if self.is_train:
            aug_input = T.AugInput(image)
            transforms = T.AugmentationList(self.augmentations)(aug_input)
            image = aug_input.image
            
            annos = [
                utils.transform_instance_annotations(obj, transforms, image.shape[:2])
                for obj in dataset_dict.pop("annotations")
            ]
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
        else:
            dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))
            
        return dataset_dict

# -----------------------------
# CONVERTER DATASET PARA DETECTRON2
# -----------------------------
def get_aircraft_dicts(img_dir, ann_file):
    """Converte anotações COCO para formato Detectron2"""
    with open(ann_file) as f:
        coco_data = json.load(f)
    
    imgs = {img['id']: img for img in coco_data['images']}
    
    img_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)
    
    dataset_dicts = []
    for img_id, img_info in imgs.items():
        filename = os.path.join(img_dir, img_info['file_name'])
        if not os.path.exists(filename):
            continue
        
        record = {
            "file_name": filename,
            "image_id": img_id,
            "height": img_info['height'],
            "width": img_info['width'],
            "annotations": []
        }
        
        if img_id in img_anns:
            for ann in img_anns[img_id]:
                x, y, w, h = ann['bbox']
                # Filtrar boxes muito pequenas (ruído)
                if w < 10 or h < 10:
                    continue
                obj = {
                    "bbox": [x, y, x + w, y + h],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                record["annotations"].append(obj)
        
        # Só adicionar imagens com pelo menos 1 anotação válida
        if len(record["annotations"]) > 0:
            dataset_dicts.append(record)
    
    return dataset_dicts

# Registrar datasets
for split in ["train", "valid", "test"]:
    dataset_name = f"aircraft_{split}"
    img_dir = os.path.join(dataset_path, split)
    ann_file = os.path.join(dataset_path, split, "_annotations.coco.json")
    
    if os.path.exists(ann_file):
        DatasetCatalog.register(
            dataset_name, 
            lambda d=img_dir, a=ann_file: get_aircraft_dicts(d, a)
        )
        MetadataCatalog.get(dataset_name).set(thing_classes=["damage"])
        print(f"✅ Registrado: {dataset_name}")

# -----------------------------
# CONFIGURAR FASTER R-CNN OTIMIZADO
# -----------------------------
print("\n⚙️ Configurando Faster R-CNN com hiperparâmetros otimizados...")

cfg = get_cfg()
# Usar ResNet-101 (mais profundo) ao invés de ResNet-50
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")

# Datasets
cfg.DATASETS.TRAIN = ("aircraft_train",)
cfg.DATASETS.TEST = ("aircraft_valid",)

# Hiperparâmetros otimizados e estáveis
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.0003  # LR mais baixo para estabilidade
cfg.SOLVER.MAX_ITER = 18000
cfg.SOLVER.STEPS = (12000, 16000)
cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.WARMUP_FACTOR = 0.001
cfg.SOLVER.WARMUP_METHOD = "linear"

# Input - maior resolução para detalhes
cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333

# RPN (Region Proposal Network) - configurações estáveis
cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.RPN.POSITIVE_FRACTION = 0.5
cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.NMS_THRESH = 0.7
cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]  # Limites de IoU estáveis
cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]
cfg.MODEL.RPN.SMOOTH_L1_BETA = 0.1  # Suavizar loss de localização

# ROI Heads - configurações aprimoradas e estáveis
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.65  # Threshold inicial moderado
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 1.0  # Beta padrão para estabilidade

# Box Predictor
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 7
cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 2

# Anchor Generator - múltiplas escalas
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

# Regularização com gradient clipping mais agressivo
cfg.MODEL.BACKBONE.FREEZE_AT = 2
cfg.SOLVER.WEIGHT_DECAY = 0.0001
cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True
cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE = "norm"
cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE = 0.5  # Mais agressivo para evitar NaN
cfg.SOLVER.CLIP_GRADIENTS.NORM_TYPE = 2.0

# Output
cfg.OUTPUT_DIR = OUTPUT_DIR
cfg.SOLVER.CHECKPOINT_PERIOD = 2000
cfg.TEST.EVAL_PERIOD = 1000

print("✅ Configuração otimizada e estável pronta!")
print(f"   - Backbone: ResNet-101-FPN")
print(f"   - Score Threshold: {cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
print(f"   - Learning Rate: {cfg.SOLVER.BASE_LR}")
print(f"   - Max Iterations: {cfg.SOLVER.MAX_ITER}")
print(f"   - Gradient Clipping: {cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE}")
print(f"   - RPN Smooth L1 Beta: 0.1 (estabilidade)")

# -----------------------------
# TRAINER CUSTOMIZADO
# -----------------------------
class CustomTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(
            cfg, 
            mapper=CustomMapper(cfg, is_train=True)
        )

# -----------------------------
# TREINAMENTO
# -----------------------------
print("\n🏋️ Iniciando treinamento otimizado...\n")

trainer = CustomTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("\n✅ Treinamento concluído!")

# -----------------------------
# AVALIAÇÃO COM MÚLTIPLOS THRESHOLDS
# -----------------------------
print("\n📊 Avaliando modelo com diferentes thresholds...")

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

from detectron2.engine import DefaultPredictor

# Testar múltiplos thresholds para encontrar o melhor
thresholds = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
best_f1 = 0
best_threshold = 0.7
best_metrics = {}

val_dataset = get_aircraft_dicts(
    os.path.join(dataset_path, "valid"),
    os.path.join(dataset_path, "valid", "_annotations.coco.json")
)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

print("\n🔍 Testando thresholds de confiança:")
print("-" * 70)

for thresh in thresholds:
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    predictor = DefaultPredictor(cfg)
    
    TP = FP = FN = 0
    iou_threshold = 0.5
    
    for d in val_dataset:
        img = cv2.imread(d["file_name"])
        outputs = predictor(img)
        
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        pred_scores = outputs["instances"].scores.cpu().numpy()
        pred_boxes = pred_boxes[pred_scores >= thresh]
        
        gt_boxes = [ann["bbox"] for ann in d["annotations"]]
        matched_gt = set()
        
        for pred_box in pred_boxes:
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt_box in enumerate(gt_boxes):
                if i in matched_gt:
                    continue
                iou = calculate_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            if best_iou >= iou_threshold:
                TP += 1
                matched_gt.add(best_gt_idx)
            else:
                FP += 1
        
        FN += len(gt_boxes) - len(matched_gt)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Threshold {thresh:.2f} → P: {precision:.4f} | R: {recall:.4f} | F1: {f1_score:.4f}")
    
    if f1_score > best_f1:
        best_f1 = f1_score
        best_threshold = thresh
        best_metrics = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "TP": TP,
            "FP": FP,
            "FN": FN,
            "threshold": thresh
        }

print("-" * 70)
print(f"\n🎯 MELHOR CONFIGURAÇÃO:")
print(f"   Threshold: {best_threshold}")
print(f"   Precision: {best_metrics['precision']:.4f}")
print(f"   Recall:    {best_metrics['recall']:.4f}")
print(f"   F1-Score:  {best_metrics['f1_score']:.4f}")

# Salvar métricas
results = {
    "best_threshold": best_threshold,
    "best_metrics": best_metrics,
    "all_thresholds": {}
}

# Avaliação COCO com melhor threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = best_threshold
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("aircraft_valid", cfg, False, output_dir=OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "aircraft_valid")
coco_results = inference_on_dataset(predictor.model, val_loader, evaluator)

results["coco_metrics"] = coco_results

with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Métricas salvas em metrics.json")

# -----------------------------
# VISUALIZAÇÃO
# -----------------------------
print("\n🖼️ Gerando visualizações...")

test_dataset = get_aircraft_dicts(
    os.path.join(dataset_path, "test"),
    os.path.join(dataset_path, "test", "_annotations.coco.json")
)

aircraft_metadata = MetadataCatalog.get("aircraft_test")

for i, d in enumerate(test_dataset[:10]):
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    
    v = Visualizer(img[:, :, ::-1], metadata=aircraft_metadata, scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result_img = out.get_image()[:, :, ::-1]
    
    output_path = os.path.join(OUTPUT_DIR, f"prediction_{i}.jpg")
    cv2.imwrite(output_path, result_img)

print(f"✅ {len(test_dataset[:10])} imagens salvas")

# -----------------------------
# RELATÓRIO FINAL
# -----------------------------
print("\n" + "="*70)
print("📊 RELATÓRIO FINAL")
print("="*70)
print(f"Modelo: Faster R-CNN ResNet-101-FPN")
print(f"Threshold ótimo: {best_threshold}")
print(f"Precision: {best_metrics['precision']:.2%}")
print(f"Recall: {best_metrics['recall']:.2%}")
print(f"F1-Score: {best_metrics['f1_score']:.2%}")
print(f"\nResultados em: {OUTPUT_DIR}")
print("="*70)

# ZIP
zip_name = f"{RUN_NAME}_results"
shutil.make_archive(zip_name, 'zip', OUTPUT_DIR)
print(f"\n📦 ZIP criado: {zip_name}.zip")
print("\n🎉 Pipeline concluído!")
