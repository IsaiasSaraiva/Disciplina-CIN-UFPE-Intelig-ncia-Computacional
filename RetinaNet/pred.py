# ==========================================
# CALCULAR PRECISION/RECALL/F1 - MODELO JÁ TREINADO
# ==========================================

import os
import json
import cv2
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

# -----------------------------
# CONFIGS
# -----------------------------
MODEL_PATH = "./output/retinanet_aircraft_damage_140ep/model_final.pth"
DATASET_PATH = "/home/debora/isaias/RetinaNet/aircraft-damage-detection-1"  # Ajuste se necessário
OUTPUT_DIR = "./output/retinanet_aircraft_damage_140ep"
IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

print(f"🔍 Calculando Precision, Recall e F1 para modelo treinado...")

# -----------------------------
# CARREGAR DATASET
# -----------------------------
def get_aircraft_dicts(img_dir, ann_file):
    """Converte anotações COCO para formato Detectron2"""
    with open(ann_file) as f:
        coco_data = json.load(f)
    
    imgs = {img['id']: img for img in coco_data['images']}
    
    # Agrupar anotações por imagem
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
                obj = {
                    "bbox": [x, y, x + w, y + h],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": 0,
                }
                record["annotations"].append(obj)
        
        dataset_dicts.append(record)
    
    return dataset_dicts

# Registrar dataset de validação
val_dir = os.path.join(DATASET_PATH, "valid")
val_ann = os.path.join(val_dir, "_annotations.coco.json")

if os.path.exists(val_ann):
    DatasetCatalog.register("aircraft_valid_eval", lambda: get_aircraft_dicts(val_dir, val_ann))
    MetadataCatalog.get("aircraft_valid_eval").set(thing_classes=["damage"])
    print(f"✅ Dataset carregado: {val_ann}")
else:
    print(f"❌ ERRO: Arquivo não encontrado: {val_ann}")
    exit()

# -----------------------------
# CARREGAR MODELO
# -----------------------------
print("\n⬇️ Carregando modelo treinado...")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = MODEL_PATH
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = CONFIDENCE_THRESHOLD

predictor = DefaultPredictor(cfg)
print(f"✅ Modelo carregado: {MODEL_PATH}")

# -----------------------------
# CALCULAR IoU
# -----------------------------
def calculate_iou(box1, box2):
    """Calcula IoU entre duas bounding boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

# -----------------------------
# AVALIAR MODELO
# -----------------------------
print(f"\n📊 Avaliando com IoU >= {IOU_THRESHOLD}, Confidence >= {CONFIDENCE_THRESHOLD}...")

val_dataset = get_aircraft_dicts(val_dir, val_ann)

TP = 0  # True Positives
FP = 0  # False Positives
FN = 0  # False Negatives

total_images = len(val_dataset)
print(f"Total de imagens: {total_images}\n")

for idx, d in enumerate(val_dataset):
    if (idx + 1) % 50 == 0:
        print(f"Processando: {idx + 1}/{total_images} imagens...")
    
    img = cv2.imread(d["file_name"])
    outputs = predictor(img)
    
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    pred_scores = outputs["instances"].scores.cpu().numpy()
    
    # Filtrar por confiança
    pred_boxes = pred_boxes[pred_scores >= CONFIDENCE_THRESHOLD]
    
    gt_boxes = [ann["bbox"] for ann in d["annotations"]]
    
    # Matching: para cada predição, encontrar melhor ground truth
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
        
        if best_iou >= IOU_THRESHOLD:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1
    
    # Ground truths não detectados
    FN += len(gt_boxes) - len(matched_gt)

print(f"✅ Avaliação concluída!\n")

# -----------------------------
# CALCULAR MÉTRICAS
# -----------------------------
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print("=" * 60)
print(f"📊 MÉTRICAS CALCULADAS (IoU >= {IOU_THRESHOLD}):")
print("=" * 60)
print(f"  True Positives (TP):   {TP}")
print(f"  False Positives (FP):  {FP}")
print(f"  False Negatives (FN):  {FN}")
print(f"\n  Precision:  {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:     {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:   {f1_score:.4f} ({f1_score*100:.2f}%)")
print("=" * 60)

# -----------------------------
# SALVAR RESULTADOS
# -----------------------------
results = {
    "model": MODEL_PATH,
    "iou_threshold": IOU_THRESHOLD,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "metrics": {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN)
    }
}

output_file = os.path.join(OUTPUT_DIR, "precision_recall_metrics.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Métricas salvas em: {output_file}")
print("\n🎉 Análise concluída!")