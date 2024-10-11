from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model(source='Database2/Insight-MVT_Annotation_Train/MVI_20011', show=True, conf=0.4, save=True)
























