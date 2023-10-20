from ultralytics import YOLO
import csv

model = YOLO('yolov8m.pt')

results = model.track(
	source='C:/Users/Rodrigo/Desktop/TCC/DataBase2/Insight-MVT_Annotation_Test/MVI_39311', 
	show=True, 
	save=True,
	save_dir='C:/Users/Rodrigo/Desktop/TCC/runs', 
	tracker='bytetrack.yaml',
	classes=(2,3,5,7)
	#0: person
	#1: bicycle
	#2: car
	#3: motorcycle
	#4: airplane
	#5: bus
	#6: train
	#7: truck
	#8: boat
)


























