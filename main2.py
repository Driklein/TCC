import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Função para calcular coeficiente de Bhattacharyya
def bhattacharyya_coefficient(hist1, hist2):
    bc = np.sum(np.sqrt(hist1 * hist2))
    bc = np.clip(bc, 0, 1)
    return bc

# Função para calcular a distância de Hellinger
def hellinger_distance(hist1, hist2):
    hist1 = hist1 / (np.sum(hist1) + 1e-6)
    hist2 = hist2 / (np.sum(hist2) + 1e-6)
    bc = bhattacharyya_coefficient(hist1, hist2)
    hellinger = np.sqrt(1 - bc)
    return hellinger

# Função para gerar histogramas
def generate_histogram(roi):
    color = ('b', 'g', 'r')
    histograms = []
    for j, col in enumerate(color):
        hist = cv2.calcHist([roi], [j], None, [256], [0, 256])
        histograms.append(hist)
    return histograms

# Função para salvar uma imagem com veículos enumerados
def save_vehicles_with_enumeration(image, boxes, enumeration_map, frame_name, output_folder):
    for idx, box in enumerate(boxes):
        vehicle_num = enumeration_map[idx] + 1
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(image, f'V{vehicle_num}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    output_path = os.path.join(output_folder, f'{frame_name}.jpg')
    cv2.imwrite(output_path, image)
    print(f"Imagem com enumeração salva em: {output_path}")

# Função para criar nova pasta de execução
def create_run_folder(base_folder='resultados'):
    run_number = 1
    while os.path.exists(os.path.join(base_folder, f'run{run_number}')):
        run_number += 1
    run_path = os.path.join(base_folder, f'run{run_number}')
    os.makedirs(run_path)
    print(f"Pasta de execução criada: {run_path}")
    return run_path

# Função para salvar resultados de Hellinger
def save_hellinger_results(run_path, results):
    output_file = os.path.join(run_path, 'results.txt')
    with open(output_file, 'w') as file:
        for result in results:
            file.write(result + '\n')
    print(f"Resultados de Hellinger salvos em: {output_file}")

# Definição do caminho da pasta de imagens
folder_path = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'
all_frames = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
even_frames = [frame for frame in all_frames if int(frame.split('img')[1].split('.')[0]) % 2 == 0]

# Carrega o modelo YOLO
model = YOLO('yolov8m.pt')
run_folder = create_run_folder()

# Cria uma pasta para salvar todos os frames com detecções
all_frames_folder = os.path.join(run_folder, 'all_frames_with_detections')
os.makedirs(all_frames_folder, exist_ok=True)
print(f"Pasta para todos os frames com detecções criada: {all_frames_folder}")

for i in range(len(even_frames) - 1):
    frame1_path = os.path.join(folder_path, even_frames[i])
    frame2_path = os.path.join(folder_path, even_frames[i + 1])
    
    pair_folder = os.path.join(run_folder, f'{even_frames[i].split(".")[0]}_{even_frames[i+1].split(".")[0]}')
    os.makedirs(pair_folder, exist_ok=True)
    print(f"Processando frames: {even_frames[i]} e {even_frames[i + 1]}")

    results_frame1 = model(source=frame1_path, classes=(2, 5, 7))
    results_frame2 = model(source=frame2_path, classes=(2, 5, 7))

    image1 = cv2.imread(frame1_path)
    image2 = cv2.imread(frame2_path)

    histograms_frame1 = {}
    histograms_frame2 = {}

    # Processa veículos do frame 1
    for j, box in enumerate(results_frame1[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        roi = image1[y1:y2, x1:x2]
        histograms_frame1[j] = generate_histogram(roi)
        
        plt.figure()
        for k, hist in enumerate(histograms_frame1[j]):
            plt.plot(hist, color=('b', 'g', 'r')[k])
        plt.title(f'Veículo {j+1} - Frame 1')
        plt.savefig(os.path.join(pair_folder, f'hist_frame1_vehicle_{j+1}.png'))
        plt.close()
        print(f"Histograma salvo para veículo {j+1} no Frame 1.")

    # Processa veículos do frame 2
    for j, box in enumerate(results_frame2[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        roi = image2[y1:y2, x1:x2]
        histograms_frame2[j] = generate_histogram(roi)
        
        plt.figure()
        for k, hist in enumerate(histograms_frame2[j]):
            plt.plot(hist, color=('b', 'g', 'r')[k])
        plt.title(f'Veículo {j+1} - Frame 2')
        plt.savefig(os.path.join(pair_folder, f'hist_frame2_vehicle_{j+1}.png'))
        plt.close()
        print(f"Histograma salvo para veículo {j+1} no Frame 2.")

    # Calcula distâncias de Hellinger
    hellinger_results = []
    for idx1, hist1 in histograms_frame1.items():
        for idx2, hist2 in histograms_frame2.items():
            distances = [hellinger_distance(hist1[channel], hist2[channel]) for channel in range(3)]
            avg_distance = np.mean(distances)
            result = f'Distância de Hellinger entre o veículo {idx1+1} no Frame 1 e o veículo {idx2+1} no Frame 2: {avg_distance}'
            hellinger_results.append(result)
            print(result)

    # Salva resultados de Hellinger
    save_hellinger_results(pair_folder, hellinger_results)

    # Enumera veículos e salva imagem
    enumeration_map_frame2 = [-1] * len(histograms_frame2)
    for idx1, hist1 in histograms_frame1.items():
        min_distance = float('inf')
        min_index = -1
        for idx2, hist2 in histograms_frame2.items():
            distances = [hellinger_distance(hist1[channel], hist2[channel]) for channel in range(3)]
            avg_distance = np.mean(distances)
            if avg_distance < min_distance:
                min_distance = avg_distance
                min_index = idx2
        enumeration_map_frame2[min_index] = idx1

    # Salva imagens com enumeração dos veículos em ambas as pastas
    save_vehicles_with_enumeration(image1, results_frame1[0].boxes.xyxy, list(range(len(results_frame1[0].boxes.xyxy))), frame_name=f'detected_frame_{even_frames[i]}', output_folder=pair_folder)
    save_vehicles_with_enumeration(image2, results_frame2[0].boxes.xyxy, enumeration_map_frame2, frame_name=f'detected_frame_{even_frames[i+1]}', output_folder=pair_folder)

    # Salva também cada frame na pasta 'all_frames_with_detections'
    save_vehicles_with_enumeration(image1, results_frame1[0].boxes.xyxy, list(range(len(results_frame1[0].boxes.xyxy))), frame_name=f'detected_frame_{even_frames[i]}', output_folder=all_frames_folder)
    save_vehicles_with_enumeration(image2, results_frame2[0].boxes.xyxy, enumeration_map_frame2, frame_name=f'detected_frame_{even_frames[i+1]}', output_folder=all_frames_folder)
