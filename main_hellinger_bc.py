import os  # Biblioteca para manipulação de diretórios
from ultralytics import YOLO  # Importa a biblioteca Ultralytics YOLO
import matplotlib.pyplot as plt  # Importa a biblioteca matplotlib para a criação de gráficos
import cv2  # Importa a biblioteca OpenCV para manipulação de imagens
import numpy as np  # Biblioteca para manipulação numérica

# Função para calcular o coeficiente de Bhattacharyya (BC)
def bhattacharyya_coefficient(hist1, hist2):
    bc = np.sum(np.sqrt(hist1 * hist2))
    return bc

# Função para calcular a distância de Hellinger usando o coeficiente de Bhattacharyya
def hellinger_distance(hist1, hist2):
    # Normaliza os histogramas
    hist1 = cv2.normalize(hist1, hist1).flatten()
    hist2 = cv2.normalize(hist2, hist2).flatten()

    # Calcula o coeficiente de Bhattacharyya
    bc = bhattacharyya_coefficient(hist1, hist2)
    
    # Calcula a distância de Hellinger usando a fórmula: H(P, Q) = sqrt(1 - BC(P, Q))
    hellinger = np.sqrt(1 - bc)
    return hellinger

# Função para gerar o histograma de uma ROI (Região de Interesse)
def calc_histogram(roi):
    color = ('b', 'g', 'r')  # Cores para canais RGB
    histograms = []
    
    for j, col in enumerate(color):  # Para cada canal de cor
        hist = cv2.calcHist([roi], [j], None, [256], [0, 256])  # Calcula o histograma
        histograms.append(hist)
    
    return histograms

# Função para salvar veículos detectados com enumeração
def save_vehicles_with_enumeration(image, boxes, frame_number, run_path):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)  # Obter coordenadas da bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Desenhar caixa em volta do veículo
        cv2.putText(image, f'Veiculo {i+1}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Numerar o veículo
    output_path = os.path.join(run_path, f'detected_frame_{frame_number}.jpg')  # Definir o caminho de salvamento
    cv2.imwrite(output_path, image)  # Salvar a imagem com as detecções

# Função para criar uma nova pasta de execução
def create_run_folder(base_folder='resultados'):
    run_number = 1
    while os.path.exists(os.path.join(base_folder, f'run{run_number}')):
        run_number += 1
    run_path = os.path.join(base_folder, f'run{run_number}')
    os.makedirs(run_path)
    return run_path

# Carregar o modelo YOLO
model = YOLO('yolov8m.pt')  # Carrega o modelo YOLOv8m

# Caminhos para as imagens de entrada
frame1_path = 'C:/Users/rodri/Desktop/TCC/imagensteste/img00090.jpg'
frame2_path = 'C:/Users/rodri/Desktop/TCC/imagensteste/img00091.jpg'

# Criar uma nova pasta para salvar os resultados
run_folder = create_run_folder()

# Realizar a detecção nos dois frames
results_frame1 = model(source=frame1_path, classes=(2, 5, 7))  # Carros, ônibus, caminhões
results_frame2 = model(source=frame2_path, classes=(2, 5, 7))

# Carregar as imagens
image1 = cv2.imread(frame1_path)  #formato BGR
image2 = cv2.imread(frame2_path)  #formato BGR

# Converter para RGB somente para exibição com Matplotlib (se necessário)
image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# Salvar e enumerar veículos detectados no frame 1
boxes_frame1 = results_frame1[0].boxes.xyxy  # Caixas do frame 1
save_vehicles_with_enumeration(image1, boxes_frame1, frame_number=1, run_path=run_folder)

# Salvar e enumerar veículos detectados no frame 2
boxes_frame2 = results_frame2[0].boxes.xyxy  # Caixas do frame 2
save_vehicles_with_enumeration(image2, boxes_frame2, frame_number=2, run_path=run_folder)

# Dicionário para armazenar os histogramas do frame 1
histograms_frame1 = {}

# Processar o frame 1
for i, result in enumerate(results_frame1):  # Para cada veículo detectado no frame 1
    boxes = result.boxes.xyxy
    for j, box in enumerate(boxes):  # Para cada caixa delimitadora
        x1, y1, x2, y2 = map(int, box)
        roi = image1[y1:y2, x1:x2]  # Região de interesse do veículo
        histograms_frame1[j] = calc_histogram(roi)  # Calcula e armazena o histograma
        plt.figure()
        for k, hist in enumerate(histograms_frame1[j]):  # Plota os histogramas
            plt.plot(hist, color=('b', 'g', 'r')[k])
        plt.title(f'Veículo {j+1} - Frame 1')
        plt.savefig(os.path.join(run_folder, f'hist_frame1_vehicle_{j+1}.png'))  # Salva o histograma

# Dicionário para armazenar os histogramas do frame 2
histograms_frame2 = {}

# Processar o frame 2
for i, result in enumerate(results_frame2):  # Para cada veículo detectado no frame 2
    boxes = result.boxes.xyxy
    for j, box in enumerate(boxes):  # Para cada caixa delimitadora
        x1, y1, x2, y2 = map(int, box)
        roi = image2[y1:y2, x1:x2]  # Região de interesse do veículo
        histograms_frame2[j] = calc_histogram(roi)  # Calcula e armazena o histograma
        plt.figure()
        for k, hist in enumerate(histograms_frame2[j]):  # Plota os histogramas
            plt.plot(hist, color=('b', 'g', 'r')[k])
        plt.title(f'Veículo {j+1} - Frame 2')
        plt.savefig(os.path.join(run_folder, f'hist_frame2_vehicle_{j+1}.png'))  # Salva o histograma

# Comparar histogramas entre o frame 1 e o frame 2
for idx1, hist1 in histograms_frame1.items():
    for idx2, hist2 in histograms_frame2.items():
        distances = []
        for ch in range(3):  # Comparar para cada canal (RGB)
            dist = hellinger_distance(hist1[ch], hist2[ch])
            distances.append(dist)
        avg_distance = np.mean(distances)  # Calcula a média das distâncias dos canais
        print(f'Distância média (Hellinger) entre o veículo {idx1+1} no Frame 1 e o veículo {idx2+1} no Frame 2: {avg_distance}')
