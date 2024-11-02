import os  # Biblioteca para manipulação de diretórios e arquivos
from ultralytics import YOLO  # Importa a biblioteca Ultralytics para uso do modelo YOLO
import matplotlib.pyplot as plt  # Biblioteca para criação de gráficos, usada para plotar histogramas
import cv2  # Biblioteca OpenCV para manipulação de imagens e vídeos
import numpy as np  # Biblioteca para operações numéricas com arrays

# Função que calcula o coeficiente de Bhattacharyya (BC) entre dois histogramas
def bhattacharyya_coefficient(hist1, hist2):
    bc = np.sum(np.sqrt(hist1 * hist2))  # Soma a raiz dos produtos correspondentes dos histogramas
    bc = np.clip(bc, 0, 1)  # Garante que o valor esteja entre 0 e 1
    return bc

# Função que calcula a distância de Hellinger entre dois histogramas usando o coeficiente de Bhattacharyya
def hellinger_distance(hist1, hist2):
    hist1 = hist1 / (np.sum(hist1) + 1e-6)  # Normaliza o histograma 1
    hist2 = hist2 / (np.sum(hist2) + 1e-6)  # Normaliza o histograma 2
    bc = bhattacharyya_coefficient(hist1, hist2)  # Calcula o coeficiente de Bhattacharyya entre os histogramas
    hellinger = np.sqrt(1 - bc)  # Calcula a distância de Hellinger
    return hellinger

# Função que gera histogramas para uma região de interesse (ROI) em uma imagem
def generate_histogram(roi):
    color = ('b', 'g', 'r')  # Canais de cor: azul, verde e vermelho
    histograms = []
    for j, col in enumerate(color):
        hist = cv2.calcHist([roi], [j], None, [256], [0, 256])  # Calcula o histograma para cada canal de cor
        histograms.append(hist)

    return histograms

# Função que salva uma imagem com retângulos e numeração nos veículos detectados
def save_vehicles_with_enumeration(image, boxes, enumeration_map, frame_number, run_path):
    for idx, box in enumerate(boxes):
        vehicle_num = enumeration_map[idx] + 1  # Atribui o número de identificação do veículo
        x1, y1, x2, y2 = map(int, box)  # Converte as coordenadas da caixa delimitadora em inteiros
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Desenha um retângulo ao redor do veículo
        cv2.putText(image, f'V{vehicle_num}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)  # Adiciona o texto
    output_path = os.path.join(run_path, f'detected_frame_{frame_number}.jpg')  # Define o caminho de saída
    cv2.imwrite(output_path, image)  # Salva a imagem

# Função que cria uma nova pasta de execução numerada
def create_run_folder(base_folder='resultados'):
    run_number = 1
    while os.path.exists(os.path.join(base_folder, f'run{run_number}')):  # Incrementa o número da pasta até encontrar um nome disponível
        run_number += 1
    run_path = os.path.join(base_folder, f'run{run_number}')
    os.makedirs(run_path)  # Cria a nova pasta
    return run_path

# Função que salva os resultados das distâncias de Hellinger em um arquivo de texto
def save_hellinger_results(run_path, results):
    output_file = os.path.join(run_path, 'results.txt')  # Define o nome do arquivo de saída
    with open(output_file, 'w') as file:
        for result in results:
            file.write(result + '\n')  # Escreve cada resultado em uma linha no arquivo

# Define o caminho da pasta que contém as imagens
folder_path = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'

# Lista e ordena os arquivos de imagem, selecionando apenas os frames pares
all_frames = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])
even_frames = [frame for frame in all_frames if int(frame.split('img')[1].split('.')[0]) % 2 == 0]

# Carrega o modelo YOLO para detecção
model = YOLO('yolov8m.pt')

# Cria a pasta principal para armazenar os resultados da execução
run_folder = create_run_folder()

# Loop para processar e comparar pares consecutivos de frames
for i in range(len(even_frames) - 1):
    frame1_path = os.path.join(folder_path, even_frames[i])
    frame2_path = os.path.join(folder_path, even_frames[i + 1])
    
    # Cria uma subpasta para armazenar os resultados de cada par de frames
    pair_folder = os.path.join(run_folder, f'{even_frames[i].split(".")[0]}_{even_frames[i+1].split(".")[0]}')
    os.makedirs(pair_folder, exist_ok=True)

    # Detecta veículos em ambos os frames usando o modelo YOLO, considerando apenas classes de veículos
    results_frame1 = model(source=frame1_path, classes=(2, 5, 7))  # Classes: carro, ônibus, caminhão
    results_frame2 = model(source=frame2_path, classes=(2, 5, 7))

    # Carrega as imagens
    image1 = cv2.imread(frame1_path)
    image2 = cv2.imread(frame2_path)

    # Dicionários para armazenar histogramas dos veículos detectados
    histograms_frame1 = {}
    histograms_frame2 = {}

    # Gera e salva histogramas para cada veículo detectado no frame 1
    for j, box in enumerate(results_frame1[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        roi = image1[y1:y2, x1:x2]  # Extrai a região do veículo
        histograms_frame1[j] = generate_histogram(roi)  # Gera o histograma
        
        # Plota e salva o histograma para cada veículo
        plt.figure()
        for k, hist in enumerate(histograms_frame1[j]):
            plt.plot(hist, color=('b', 'g', 'r')[k])
        plt.title(f'Veículo {j+1} - Frame 1')
        plt.savefig(os.path.join(pair_folder, f'hist_frame1_vehicle_{j+1}.png'))
        plt.close()

    # Gera e salva histogramas para cada veículo detectado no frame 2
    for j, box in enumerate(results_frame2[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        roi = image2[y1:y2, x1:x2]  # Extrai a região do veículo
        histograms_frame2[j] = generate_histogram(roi)  # Gera o histograma
        
        # Plota e salva o histograma para cada veículo
        plt.figure()
        for k, hist in enumerate(histograms_frame2[j]):
            plt.plot(hist, color=('b', 'g', 'r')[k])
        plt.title(f'Veículo {j+1} - Frame 2')
        plt.savefig(os.path.join(pair_folder, f'hist_frame2_vehicle_{j+1}.png'))
        plt.close()

    # Calcula e armazena as distâncias de Hellinger entre veículos do frame 1 e do frame 2
    hellinger_results = []
    for idx1, hist1 in histograms_frame1.items():
        for idx2, hist2 in histograms_frame2.items():
            distances = [hellinger_distance(hist1[channel], hist2[channel]) for channel in range(3)]  # Distância de Hellinger para cada canal de cor
            avg_distance = np.mean(distances)  # Média das distâncias entre os canais
            result = f'Distância de Hellinger entre o veículo {idx1+1} no Frame 1 e o veículo {idx2+1} no Frame 2: {avg_distance}'
            hellinger_results.append(result)

    # Salva os resultados das distâncias em um arquivo de texto
    save_hellinger_results(pair_folder, hellinger_results)

    # Define a correspondência de veículos com base na menor distância média de Hellinger
    enumeration_map_frame2 = [-1] * len(histograms_frame2)
    for idx1, hist1 in histograms_frame1.items():
        min_distance = float('inf')
        min_index = -1
        for idx2, hist2 in histograms_frame2.items():
            distances = [hellinger_distance(hist1[channel], hist2[channel]) for channel in range(3)]
            avg_distance = np.mean(distances)
            if avg_distance < min_distance:  # Atualiza a menor distância e o índice correspondente
                min_distance = avg_distance
                min_index = idx2
        enumeration_map_frame2[min_index] = idx1  # Mapeia o veículo correspondente com a menor distância

    # Salva as imagens com os veículos numerados para visualização
    save_vehicles_with_enumeration(image1, results_frame1[0].boxes.xyxy, list(range(len(results_frame1[0].boxes.xyxy))), frame_number=1, run_path=pair_folder)
    save_vehicles_with_enumeration(image2, results_frame2[0].boxes.xyxy, enumeration_map_frame2, frame_number=2, run_path=pair_folder)
