import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Função para criar nova pasta de execução
def criar_pasta_run(base_folder='resultados'):
    run_number = 1
    while os.path.exists(os.path.join(base_folder, f'run{run_number}')):
        run_number += 1
    run_path = os.path.join(base_folder, f'run{run_number}')
    os.makedirs(run_path)
    print(f"Pasta de execução criada: {run_path}")
    return run_path

# Função para calcular coeficiente de Bhattacharyya
def coeficiente_de_bhattacharyya(hist1, hist2):
    bc = np.sum(np.sqrt(hist1 * hist2))
    bc = np.clip(bc, 0, 1)
    return bc

# Função para calcular a distância de Hellinger
def distancia_de_hellinger(hist1, hist2):
    hist1 = hist1 / (np.sum(hist1) + 1e-6)
    hist2 = hist2 / (np.sum(hist2) + 1e-6)
    bc = coeficiente_de_bhattacharyya(hist1, hist2)
    hellinger = np.sqrt(1 - bc)
    return hellinger

# Função para gerar histogramas dos veículos detectados em um quadro
def gerar_histogramas_de_veiculos_de_um_quadro(img, deteccoes):
    histogramas_com_coordenadas = []
    if deteccoes and hasattr(deteccoes[0], 'boxes'):
        for det in deteccoes[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            veiculo = img[y1:y2, x1:x2]
            hist_r = cv2.calcHist([veiculo], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([veiculo], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([veiculo], [2], None, [256], [0, 256])
            histogramas_com_coordenadas.append(([hist_r, hist_g, hist_b], (x1, y1, x2, y2)))
    return histogramas_com_coordenadas

# Função para salvar histogramas
def salva_histograma_atual_e_seu_semelhante(histograma_atual, histograma_semelhante, numero_veiculo, numero_quadro, pasta_run):
    pasta_quadro = os.path.join(pasta_run, f"quadro_{numero_quadro}")
    pasta_histograma = os.path.join(pasta_quadro, f"veiculo_{numero_veiculo}")
    os.makedirs(pasta_histograma, exist_ok=True)
    
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_atual[canal], color=cor, label=f"Atual - {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Atual - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma, "atual.png"))
    plt.close()

    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_semelhante[canal], color=cor, label=f"Semelhante - {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Semelhante - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma, "semelhante.png"))
    plt.close()

# Função para desenhar retângulos e números nos veículos detectados
def desenhar_veiculo(img, coordenadas, numero_veiculo, cor):
    x1, y1, x2, y2 = coordenadas
    cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
    cv2.putText(img, f"{numero_veiculo}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

# Função principal para processar quadros
def processar_quadros(model, quadros, pasta_quadros, pasta_run):
    veiculos_rastreio = {}  # Mapeia IDs de veículos para suas coordenadas, histogramas e quadros ausentes
    contador_id = 1  # Contador para novos veículos
    max_quadros_ausentes = 5  # Número máximo de quadros permitidos sem detecção

    for i, quadro in enumerate(quadros):
        img_atual = cv2.imread(os.path.join(pasta_quadros, quadro))
        deteccoes = model(source=os.path.join(pasta_quadros, quadro), classes=(2, 5, 7))
        histogramas_atuais = gerar_histogramas_de_veiculos_de_um_quadro(img_atual, deteccoes)

        # Atualizar veículos rastreados
        novos_veiculos = {}
        for histograma_atual, coord_atual in histogramas_atuais:
            menor_distancia = float('inf')
            id_rastreado = None
            histograma_semelhante = None  # Armazena o histograma mais semelhante

            for id_veiculo, (histograma_rastreado, coordenada_rasteada, quadros_ausentes) in veiculos_rastreio.items():
                distancia_r = distancia_de_hellinger(histograma_atual[0], histograma_rastreado[0])
                distancia_g = distancia_de_hellinger(histograma_atual[1], histograma_rastreado[1])
                distancia_b = distancia_de_hellinger(histograma_atual[2], histograma_rastreado[2])
                distancia_media = (distancia_r + distancia_g + distancia_b) / 3

                if distancia_media < menor_distancia:
                    menor_distancia = distancia_media
                    id_rastreado = id_veiculo
                    histograma_semelhante = histograma_rastreado

            if menor_distancia < 0.4:  # Limite de similaridade
                novos_veiculos[id_rastreado] = (histograma_atual, coord_atual, 0)  # Reseta contagem de ausência
            else:
                novos_veiculos[contador_id] = (histograma_atual, coord_atual, 0)
                contador_id += 1

            if histograma_semelhante is not None:
                salva_histograma_atual_e_seu_semelhante(histograma_atual, histograma_semelhante, id_rastreado, i, pasta_run)

        # Incrementar contagem de ausência para veículos não detectados
        for id_veiculo, (histograma, coord, quadros_ausentes) in veiculos_rastreio.items():
            if id_veiculo not in novos_veiculos:
                if quadros_ausentes + 1 < max_quadros_ausentes:
                    novos_veiculos[id_veiculo] = (histograma, coord, quadros_ausentes + 1)

        veiculos_rastreio = novos_veiculos

        # Desenhar os veículos na imagem
        for id_veiculo, (hist, coord, _) in veiculos_rastreio.items():
            desenhar_veiculo(img_atual, coord, id_veiculo, (0, 255, 0))

        cv2.imwrite(os.path.join(pasta_run, f"quadro_{i+1}.jpg"), img_atual)

# Função principal
def main():
    pasta_quadros = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_noturno_intenso'
    quadros = sorted([f for f in os.listdir(pasta_quadros) if f.endswith('.jpg')])
    quadros_pares = [quadro for quadro in quadros if int(quadro.split('img')[1].split('.')[0]) % 2 == 0]
    model = YOLO('yolov8m.pt')
    pasta_run = criar_pasta_run()

    processar_quadros(model, quadros_pares, pasta_quadros, pasta_run)

if __name__ == "__main__":
    main()
