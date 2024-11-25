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

def salva_histograma_atual_e_seu_semelhante(histograma_atual, histograma_semelhante, numero_veiculo, numero_quadro, pasta_run):
    # Criar pasta para o quadro atual
    pasta_quadro_atual = os.path.join(pasta_run, f"quadro_{numero_quadro}")
    # Criar pasta específica para os histogramas do veículo
    pasta_histograma = os.path.join(pasta_quadro_atual, f"histogramas_semelhantes_{numero_veiculo}")
    os.makedirs(pasta_histograma, exist_ok=True)
    
    # Salvar histograma atual
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_atual[canal], color=cor, label=f"Atual - {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Atual - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma, f"atual_{numero_veiculo}.png"))
    plt.close()

    # Salvar histograma semelhante
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_semelhante[canal], color=cor, label=f"Semelhante - {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Semelhante - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma, f"semelhante_{numero_veiculo}.png"))
    plt.close()

def rastrear_e_numerar(histogramas_atual, histogramas_anterior, rastreamento, pasta_run):
    novo_rastreamento = {}
    usado = set()

    for id_anterior, (histograma_anterior, coordenada_anterior) in rastreamento.items():
        menor_distancia = float('inf')
        id_associado = None
        histograma_semelhante = None

        for i, (histograma_atual, coordenada_atual) in enumerate(histogramas_atual):
            if i in usado:
                continue
            distancia_r = distancia_de_hellinger(histograma_anterior[0], histograma_atual[0])
            distancia_g = distancia_de_hellinger(histograma_anterior[1], histograma_atual[1])
            distancia_b = distancia_de_hellinger(histograma_anterior[2], histograma_atual[2])
            distancia_media = (distancia_r + distancia_g + distancia_b) / 3

            if distancia_media < menor_distancia:
                menor_distancia = distancia_media
                id_associado = i
                histograma_semelhante = histograma_atual

        if id_associado is not None:
            usado.add(id_associado)
            novo_rastreamento[id_anterior] = (histogramas_atual[id_associado][0], histogramas_atual[id_associado][1])

            # Salvar histogramas apenas se um semelhante foi encontrado
            if histograma_semelhante is not None:
                salva_histograma_atual_e_seu_semelhante(
                    histogramas_atual[id_associado][0], histograma_semelhante, id_associado, id_anterior, pasta_run
                )

    # Adicionar novos objetos não associados
    proximo_id = max(rastreamento.keys(), default=0) + 1
    for i, (histograma_atual, coordenada_atual) in enumerate(histogramas_atual):
        if i not in usado:
            novo_rastreamento[proximo_id] = (histograma_atual, coordenada_atual)
            proximo_id += 1

    return novo_rastreamento


# Função para salvar o quadro atual com as detecções realizadas
def salva_quadro_atual_com_deteccoes_realizadas(img_atual, rastreamento, pasta_run, numero_quadro):
    for id_objeto, (_, coordenada) in rastreamento.items():
        x1, y1, x2, y2 = coordenada
        cv2.rectangle(img_atual, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img_atual,
            f"{id_objeto}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )
    caminho_quadro = os.path.join(pasta_run, f"quadro_{numero_quadro}_numerado.jpg")
    cv2.imwrite(caminho_quadro, img_atual)

def processar_quadros(model, quadros, pasta_quadros, pasta_run):
    rastreamento = {}

    for i in range(len(quadros)):
        quadro_atual = os.path.join(pasta_quadros, quadros[i])
        img_atual = cv2.imread(quadro_atual)

        deteccoes_atual = model(source=quadro_atual, classes=(2, 5, 7))
        histogramas_atual = gerar_histogramas_de_veiculos_de_um_quadro(img_atual, deteccoes_atual)

        if i == 0:
            # Inicializar rastreamento no primeiro quadro
            rastreamento = {
                j + 1: (histograma, coordenada)
                for j, (histograma, coordenada) in enumerate(histogramas_atual)
            }
        else:
            # Rastrear e numerar objetos
            rastreamento = rastrear_e_numerar(histogramas_atual, histogramas_atual, rastreamento, pasta_run)
        
        # Salvar quadro numerado
        salva_quadro_atual_com_deteccoes_realizadas(img_atual, rastreamento, pasta_run, i + 1)

# Função principal
def main():
    pasta_quadros = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'
    quadros = sorted([f for f in os.listdir(pasta_quadros) if f.endswith('.jpg')])
    quadros_pares = [quadro for quadro in quadros if int(quadro.split('img')[1].split('.')[0]) % 2 == 0]
    model = YOLO('yolov8m.pt')
    pasta_run = criar_pasta_run()

    processar_quadros(model, quadros_pares, pasta_quadros, pasta_run)

if __name__ == "__main__":
    main()
