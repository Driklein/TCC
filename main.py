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
def gerar_histogramas_dos_veiculos_de_um_quadro(img, deteccoes):
    histogramas_com_coordenadas = []
    if deteccoes and hasattr(deteccoes[0], 'boxes'):
        for det in deteccoes[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            veiculo = img[y1:y2, x1:x2]
            hist_r = cv2.calcHist([veiculo], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([veiculo], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([veiculo], [2], None, [256], [0, 256])
            hist_r = cv2.normalize(hist_r, hist_r)
            hist_g = cv2.normalize(hist_g, hist_g)
            hist_b = cv2.normalize(hist_b, hist_b)
            histogramas_com_coordenadas.append(([hist_r, hist_g, hist_b], (x1, y1, x2, y2)))
    return histogramas_com_coordenadas

# Função para salvar histogramas dos canais R, G e B em arquivos separados
def plotar_histogramas_rgb(histograma_atual, histograma_similar, numero_veiculo, pasta_run):
    pasta_histograma_par = os.path.join(pasta_run, f"histogramas_semelhantes_{numero_veiculo}")
    os.makedirs(pasta_histograma_par, exist_ok=True)
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_atual[canal], color=cor, label=f"Histograma Atual - Canal {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Atual - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma_par, f"histograma_atual_veiculo_{numero_veiculo}.png"))
    plt.close()
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_similar[canal], color=cor, label=f"Histograma Similar - Canal {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Similar - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma_par, f"histograma_similar_veiculo_{numero_veiculo}.png"))
    plt.close()

# Dicionário para armazenar a numeração dos veículos
numeracao_veiculos = {}

# Função para enumerar veículos com histogramas similares
def enumerar_veiculo_com_histogramas_similares(histograma_atual, histograma_similar, numero_veiculo):
    numeracao_veiculos[numero_veiculo] = histograma_atual
    return numero_veiculo

# Função para plotar quadro atual com detecções e enumerações
def plotar_quadro_atual_com_deteccoes_e_veiculos_enumerados(img, deteccoes, pasta_run, numero_quadro):
    for numero_veiculo, (histograma, (x1, y1, x2, y2)) in enumerate(deteccoes, start=1):
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'Veiculo {numero_veiculo}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    caminho_salvar = os.path.join(pasta_run, f'quadro_{numero_quadro}_com_deteccoes.png')
    cv2.imwrite(caminho_salvar, img)
    print(f"Quadro {numero_quadro} salvo com as detecções enumeradas em: {caminho_salvar}")

# Função principal
def main():
    pasta_quadros = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'
    todos_quadros = sorted([f for f in os.listdir(pasta_quadros) if f.endswith('.jpg')])
    quadros_pares = [quadro for quadro in todos_quadros if int(quadro.split('img')[1].split('.')[0]) % 2 == 0]
    quadros = quadros_pares
    model = YOLO('yolov8m.pt')
    pasta_run = criar_pasta_run()

    # Processando todos os quadros
    for numero_quadro in range(len(quadros) - 1):
        diretorio_atual_quadro = os.path.join(pasta_quadros, quadros[numero_quadro]) 
        diretorio_proximo_quadro = os.path.join(pasta_quadros, quadros[numero_quadro + 1])
        img_atual = cv2.imread(diretorio_atual_quadro)
        img_proximo = cv2.imread(diretorio_proximo_quadro)

        # Realiza a detecção utilizando o YOLO
        veiculos_detectados_atual_quadro = model(source=diretorio_atual_quadro, classes=(2, 5, 7))
        veiculos_detectados_proximo_quadro = model(source=diretorio_proximo_quadro, classes=(2, 5, 7))

        # Gera os histogramas dos veículos detectados em ambos os quadros
        histogramas_atual_quadro = gerar_histogramas_dos_veiculos_de_um_quadro(img_atual, veiculos_detectados_atual_quadro)
        histogramas_proximo_quadro = gerar_histogramas_dos_veiculos_de_um_quadro(img_proximo, veiculos_detectados_proximo_quadro)

        for numero_veiculo, (histograma_atual, (x1, y1, x2, y2)) in enumerate(histogramas_atual_quadro, start=1):
            menor_distancia = float('inf')
            histograma_similar_ao_atual = None
            for histograma_proximo, _ in histogramas_proximo_quadro:
               
                distancia_r = distancia_de_hellinger(histograma_atual[0], histograma_proximo[0])
                distancia_g = distancia_de_hellinger(histograma_atual[1], histograma_proximo[1])
                distancia_b = distancia_de_hellinger(histograma_atual[2], histograma_proximo[2])
               
                distancia_media = (distancia_r + distancia_g + distancia_b) / 3
                
                if distancia_media < menor_distancia:
                    menor_distancia = distancia_media
                    histograma_similar_ao_atual = histograma_proximo
            enumerar_veiculo_com_histogramas_similares(histograma_atual, histograma_similar_ao_atual, numero_veiculo)
            plotar_histogramas_rgb(histograma_atual, histograma_similar_ao_atual, numero_veiculo, pasta_run)

        # Plotar e salvar todos os quadros com detecções e enumerações
        plotar_quadro_atual_com_deteccoes_e_veiculos_enumerados(img_atual, histogramas_atual_quadro, pasta_run, numero_quadro)

    return 0

if __name__ == "__main__":
    main()
