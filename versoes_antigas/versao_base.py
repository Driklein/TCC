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

# Função para salvar histogramas dos canais R, G e B em arquivos separados
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


# Função para desenhar retângulos e números nos veículos detectados
def enumera_veiculo_atual_e_seu_semelhante_com_o_mesmo_numero(img_atual, img_proximo, coordenada_atual, coordenada_semelhante, numero_veiculo, pasta_run):
    # Desenhar no quadro atual
    x1, y1, x2, y2 = coordenada_atual
    cv2.rectangle(img_atual, (x1, y1), (x2, y2), (0, 255, 0), 2) 
    cv2.putText(
        img_atual, 
        f"{numero_veiculo}", 
        (x1, y1 - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.9, 
        (0, 255, 0), 
        2
    )

    # Desenhar no quadro próximo
    x1, y1, x2, y2 = coordenada_semelhante
    cv2.rectangle(img_proximo, (x1, y1), (x2, y2), (0, 255, 0), 2)  
    cv2.putText(
        img_proximo, 
        f"{numero_veiculo}", 
        (x1, y1 - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.9, 
        (255, 0, 0), 
        2
    )

# Função para salvar o quadro atual com as detecções realizadas
def salva_quadro_atual_com_deteccoes_realizadas(img_atual, numero_quadro, pasta_run):
    # Caminho para salvar o quadro
    caminho_quadro = os.path.join(pasta_run, f"quadro_{numero_quadro}_detecoes.jpg")
    cv2.imwrite(caminho_quadro, img_atual)

def processar_quadros(model, quadros, pasta_quadros, pasta_run):
    for i in range(len(quadros) - 1):
        quadro_atual = os.path.join(pasta_quadros, quadros[i])
        quadro_proximo = os.path.join(pasta_quadros, quadros[i + 1])
        img_atual = cv2.imread(quadro_atual)
        img_proximo = cv2.imread(quadro_proximo)

        deteccoes_atual = model(source=quadro_atual, classes=(2, 5, 7))
        deteccoes_proximo = model(source=quadro_proximo, classes=(2, 5, 7))

        histogramas_atual = gerar_histogramas_de_veiculos_de_um_quadro(img_atual, deteccoes_atual)
        histogramas_proximo = gerar_histogramas_de_veiculos_de_um_quadro(img_proximo, deteccoes_proximo)

        numero_quadro = i + 1
        for numero_veiculo, (histograma_atual, coordenada_atual) in enumerate(histogramas_atual, start=1):
            menor_distancia = float('inf')
            histograma_semelhante = None
            coordenada_semelhante = None

            for histograma_proximo, coordenada_proximo in histogramas_proximo:
                distancia_r = distancia_de_hellinger(histograma_atual[0], histograma_proximo[0])
                distancia_g = distancia_de_hellinger(histograma_atual[1], histograma_proximo[1])
                distancia_b = distancia_de_hellinger(histograma_atual[2], histograma_proximo[2])
                distancia_media = (distancia_r + distancia_g + distancia_b) / 3

                if distancia_media < menor_distancia:
                    menor_distancia = distancia_media
                    histograma_semelhante = histograma_proximo
                    coordenada_semelhante = coordenada_proximo

            # Salvar histogramas
            salva_histograma_atual_e_seu_semelhante(histograma_atual, histograma_semelhante, numero_veiculo, numero_quadro, pasta_run)


            # Desenhar detecções e numerar os veículos
            enumera_veiculo_atual_e_seu_semelhante_com_o_mesmo_numero(
                img_atual, img_proximo, coordenada_atual, coordenada_semelhante, numero_veiculo, pasta_run
            )

        # Salvar o quadro atual com as detecções realizadas
        salva_quadro_atual_com_deteccoes_realizadas(img_atual, numero_quadro, pasta_run)

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
