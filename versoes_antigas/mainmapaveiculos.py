import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Função para criar uma pasta exclusiva para a execução atual
def criar_pasta_run(base_folder='resultados'):
    """
    Cria uma nova pasta com nome `runX`, onde X é o próximo número disponível.
    """
    run_number = 1
    while os.path.exists(os.path.join(base_folder, f'run{run_number}')):
        run_number += 1
    run_path = os.path.join(base_folder, f'run{run_number}')
    os.makedirs(run_path)
    print(f"Pasta de execução criada: {run_path}")
    return run_path

# Função para calcular o coeficiente de Bhattacharyya entre dois histogramas
def coeficiente_de_bhattacharyya(hist1, hist2):
    """
    Calcula o coeficiente de Bhattacharyya, que mede a similaridade entre dois histogramas.
    """
    bc = np.sum(np.sqrt(hist1 * hist2))
    bc = np.clip(bc, 0, 1)
    return bc

# Função para calcular a distância de Hellinger
def distancia_de_hellinger(hist1, hist2):
    """
    Converte os histogramas para probabilidade e calcula a distância de Hellinger,
    que mede a diferença entre os dois histogramas.
    """
    hist1 = hist1 / (np.sum(hist1) + 1e-6)
    hist2 = hist2 / (np.sum(hist2) + 1e-6)
    bc = coeficiente_de_bhattacharyya(hist1, hist2)
    hellinger = np.sqrt(1 - bc)
    return hellinger

# Função para calcular a distância entre os centros de duas caixas delimitadoras
def distancia_centro(caixa1, caixa2):
    """
    Calcula a distância euclidiana entre os centros de duas caixas delimitadoras (bounding boxes).
    """
    x1_centro = (caixa1[0] + caixa1[2]) / 2
    y1_centro = (caixa1[1] + caixa1[3]) / 2
    x2_centro = (caixa2[0] + caixa2[2]) / 2
    y2_centro = (caixa2[1] + caixa2[3]) / 2
    return np.sqrt((x1_centro - x2_centro)**2 + (y1_centro - y2_centro)**2)

# Função para gerar histogramas RGB de veículos detectados em um quadro
def gerar_histogramas_dos_veiculos_de_um_quadro(img, deteccoes):
    """
    Gera os histogramas RGB normalizados dos veículos detectados em uma imagem.
    Retorna uma lista de histogramas e as coordenadas das caixas.
    """
    histogramas_com_coordenadas = []
    if deteccoes and hasattr(deteccoes[0], 'boxes'):
        for det in deteccoes[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Coordenadas da caixa
            veiculo = img[y1:y2, x1:x2]  # Recorte do veículo
            # Calcula histogramas para os canais RGB
            hist_r = cv2.calcHist([veiculo], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([veiculo], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([veiculo], [2], None, [256], [0, 256])
            histogramas_com_coordenadas.append(([hist_r, hist_g, hist_b], (x1, y1, x2, y2)))
    return histogramas_com_coordenadas

# Função para salvar histogramas RGB semelhantes
def salvar_histogramas_em_pares(histograma_atual, histograma_similar, veiculo_id, pasta_run):
    """
    Salva os histogramas RGB de um veículo atual e de seu correspondente semelhante em pastas organizadas.
    """
    pasta_histograma_par = os.path.join(pasta_run, f"histogramas_semelhantes_{veiculo_id}")
    os.makedirs(pasta_histograma_par, exist_ok=True)
    for histograma, nome in zip([histograma_atual, histograma_similar], ['atual', 'similar']):
        plt.figure()
        for canal, cor in zip(range(3), ['r', 'g', 'b']):
            plt.plot(histograma[canal], color=cor, label=f"Canal {cor.upper()}")
        plt.legend()
        plt.title(f"Histograma {nome.capitalize()} - Veículo {veiculo_id}")
        plt.savefig(os.path.join(pasta_histograma_par, f"histograma_{nome}_veiculo_{veiculo_id}.png"))
        plt.close()

# Função para desenhar as caixas e IDs dos veículos rastreados
def desenhar_veiculos(img, mapa_veiculos):
    """
    Desenha retângulos e IDs nas caixas delimitadoras dos veículos rastreados.
    """
    for numero_veiculo, (_, (x1, y1, x2, y2)) in mapa_veiculos.items():
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{numero_veiculo}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return img

# Função principal
def main():
    """
    Função principal para realizar detecção, rastreamento e armazenamento de histogramas RGB
    de veículos detectados em uma sequência de quadros.
    """
    # Caminho para os quadros e organização dos arquivos
    pasta_quadros = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'
    todos_quadros = sorted([f for f in os.listdir(pasta_quadros) if f.endswith('.jpg')])
    quadros_pares = [quadro for quadro in todos_quadros if int(quadro.split('img')[1].split('.')[0]) % 2 == 0]
    quadros = quadros_pares
    
    # Carrega o modelo YOLO
    model = YOLO('yolov8m.pt')

    # Cria pasta para salvar resultados desta execução
    pasta_run = criar_pasta_run()

    # Variáveis para rastrear veículos
    mapa_veiculos = {}
    proximo_id = 1

    # Processa os quadros sequencialmente
    for i, quadro in enumerate(quadros[:-1]):
        # Carrega os quadros atual e próximo
        img_atual = cv2.imread(os.path.join(pasta_quadros, quadro))
        img_proximo = cv2.imread(os.path.join(pasta_quadros, quadros[i + 1]))

        # Realiza detecção no quadro atual e próximo
        deteccoes_atual = model(source=os.path.join(pasta_quadros, quadro), classes=(2, 5, 7))
        deteccoes_proximo = model(source=os.path.join(pasta_quadros, quadros[i + 1]), classes=(2, 5, 7))

        # Gera histogramas RGB dos veículos detectados
        histogramas_atual = gerar_histogramas_dos_veiculos_de_um_quadro(img_atual, deteccoes_atual)
        histogramas_proximo = gerar_histogramas_dos_veiculos_de_um_quadro(img_proximo, deteccoes_proximo)

        # Atualiza o rastreamento
        novo_mapa_veiculos = {}
        for hist_atual, bbox_atual in histogramas_atual:
            melhor_id = None
            menor_distancia = float('inf')
            hist_similar = None

            # Associa veículo atual a um veículo rastreado
            for id_existente, (hist_existente, bbox_existente) in mapa_veiculos.items():
                # Calcula distância baseada no histograma e na posição
                distancia_hist = sum(distancia_de_hellinger(hist_atual[c], hist_existente[c]) for c in range(3)) / 3
                distancia_pos = distancia_centro(bbox_atual, bbox_existente)
                distancia_total = distancia_hist + 0.01 * distancia_pos

                if distancia_total < menor_distancia:
                    menor_distancia = distancia_total
                    melhor_id = id_existente
                    hist_similar = hist_existente

            # Atualiza o mapa de veículos com base no menor custo de associação
            if melhor_id is not None and menor_distancia < 0.3:  # Limite para considerar o mesmo veículo
                novo_mapa_veiculos[melhor_id] = (hist_atual, bbox_atual)
                salvar_histogramas_em_pares(hist_atual, hist_similar, melhor_id, pasta_run)
            else:
                novo_mapa_veiculos[proximo_id] = (hist_atual, bbox_atual)
                proximo_id += 1

        mapa_veiculos = novo_mapa_veiculos

        # Desenha as caixas dos veículos no quadro atual
        img_com_veiculos = desenhar_veiculos(img_atual, mapa_veiculos)

        # Salva o quadro anotado
        cv2.imwrite(os.path.join(pasta_run, f"frame_{i + 1}.jpg"), img_com_veiculos)

    # Fecha todas as janelas de visualização
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
