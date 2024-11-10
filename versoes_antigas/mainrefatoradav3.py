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
    
    # Verifica se há alguma detecção e itera nas caixas de detecção
    if deteccoes and hasattr(deteccoes[0], 'boxes'):
        for det in deteccoes[0].boxes:
            x1, y1, x2, y2 = map(int, det.xyxy[0])  # Coordenadas da caixa de detecção
            veiculo = img[y1:y2, x1:x2]  # Recorta o veículo detectado

            # Calcula o histograma normalizado para cada canal de cor (R, G, B)
            hist_r = cv2.calcHist([veiculo], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([veiculo], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([veiculo], [2], None, [256], [0, 256])

            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
            
            # Armazena os histogramas dos canais R, G, B junto com as coordenadas
            histogramas_com_coordenadas.append(([hist_r, hist_g, hist_b], (x1, y1, x2, y2)))
    
    return histogramas_com_coordenadas

# Inicializa um dicionário para armazenar o histórico de veículos detectados
historico_veiculos = {}

# Função para salvar histogramas dos canais R, G e B em arquivos separados
def salvar_histogramas_rgb(histograma_atual, histograma_similar, numero_veiculo, pasta_run):
    # Cria uma pasta específica para salvar histogramas dos veículos semelhantes
    pasta_histograma_par = os.path.join(pasta_run, f"histogramas_semelhantes_{numero_veiculo}")
    os.makedirs(pasta_histograma_par, exist_ok=True)

    # Plota e salva o histograma atual
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_atual[canal], color=cor, label=f"Histograma Atual - Canal {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Atual - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma_par, f"histograma_atual_veiculo_{numero_veiculo}.png"))
    plt.close()

    # Plota e salva o histograma mais semelhante ao atual
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_similar[canal], color=cor, label=f"Histograma Similar - Canal {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Similar - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma_par, f"histograma_similar_veiculo_{numero_veiculo}.png"))
    plt.close()

# Função para salvar o quadro atual com todos os veículos detectados e enumerados
def salvar_quadro_com_veiculos(img, deteccoes, correspondencias, numero_quadro, pasta_run):
    img_deteccoes = img.copy()
    
    # Desenha cada veículo com seu número baseado nas correspondências
    for numero_veiculo, det in correspondencias.items():
        (x1, y1, x2, y2) = det['coords']
        
        # Desenha o retângulo ao redor do veículo
        cv2.rectangle(img_deteccoes, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Adiciona o número do veículo
        cv2.putText(img_deteccoes, f"{numero_veiculo}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Salva a imagem com as detecções e números na pasta de execução
    caminho_imagem = os.path.join(pasta_run, f"quadro_com_veiculos_{numero_quadro}.jpg")
    cv2.imwrite(caminho_imagem, img_deteccoes)
    print(f"Quadro {numero_quadro} com veículos detectados salvo em {caminho_imagem}")

# Função principal que processa os quadros e aplica detecção de veículos
def main():
    # Define o diretório contendo as imagens dos quadros e seleciona apenas os quadros pares
    pasta_quadros = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'
    todos_quadros = sorted([f for f in os.listdir(pasta_quadros) if f.endswith('.jpg')])
    quadros_pares = [quadro for quadro in todos_quadros if int(quadro.split('img')[1].split('.')[0]) % 2 == 0]
    quadros = quadros_pares
    
    # Carrega o modelo YOLO para detecção de veículos
    model = YOLO('yolov8m.pt')
    
    # Cria uma pasta para salvar os resultados da execução atual
    pasta_run = criar_pasta_run()

    # Inicializa a numeração dos veículos e o histórico de veículos para rastreamento entre quadros
    numero_veiculo = 1
    historico_veiculos = {}
    
    # Define um limiar de distância para identificar veículos semelhantes
    limiar_distancia = 0.3  

    # Itera sobre os quadros, processando dois quadros consecutivos de cada vez
    for i in range(len(quadros) - 1):
        # Caminhos para os quadros atual e próximo
        diretorio_atual_quadro = os.path.join(pasta_quadros, quadros[i])
        diretorio_proximo_quadro = os.path.join(pasta_quadros, quadros[i + 1])

        # Carrega as imagens dos quadros
        img_atual = cv2.imread(diretorio_atual_quadro)
        img_proximo = cv2.imread(diretorio_proximo_quadro)

        # Realiza a detecção de veículos nos quadros atual e próximo
        veiculos_detectados_atual_quadro = model(source=diretorio_atual_quadro, classes=(2, 5, 7))
        veiculos_detectados_proximo_quadro = model(source=diretorio_proximo_quadro, classes=(2, 5, 7))

        # Gera os histogramas de cor dos veículos detectados com suas coordenadas
        histogramas_atual_quadro = gerar_histogramas_dos_veiculos_de_um_quadro(img_atual, veiculos_detectados_atual_quadro)
        histogramas_proximo_quadro = gerar_histogramas_dos_veiculos_de_um_quadro(img_proximo, veiculos_detectados_proximo_quadro)

        # Dicionário para manter a correspondência entre veículos nos dois quadros
        correspondencias = {}

        # Itera sobre os veículos detectados no quadro atual
        for idx_atual, (histograma_atual, coords_atual) in enumerate(histogramas_atual_quadro):
            id_veiculo_atual = None
            menor_distancia = float('inf')
            histograma_similar = None

            # Compara o histograma atual com os histogramas de veículos do histórico
            for numero_id, veiculo_hist in historico_veiculos.items():
                # Calcula a distância de Hellinger para cada canal (R, G, B) e tira a média
                distancia_r = distancia_de_hellinger(histograma_atual[0], veiculo_hist[0][0])
                distancia_g = distancia_de_hellinger(histograma_atual[1], veiculo_hist[0][1])
                distancia_b = distancia_de_hellinger(histograma_atual[2], veiculo_hist[0][2])
                distancia_media = (distancia_r + distancia_g + distancia_b) / 3

                # Verifica se a distância média é a menor encontrada até agora
                if distancia_media < menor_distancia:
                    menor_distancia = distancia_media
                    id_veiculo_atual = numero_id
                    histograma_similar = veiculo_hist[0]

            # Verifica se a menor distância é menor que o limiar, indicando o mesmo veículo
            if menor_distancia < limiar_distancia:
                correspondencias[id_veiculo_atual] = {'coords': coords_atual, 'hist': histograma_atual}
                salvar_histogramas_rgb(histograma_atual, histograma_similar, id_veiculo_atual, pasta_run)
            else:
                # Se a distância ultrapassar o limiar, considera-se um veículo novo
                correspondencias[numero_veiculo] = {'coords': coords_atual, 'hist': histograma_atual}
                salvar_histogramas_rgb(histograma_atual, histograma_atual, numero_veiculo, pasta_run)
                numero_veiculo += 1  # Incrementa o número do próximo veículo novo

        # Atualiza o histórico de veículos com os identificados no quadro atual
        historico_veiculos.clear()
        for id_veiculo, dados in correspondencias.items():
            historico_veiculos[id_veiculo] = (dados['hist'], dados['coords'])

        # Salva o quadro atual com os veículos detectados e enumerados
        salvar_quadro_com_veiculos(img_atual, veiculos_detectados_atual_quadro, correspondencias, i, pasta_run)

    return 0

if __name__ == "__main__":
    main()
