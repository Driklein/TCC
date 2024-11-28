# Importa bibliotecas necessárias
import os  # Para manipulação de arquivos e diretórios
from ultralytics import YOLO  # Para uso do modelo YOLO para detecção de objetos
import matplotlib.pyplot as plt  # Para criação de gráficos e visualização de histogramas
import cv2  # Para manipulação de imagens e vídeos
import numpy as np  # Para manipulação de arrays numéricos

# Função para criar uma nova pasta para armazenar os resultados da execução
def criar_pasta_run(base_folder='resultados'):
    # Inicializa o número do diretório
    run_number = 1
    # Incrementa o número até encontrar um nome de pasta que não exista
    while os.path.exists(os.path.join(base_folder, f'run{run_number}')):
        run_number += 1
    # Cria o caminho da nova pasta
    run_path = os.path.join(base_folder, f'run{run_number}')
    # Cria a pasta
    os.makedirs(run_path)
    print(f"Pasta de execução criada: {run_path}")
    return run_path  # Retorna o caminho da pasta criada

# Função para calcular o coeficiente de Bhattacharyya entre dois histogramas
def coeficiente_de_bhattacharyya(hist1, hist2):
    # Calcula a soma das raízes do produto dos dois histogramas
    bc = np.sum(np.sqrt(hist1 * hist2))
    # Garante que o valor fique no intervalo [0, 1]
    bc = np.clip(bc, 0, 1)
    return bc  # Retorna o coeficiente

# Função para calcular a distância de Hellinger entre dois histogramas
def distancia_de_hellinger(hist1, hist2):
    # Normaliza os histogramas
    hist1 = hist1 / (np.sum(hist1) + 1e-6)
    hist2 = hist2 / (np.sum(hist2) + 1e-6)
    # Calcula o coeficiente de Bhattacharyya
    bc = coeficiente_de_bhattacharyya(hist1, hist2)
    # Calcula e retorna a distância de Hellinger
    hellinger = np.sqrt(1 - bc)
    return hellinger

# Função para gerar histogramas dos veículos detectados em uma imagem
def gerar_histogramas_de_veiculos_de_um_quadro(img, deteccoes):
    histogramas_com_coordenadas = []  # Lista para armazenar histogramas e coordenadas dos veículos
    # Verifica se há detecções válidas com caixas delimitadoras
    if deteccoes and hasattr(deteccoes[0], 'boxes'):
        for det in deteccoes[0].boxes:  # Itera sobre as caixas detectadas
            # Extrai as coordenadas da caixa delimitadora
            x1, y1, x2, y2 = map(int, det.xyxy[0])
            # Recorta a região correspondente ao veículo
            veiculo = img[y1:y2, x1:x2]
            # Calcula os histogramas dos canais RGB
            hist_r = cv2.calcHist([veiculo], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([veiculo], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([veiculo], [2], None, [256], [0, 256])
            # Adiciona os histogramas e as coordenadas à lista
            histogramas_com_coordenadas.append(([hist_r, hist_g, hist_b], (x1, y1, x2, y2)))
    return histogramas_com_coordenadas  # Retorna os histogramas e as coordenadas

# Função para salvar os histogramas do veículo atual e do veículo mais semelhante
def salva_histograma_atual_e_seu_semelhante(histograma_atual, histograma_semelhante, numero_veiculo, numero_quadro, pasta_run):
    # Cria a pasta para o quadro e o veículo
    pasta_quadro = os.path.join(pasta_run, f"quadro_{numero_quadro}")
    pasta_histograma = os.path.join(pasta_quadro, f"veiculo_{numero_veiculo}")
    os.makedirs(pasta_histograma, exist_ok=True)
    
    # Salva o histograma do veículo atual
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_atual[canal], color=cor, label=f"Atual - {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Atual - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma, "atual.png"))
    plt.close()

    # Salva o histograma do veículo mais semelhante
    plt.figure()
    for canal, cor in zip(range(3), ['r', 'g', 'b']):
        plt.plot(histograma_semelhante[canal], color=cor, label=f"Semelhante - {cor.upper()}")
    plt.legend()
    plt.title(f"Histograma Semelhante - Veículo {numero_veiculo}")
    plt.savefig(os.path.join(pasta_histograma, "semelhante.png"))
    plt.close()

# Função para desenhar retângulos e números nos veículos detectados
def desenhar_veiculo(img, coordenadas, numero_veiculo, cor):
    # Desenha um retângulo ao redor do veículo
    x1, y1, x2, y2 = coordenadas
    cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
    # Adiciona um texto com o número do veículo
    cv2.putText(img, f"{numero_veiculo}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

# Função principal para processar os quadros e rastrear veículos
def processar_quadros(model, quadros, pasta_quadros, pasta_run):
    veiculos_rastreio = {}  # Dicionário para rastrear veículos detectados
    contador_id = 1  # Inicializa contador para atribuição de IDs únicos
    max_quadros_ausentes = 5  # Número máximo de quadros que um veículo pode estar ausente

    for i, quadro in enumerate(quadros):  # Itera sobre os quadros
        img_atual = cv2.imread(os.path.join(pasta_quadros, quadro))  # Carrega a imagem do quadro atual
        # Realiza detecção usando o modelo YOLO
        deteccoes = model(source=os.path.join(pasta_quadros, quadro), classes=(2, 5, 7))
        # Gera histogramas dos veículos detectados
        histogramas_atuais = gerar_histogramas_de_veiculos_de_um_quadro(img_atual, deteccoes)

        novos_veiculos = {}  # Dicionário para rastrear veículos no quadro atual
        for histograma_atual, coord_atual in histogramas_atuais:
            menor_distancia = float('inf')  # Inicializa a menor distância como infinita
            id_rastreado = None  # Inicializa o ID do veículo mais semelhante como None
            histograma_semelhante = None  # Inicializa o histograma mais semelhante

            for id_veiculo, (histograma_rastreado, coordenada_rasteada, quadros_ausentes) in veiculos_rastreio.items():
                # Calcula a distância de Hellinger para cada canal
                distancia_r = distancia_de_hellinger(histograma_atual[0], histograma_rastreado[0])
                distancia_g = distancia_de_hellinger(histograma_atual[1], histograma_rastreado[1])
                distancia_b = distancia_de_hellinger(histograma_atual[2], histograma_rastreado[2])
                distancia_media = (distancia_r + distancia_g + distancia_b) / 3

                # Atualiza o menor valor de distância e o ID mais semelhante, se aplicável
                if distancia_media < menor_distancia:
                    menor_distancia = distancia_media
                    id_rastreado = id_veiculo
                    histograma_semelhante = histograma_rastreado

            if menor_distancia < 0.4:  # Se a distância média for menor que o limite
                novos_veiculos[id_rastreado] = (histograma_atual, coord_atual, 0)  # Atualiza o veículo rastreado
            else:
                novos_veiculos[contador_id] = (histograma_atual, coord_atual, 0)  # Adiciona novo veículo
                contador_id += 1

            # Salva os histogramas se foi encontrado um veículo semelhante
            if histograma_semelhante is not None:
                salva_histograma_atual_e_seu_semelhante(histograma_atual, histograma_semelhante, id_rastreado, i, pasta_run)

        # Incrementa contagem de ausência para veículos não detectados
        for id_veiculo, (histograma, coord, quadros_ausentes) in veiculos_rastreio.items():
            if id_veiculo not in novos_veiculos:
                if quadros_ausentes + 1 < max_quadros_ausentes:  # Verifica se ainda está dentro do limite
                    novos_veiculos[id_veiculo] = (histograma, coord, quadros_ausentes + 1)

        veiculos_rastreio = novos_veiculos  # Atualiza o rastreamento com os veículos do quadro atual

        # Desenha retângulos e IDs nos veículos detectados
        for id_veiculo, (hist, coord, _) in veiculos_rastreio.items():
            desenhar_veiculo(img_atual, coord, id_veiculo, (0, 255, 0))

        # Salva a imagem com os veículos destacados
        cv2.imwrite(os.path.join(pasta_run, f"quadro_{i+1}.jpg"), img_atual)

# Função principal
def main():
    # Caminho para a pasta contendo os quadros
    pasta_quadros = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'
    # Lista e ordena os arquivos de imagem
    quadros = sorted([f for f in os.listdir(pasta_quadros) if f.endswith('.jpg')])
    # Filtra apenas os quadros pares
    quadros_pares = [quadro for quadro in quadros if int(quadro.split('img')[1].split('.')[0]) % 2 == 0]
    # Carrega o modelo YOLO
    model = YOLO('yolov8m.pt')
    # Cria a pasta para armazenar os resultados
    pasta_run = criar_pasta_run()

    # Processa os quadros para rastrear veículos
    processar_quadros(model, quadros_pares, pasta_quadros, pasta_run)

# Chamada da Função principal
if __name__ == "__main__":
    main()  # Chama a função principal
