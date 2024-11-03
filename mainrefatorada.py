
import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Função para criar nova pasta de execução
def create_run_folder(base_folder='resultados'):
    run_number = 1
    while os.path.exists(os.path.join(base_folder, f'run{run_number}')):
        run_number += 1
    run_path = os.path.join(base_folder, f'run{run_number}')
    os.makedirs(run_path)
    print(f"Pasta de execução criada: {run_path}")
    return run_path

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



def main():
	
    # Processando pasta contendo os quadros e armazenando somente quadros pares
    pasta_quadros = 'C:/Users/rodri/Desktop/TCC/imagens_tempo_diurno_2'
    todos_quadros = sorted([f for f in os.listdir(pasta_quadros) if f.endswith('.jpg')])
    quadros_pares = [quadro for quadro in todos_quadros if int(quadro.split('img')[1].split('.')[0]) % 2 == 0]
    quadros = quadros_pares
        
    # Carrega o modelo YOLO
    model = YOLO('yolov8m.pt')

    #Cria pasta com a execução atual
    run_folder = create_run_folder()

    #Para cada quadro
    for quadro in range(len(quadros) - 1):
        
        #Armazena os diretórios
        diretorio_atual_quadro = os.path.join(pasta_quadros, quadros_pares[i]) 
        diretorio_proximo_quadro = os.path.join(pasta_quadros, quadros_pares[i + 1])

        #Realiza a detecção utilizando o YOLO
        veiculos_detectados_atual_quadro = model(source=diretorio_atual_quadro, classes=(2, 5, 7))
        veiculos_detectados_proximo_quadro = model(source=diretorio_proximo_quadro, classes=(2, 5, 7))

        #Armazena os histogramas dos veículos detectados em cada quadro
        histogramas_atual_quadro = gerar_histogramas_dos_veiculos_de_um_quadro(diretorio_atual_quadro, veiculos_detectados_atual_quadro) #Implemente a função
        histogramas_proximo_quadro = gerar_histogramas_dos_veiculos_de_um_quadro(diretorio_proximo_quadro, veiculos_detectados_proximo_quadro) 

        #Variáveis a serem utilizadas
        menor_distancia = -1
        histogramas_semelhantes = []
        numero_veiculo_histograma_atual = 0

        #Para cada histograma dos veículo detectados no quadro atual
        for histograma_atual in histogramas_atual_quadro:
            
            #Realizar a comparação do histograma_atual com todos os histogramas gerados no próximo quadro (histogramas_proximo_quadro) utilizando o método da distância de hellinger (usando o coeficiente de bhattacharyya) já implementado (ao se tratar do último quadro, não realizar a comparação pois não existirá próximo quadro).
            for histograma_proximo in histogramas_proximo_quadro:
                distancia_entre_histograma_atual_e_proximo = calcula_distancia_entre_dois_histogramas(histograma_atual, histograma_proximo) #Implemente a função

                if(distancia_entre_histograma_atual_e_proximo < menor_distancia):
                    menor_distancia = distancia_entre_histograma_atual_e_proximo
                    histograma_similar_ao_atual = histograma_proximo

            histogramas_semelhantes.append((histograma_atual, histograma_similar_ao_atual, numero_veiculo_histograma_atual))
            #Plotar nessa linha o quadro atual com os veículos enumerados


        

    return 0

    if __name__ == "__main__":
        main()
