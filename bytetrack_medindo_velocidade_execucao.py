import os
import cv2
from ultralytics import YOLO
from pathlib import Path
import time  # Importa o módulo para medir o tempo

# Caminho para as imagens e pasta de saída
pasta_imagens = 'C:/Users/rodri/Desktop/TCC/imagens_trafego_intenso'
saida_pasta = 'C:/Users/rodri/Desktop/TCC/run_bytetrack'
Path(saida_pasta).mkdir(parents=True, exist_ok=True)

# Inicialize o modelo YOLO
modelo = YOLO('yolov8m.pt')

# Configuração do rastreamento ByteTrack com o arquivo de configuração
configuracao_rastreador = "bytetrack.yaml"  # Caminho do arquivo de configuração do ByteTrack

# Função para processar as imagens e rastrear os veículos
def processar_rastreamento():
    for nome_imagem in os.listdir(pasta_imagens):
        caminho_imagem = os.path.join(pasta_imagens, nome_imagem)
        
        # Carregar a imagem
        imagem = cv2.imread(caminho_imagem)
        
        if imagem is not None:
            # Realiza a detecção e rastreamento com o YOLO e ByteTrack
            resultados = modelo.track(imagem, tracker=configuracao_rastreador, persist=True)  # Usando o ByteTrack
            
            # Acessa o frame anotado com o rastreamento
            quadro_anotado = resultados[0].plot()

            # Salva a imagem anotada
            caminho_imagem_saida = os.path.join(saida_pasta, nome_imagem)
            cv2.imwrite(caminho_imagem_saida, quadro_anotado)

def main():
    inicio = time.time()  # Inicia a contagem do tempo

    processar_rastreamento()
    
    fim = time.time()  # Finaliza a contagem do tempo
    tempo_total = fim - inicio
    print(f"Tempo total de execução: {tempo_total:.2f} segundos")

if __name__ == "__main__":
    main()
