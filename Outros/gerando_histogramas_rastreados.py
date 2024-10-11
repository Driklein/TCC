from ultralytics import YOLO  # Importa a biblioteca Ultralytics YOLO
import matplotlib.pyplot as plt  # Importa a biblioteca matplotlib para a criação de gráficos
import cv2  # Importa a biblioteca OpenCV para manipulação de imagens

# Carregar o modelo YOLO
model = YOLO('yolov8m.pt')  # Carrega o modelo YOLOv8m

# Realizar o rastreamento na imagem de teste
results = model.track(
    source='C:/Users/rodri/Desktop/TCC/imagem_teste.jpg',  # Define a imagem de entrada
    show=True,  # Exibe a imagem com as detecções
    save=True,  # Salva a imagem com as detecções
    save_dir='C:/Users/rodri/Desktop/TCC/',  # Define o diretório onde a imagem será salva
    tracker='bytetrack.yaml',  # Define o algoritmo de rastreamento a ser usado
    classes=(2, 3, 5, 7)  # Define as classes a serem detectadas
)
for result in results:  # Para cada resultado na lista de resultados
    boxes = result.boxes.xyxy  # Obtenha as caixas delimitadoras no formato xyxy
    image = cv2.imread('C:/Users/rodri/Desktop/TCC/imagem_teste.jpg')  # Carrega a imagem do caminho especificado
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Converte a imagem de BGR para RGB

    for i, box in enumerate(boxes):  # Para cada caixa delimitadora na lista de caixas
        x1, y1, x2, y2 = map(int, box)  # Obtém as coordenadas da caixa delimitadora
        roi = image[y1:y2, x1:x2]  # Extrai a região de interesse (ROI) da imagem
        color = ('b','g','r')  # Define as cores para os canais RGB
        for j, col in enumerate(color):  # Para cada canal de cor
            histr = cv2.calcHist([roi],[j],None,[256],[0,256])  # Calcula o histograma para o canal de cor atual
            plt.plot(histr, color = col, label=f'Veículo ID: {i+1}, Canal: {col}')  # Plota o histograma na cor correspondente e adiciona a legenda
            plt.xlim([0,256])  # Define o limite do eixo x para o gráfico
            plt.ylim([0,1500])  # Define o limite do eixo y para o gráfico
        plt.legend()  # Adiciona a legenda ao gráfico
        plt.savefig(f'C:/Users/rodri/Desktop/TCC/histogram_{i+1}.png')  # Salva o gráfico como uma imagem
        plt.show()  # Exibe o gráfico
