import cv2
import numpy as np
import os
import random
import xml.etree.cElementTree as ET
from xml.dom import minidom
from pathlib import Path

# Caminhos (relativos à raiz do repositório, independente de onde o script é executado)
ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / 'data' / 'raw' / 'Dataset_Soldas'
OUT_DIR = ROOT / 'data' / 'synthetic'
IMAGES_DIR = OUT_DIR / 'images'
YOLO_DIR = OUT_DIR / 'annotations' / 'yolo'
VOC_DIR = OUT_DIR / 'annotations' / 'pascal_voc'
REPORT_DIR = OUT_DIR / 'reports'

# Configurações globais
espacoBordaVertical = 30
espacoBordaHorizontal = 90
espacoLargura = 40       # Espaço entre as imagens
espacoAltura = 60       
tamanho_imagem = 40  # Tamanho de cada imagem
corPlaca = [30,105,210]


def calcular_dimensoes_placa(num_colunas, num_linhas):
    """Calcula a largura e altura da placa."""
    placa_largura = (num_colunas * tamanho_imagem) + (num_colunas - 1)* espacoLargura + espacoBordaHorizontal*2
    placa_altura = (num_linhas * tamanho_imagem) + (num_linhas - 1) * espacoAltura + espacoBordaVertical*2
    return placa_largura, placa_altura

def carregar_imagens(caminho):
    """Carrega o caminho das imagens de cada categoria a partir do diretório fornecido."""
    caminho_por_categoria = {}
    for root, dirs, files in os.walk(caminho):
        categoria = os.path.basename(root)
        if categoria not in caminho_por_categoria:
            caminho_por_categoria[categoria] = [] # Inicializa uma lista vazia para a nova categoria
        for file in files:
            if file.endswith(".png"):
                caminho_por_categoria[categoria].append(os.path.join(root, file)) # Adiciona a imagem à sua categoria
    return caminho_por_categoria

def carregar_imagens_qtd(caminho_por_categoria: dict, imagens_a_distribuir: dict):
    """ Carrega as imagens de acordo com a quantidade. """
    imagens_por_categoria = {}
    for categoria, qtd in imagens_a_distribuir.items():
        if caminho_por_categoria.get(categoria):
            files = caminho_por_categoria[categoria]
            random.shuffle(files)
            imagens_por_categoria[categoria] = []
            for file in files[:qtd]:
                imagem = cv2.imread(file)
                if imagem is not None:
                    imagens_por_categoria[categoria].append(cv2.resize(imagem, (tamanho_imagem, tamanho_imagem)))
    return imagens_por_categoria

def solicitar_porcentagens(categorias):
    """Solicita as porcentagens de cada categoria ao usuário e valida a soma."""
    porcentagens = {}
    total_porcentagem = 0
    for categoria in categorias:
        while True:
            try:
                porcentagem = float(input(f"Porcentagem para {categoria} (0 a 1): "))
                if 0 <= porcentagem <= 1:
                    porcentagens[categoria] = porcentagem # Adiciona a porcentagem à tal categoria
                    total_porcentagem += porcentagem
                    break
                else:
                    print("Insira um valor entre 0 e 1.")
            except ValueError:
                print("Entrada inválida. Por favor, insira um número.")
    
    # Verificação se a soma das porcentagens é igual (aproximadamente) a 1 (100%)
    if not np.isclose(total_porcentagem, 1.0):
        raise Exception("A soma das porcentagens deve ser igual a 1 (100%).")
    
    return porcentagens

def ajustar_quantidades(total_imagens, porcentagens):
    """Ajusta as quantidades de imagens a serem distribuídas com base nas porcentagens."""
    imagens_a_distribuir = {}
    for categoria, porcentagem in porcentagens.items():
        imagens_a_distribuir[categoria] = int(total_imagens * porcentagem) # Quantidade de imagens p/ cada categoria
    
    # Se por causa do arredondamento, tiver imagem a menos do que o total, somamos uma imagem à cada categoria
    for categoria in imagens_a_distribuir:
        while imagens_a_distribuir[categoria] < total_imagens * porcentagens[categoria]:
            imagens_a_distribuir[categoria] += 1
    
    return imagens_a_distribuir

def criar_placa(placa_altura, placa_largura):
    """Cria uma placa para colocar as imagens."""
    return np.ones((placa_altura, placa_largura, 3), dtype=np.uint8) * np.array(corPlaca, dtype=np.uint8)
def pegar_imagem_aleatoria(categoria, imagens_por_categoria):
    """Pega uma imagem aleatória de uma categoria específica."""
    if categoria == "Espaco_branco":
        return None
    elif imagens_por_categoria[categoria]:
        return random.choice(imagens_por_categoria[categoria])
    return None

def iniciar_XML(nomeSaida, dirSaida, placa_largura, placa_altura):
    """Cria o cabeçalho da anotação Pascal VOC (formato usado pelo SSD)."""
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder')
    ET.SubElement(annotation, 'filename').text = nomeSaida+'.png'
    ET.SubElement(annotation, 'path').text = os.path.join(dirSaida, nomeSaida+'.png')
    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Dataset Soldas"
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(placa_largura)
    ET.SubElement(size, 'height').text = str(placa_altura)
    ET.SubElement(size, 'depth').text = '3'
    ET.SubElement(annotation, 'segmented').text = '0'
    return annotation

def colocar_imagens_na_placa(num_linhas, num_colunas, imagens_a_distribuir, imagens_por_categoria, placa, placa_largura, placa_altura, categorias):
    """Coloca as imagens na placa e retorna as coordenadas no formato YOLO e XML."""
    coordenadasTXT = [] # Lista para armazenar as coordenadas (YOLO)
    coordenadasXML = [] # Lista para armazenar as coordenadas
    
    for x in range(espacoBordaVertical, placa_altura-espacoBordaVertical, tamanho_imagem+espacoAltura):
        for y in range(espacoBordaHorizontal, placa_largura-espacoBordaHorizontal, tamanho_imagem+espacoLargura):
            
            # Escolher uma categoria de forma aleatória com base nas porcentagens
            categoria = random.choices(list(imagens_a_distribuir.keys()), list(imagens_a_distribuir.values()))[0]
            # Pegar uma imagem aleatória da categoria 
            imagem_solda = pegar_imagem_aleatoria(categoria, imagens_por_categoria)
            
            if imagem_solda is not None:
                # Colocar a imagem na placa
                placa[x:x + tamanho_imagem, y:y + tamanho_imagem] = imagem_solda
                
                # Para saber a "classe" de cada imagem
                numCategoria = categorias.index(categoria)
                
                # Armazena as coordenadas
                x_min, y_min = x, y
                x_max, y_max = x + tamanho_imagem, y + tamanho_imagem
                
                # Calcula o centro e dimensões (formato do YOLO)
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                # Normaliza as coordenadas (formato do YOLO)
                x_center_normalizado = x_center / placa_largura
                y_center_normalizado = y_center / placa_altura
                width_normalizado = width / placa_largura
                height_normalizado = height / placa_altura
                
                # Armazena as coordenadas no formato YOLO
                coordenadasTXT.append((numCategoria, x_center_normalizado, y_center_normalizado, width_normalizado, height_normalizado))
                
                # Armazena as coordenadas no formato SSD
                coordenadasXML.append((categoria, x_min, y_min, x_max, y_max))
                
                # Reduz o contador de imagens que faltam da categoria
                imagens_a_distribuir[categoria] -= 1
                
                # Se esgotar a quantidade de imagens de uma categoria, removemos das opções
                if imagens_a_distribuir[categoria] <= 0:
                    del imagens_a_distribuir[categoria]
            else:
                # Se for espaço em branco apenas diminuímos 1 na contagem e não adicionamos nada à placa
                imagens_a_distribuir[categoria] -= 1
    
    return coordenadasTXT, coordenadasXML

def salvar_placa_e_coordenadas(placa, nomePlaca,
                               dirImagens, dirYolo, dirXML,
                               coordenadas, coordenadasXML,
                               placa_largura, placa_altura):
    """Salva a placa como imagem, e as coordenadas em .txt (YOLO) e .xml (Pascal VOC / SSD)."""
    for d in (dirImagens, dirYolo, dirXML): # Cria os diretórios
        os.makedirs(d, exist_ok=True)

    cv2.imwrite(os.path.join(dirImagens, nomePlaca+'.png'), placa)

    with open(os.path.join(dirYolo, nomePlaca+'.txt'), 'w') as file:
        for coord in coordenadas:
            linha = f"{coord[0]} {coord[1]:.6f} {coord[2]:.6f} {coord[3]:.6f} {coord[4]:.6f}\n"
            file.write(linha)

    annotation =  iniciar_XML(nomePlaca, dirImagens, placa_largura, placa_altura) # Coordenadas do XML
    for coord in coordenadasXML:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = coord[0]
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(coord[1]) 
        ET.SubElement(bbox, 'ymin').text = str(coord[2])
        ET.SubElement(bbox, 'xmax').text = str(coord[3])
        ET.SubElement(bbox, 'ymax').text = str(coord[4])

    xml_str = ET.tostring(annotation)
    xml_pretty_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    xml_pretty_str = "\n".join(xml_pretty_str.split("\n")[1:])
    xml_file = os.path.join(dirXML, nomePlaca+'.xml')
    with open(xml_file, "w") as f:
        f.write(xml_pretty_str)

def salvar_relatorio(nomePlaca, dirRelatorio, coordenadas, categorias):
    """Salva um relatório de contagem das categorias em um arquivo .txt."""
    os.makedirs(dirRelatorio, exist_ok=True) # Cria o diretório
    
    # Inicializa o dicionário de contagem para cada categoria, não adiciona a categoria de "Espaço em branco"
    contagem_categorias = {categoria: 0 for categoria in categorias[:-1]}
    
    # Conta a ocorrência de cada categoria
    for coord in coordenadas:
        categoria = coord[0]
        contagem_categorias[categoria] += 1
    
    # Cria o relatório
    relatorio_path = os.path.join(dirRelatorio, nomePlaca + '_relatorio.txt')
    with open(relatorio_path, 'w') as relatorio_file:
        relatorio_file.write(f"Relatorio de contagem de soldas para a placa: {nomePlaca}\n\n")
        for categoria, contagem in contagem_categorias.items():
            relatorio_file.write(f"{categoria}: {contagem}\n")

def gerar_porcentagens(categorias):
    """Gera as porcentagens para cada categoria aleatoriamente (para usar caso o usuário escolha gerar mais de 1 placa)"""
    valores = np.random.rand(len(categorias))
    
    # Normalizar para que a soma seja exatamente 1
    valores_normalizados = valores / np.sum(valores)
    porcentagens = {
        categoria: valor
        for categoria, valor in zip(categorias, valores_normalizados.tolist())
    }
    return porcentagens

# Main
def main():
    # Input do usuário
    print('1 - Para Gerar uma Placa')
    print('2 - Para Gerar uma quantidade X de Placas')
    op = int(input('Escolha uma opção: '))

    os.system('clear||cls')

    num_colunas = int(input("Número de colunas: "))
    num_linhas = int(input("Número de linhas: "))
    
    placa_largura, placa_altura = calcular_dimensoes_placa(num_colunas, num_linhas)
    
    caminho_imagens = DATASET_DIR
    caminho_por_categoria = carregar_imagens(caminho_imagens) # Carregar todas as imagens de solda
    if not any(caminho_por_categoria.values()): # Se houver erro para carregar as imagens
        raise Exception("Nenhuma imagem de solda encontrada no caminho especificado.")
    
    categorias = ["Soldas_ausente", "Soldas_boas", "Soldas_excesso", "Soldas_ponte", "Soldas_pouca", "Espaco_branco"]
    total_imagens = num_colunas * num_linhas

    match op:
        case 1:
            porcentagens = solicitar_porcentagens(categorias)
    
            imagens_a_distribuir = ajustar_quantidades(total_imagens, porcentagens)

            imagens_por_categoria = carregar_imagens_qtd(caminho_por_categoria, imagens_a_distribuir)

            placa = criar_placa(placa_altura, placa_largura)
            coordenadas, coordenadasXML = colocar_imagens_na_placa(num_linhas, num_colunas, imagens_a_distribuir, 
                                                                   imagens_por_categoria, placa, placa_largura, 
                                                                   placa_altura, categorias)
            
            salvar_placa_e_coordenadas(placa, 'output0',
                                    IMAGES_DIR, YOLO_DIR, VOC_DIR,
                                    coordenadas, coordenadasXML,
                                    placa_largura, placa_altura)
            salvar_relatorio('output0', REPORT_DIR, coordenadasXML, categorias)

        case 2:
            qtd_placas = int(input('Informe a quantidade de Placas: '))
            for i in range(qtd_placas):
                porcentagens = gerar_porcentagens(categorias)
                imagens_a_distribuir = ajustar_quantidades(total_imagens, porcentagens)
                imagens_por_categoria = carregar_imagens_qtd(caminho_por_categoria, imagens_a_distribuir)
                placa = criar_placa(placa_altura, placa_largura)
                coordenadas, coordenadasXML = colocar_imagens_na_placa(num_linhas, num_colunas, imagens_a_distribuir, 
                                                                       imagens_por_categoria, placa, placa_largura, 
                                                                       placa_altura, categorias)
                salvar_placa_e_coordenadas(placa, f'output{i}',
                                    IMAGES_DIR, YOLO_DIR, VOC_DIR,
                                    coordenadas, coordenadasXML,
                                    placa_largura, placa_altura)
                salvar_relatorio(f'output{i}', REPORT_DIR, coordenadasXML, categorias)

        case _:
            print('Opção invalida')
            

if __name__ == "__main__":
    main()
