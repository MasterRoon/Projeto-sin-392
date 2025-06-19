# Projeto SIN 392 - Editor Interativo de Imagens Digitais

Este é um projeto desenvolvido como parte da disciplina **SIN 392 - Introdução ao Processamento Digital de Imagens** na Universidade Federal de Viçosa - Campus Rio Paranaíba (2025/1).

O sistema permite realizar diversas operações de processamento de imagens de forma interativa com interface gráfica.

---

## 📦 Funcionalidades Implementadas

### 🎨 Pré-processamento e Visualização

- Carregar e exibir imagem em tons de cinza (redimensionada para 512x512)
- Exibir histograma da imagem
- Salvar imagem carregada

### 🔧 Transformações de Intensidade

- Alargamento de Contraste (Stretching)
- Equalização de Histograma

### 🔍 Filtros

- **Passa-Baixa**:
  - Média
  - Mediana
  - Gaussiano
  - Mínimo
  - Máximo
- **Passa-Alta**:
  - Laplaciano
  - Roberts
  - Prewitt
  - Sobel

### 📡 Domínio da Frequência

- Exibir Espectro de Fourier
- Convolução no Domínio da Frequência:
  - Filtro Passa-Baixa (corte central no espectro)
  - Filtro Passa-Alta (remoção do centro no espectro)

### ⚙️ Morfologia Matemática

- Erosão
- Dilatação

### 🧠 Segmentação

- Segmentação por Limiarização de Otsu

### 📊 Descritores

- **Cor**: Histograma RGB (simulado a partir de tons de cinza)
- **Textura**: Local Binary Pattern (LBP)
- **Forma**: Extração de contornos com contagem

---

## 🖼️ Interface Gráfica (GUI)

O sistema possui uma interface interativa construída com `tkinter`, permitindo fácil acesso a todas as operações através de botões:

&#x20;

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.10 ou superior
- Anaconda (recomendado) ou instalação manual de bibliotecas

### Instalação via Anaconda

```bash
conda create -n sin392 python=3.10
conda activate sin392
pip install opencv-python matplotlib pillow
```

### Execução

```bash
python main.py
```

---

## 📁 Estrutura do Projeto

```
projeto_sin392/
├── main.py              # Código principal da interface
├── README.md            # Este arquivo
```

---

## 👨‍🏫 Autoria

**Disciplina**: SIN 392 - Introdução ao Processamento de Imagens Digitais\
**Período**: 2025/1\
**Instituição**: Universidade Federal de Viçosa - Campus Rio Paranaíba

**Desenvolvedor**: [João Matheus Rosano Rocha]

---

## 📽️ Demonstração em Vídeo

O vídeo demonstrativo do funcionamento do sistema está disponível no YouTube:\
📺 [Link para o vídeo demonstrativo]([https://youtube.com](https://youtu.be/vKDRuScvzvY))&#x20;

---

## 📄 Licença

Este projeto é acadêmico e livre para uso educacional. Compartilhe conhecimento! 🤝

