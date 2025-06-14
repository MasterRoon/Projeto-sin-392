import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class EditorImagens:
    def __init__(self, root):
        self.root = root
        self.root.title("Projeto Pratico SIN 392")
        self.root.geometry("820x680")

        self.imagem_cv = None
        self.imagem_original_shape = None

        self.label_imagem = tk.Label(self.root)
        self.label_imagem.pack(pady=10)

        self.label_tamanho = tk.Label(self.root, text="Tamanho original: -")
        self.label_tamanho.pack()

        frame_botoes = tk.Frame(self.root)
        frame_botoes.pack(pady=10)

        ttk.Button(frame_botoes, text="Carregar Imagem", command=self.carregar_imagem).grid(row=0, column=0, padx=5)
        ttk.Button(frame_botoes, text="Salvar Imagem", command=self.salvar_imagem).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Histograma", command=self.exibir_histograma).grid(row=0, column=1, padx=5)
        ttk.Button(frame_botoes, text="Contraste", command=self.alargamento_contraste).grid(row=0, column=2, padx=5)
        ttk.Button(frame_botoes, text="Equalização", command=self.equalizacao_histograma).grid(row=0, column=3, padx=5)
        ttk.Button(frame_botoes, text="Passa-Baixa", command=self.aplicar_passabaixa).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Passa-Alta", command=self.aplicar_passaalta).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Fourier", command=self.exibir_fourier).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Filtro Fourier PB", command=self.fourier_passabaixa).grid(row=3, column=1, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Filtro Fourier PA", command=self.fourier_passaalta).grid(row=3, column=2, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Morfologia", command=self.aplicar_morfologia).grid(row=1, column=3, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Segmentação Otsu", command=self.segmentar_otsu).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Descritor de Cor", command=self.descritor_cor).grid(row=2, column=2, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Descritor de Textura", command=self.descritor_textura).grid(row=2, column=3, padx=5, pady=5)
        ttk.Button(frame_botoes, text="Descritor de Forma", command=self.descritor_forma).grid(row=3, column=0, padx=5, pady=5)

    def carregar_imagem(self):
        caminho = filedialog.askopenfilename(filetypes=[("Imagens", "*.png *.jpg *.bmp *.tif")])
        if caminho:
            imagem = cv2.imread(caminho)
            if imagem is not None:
                if len(imagem.shape) == 3:
                    imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

                self.imagem_cv = imagem
                self.imagem_original_shape = imagem.shape

                imagem_redimensionada = cv2.resize(imagem, (512, 512))
                imagem_rgb = cv2.cvtColor(imagem_redimensionada, cv2.COLOR_GRAY2RGB)
                imagem_pil = Image.fromarray(imagem_rgb)
                imagem_tk = ImageTk.PhotoImage(imagem_pil)

                self.label_imagem.configure(image=imagem_tk)
                self.label_imagem.image = imagem_tk
                self.label_tamanho.config(text=f"Tamanho original: {imagem.shape[1]}x{imagem.shape[0]}")

    def salvar_imagem(self):
        if self.imagem_cv is not None:
            caminho = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png"), ("JPG", "*.jpg"), ("BMP", "*.bmp")])
            if caminho:
                cv2.imwrite(caminho, self.imagem_cv)
                messagebox.showinfo("Imagem Salva", f"Imagem salva em:\n{caminho}")
        else:
            messagebox.showwarning("Aviso", "Nenhuma imagem carregada para salvar.")

    def fourier_passabaixa(self):
        self._filtro_frequencia(tipo='passa-baixa')

    def fourier_passaalta(self):
        self._filtro_frequencia(tipo='passa-alta')

    def _filtro_frequencia(self, tipo='passa-baixa', raio=30):
        if self.imagem_cv is not None:
            img = self.imagem_cv
            linhas, colunas = img.shape
            cx, cy = colunas // 2, linhas // 2
            fft = np.fft.fft2(img)
            fft_shift = np.fft.fftshift(fft)
            mask = np.zeros((linhas, colunas), np.uint8)
            if tipo == 'passa-baixa':
                cv2.circle(mask, (cx, cy), raio, 1, -1)
            elif tipo == 'passa-alta':
                mask[:] = 1
                cv2.circle(mask, (cx, cy), raio, 0, -1)
            filtrada = fft_shift * mask
            fft_ishift = np.fft.ifftshift(filtrada)
            img_filtrada = np.fft.ifft2(fft_ishift)
            img_filtrada = np.abs(img_filtrada)
            img_filtrada = np.uint8(np.clip(img_filtrada, 0, 255))
            espectro = 20 * np.log(np.abs(fft_shift) + 1)
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(img, cmap='gray')
            plt.title("Original")
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(espectro, cmap='gray')
            plt.title("Espectro de Fourier")
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(img_filtrada, cmap='gray')
            plt.title(f"{tipo.capitalize()} na Frequência")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def descritor_cor(self):
        if self.imagem_cv is not None:
            imagem_rgb = cv2.merge([self.imagem_cv]*3)  # Simula imagem RGB duplicando tons de cinza
            chans = cv2.split(imagem_rgb)
            cores = ('b', 'g', 'r')
            nomes = ('Azul (B)', 'Verde (G)', 'Vermelho (R)')
            plt.figure(figsize=(8, 4))
            for chan, cor, nome in zip(chans, cores, nomes):
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                plt.plot(hist, color=cor, label=nome)
            plt.title("Histograma Normalizado RGB (Simulado a partir de tons de cinza)")
            plt.xlabel("Intensidade")
            plt.ylabel("Frequência")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def descritor_textura(self):
        if self.imagem_cv is not None:
            imagem = self.imagem_cv
            lbp_img = np.zeros_like(imagem)
            for i in range(1, imagem.shape[0] - 1):
                for j in range(1, imagem.shape[1] - 1):
                    centro = imagem[i, j]
                    vizinhos = imagem[i-1:i+2, j-1:j+2] >= centro
                    vizinhos[1, 1] = 0
                    codigo = np.packbits(vizinhos.flatten())[0]
                    lbp_img[i, j] = codigo
            hist, _ = np.histogram(lbp_img.ravel(), bins=256, range=(0, 256))
            hist = hist.astype("float")
            hist /= (hist.sum() + 1e-6)
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(lbp_img, cmap='gray')
            plt.title("Imagem LBP")
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.plot(hist)
            plt.title("Histograma de Textura (LBP)")
            plt.xlabel("Padrões LBP")
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def descritor_forma(self):
        if self.imagem_cv is not None:
            _, binaria = cv2.threshold(self.imagem_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            img_contornos = np.zeros_like(self.imagem_cv)
            cv2.drawContours(img_contornos, contornos, -1, 255, 1)
            plt.figure(figsize=(6, 4))
            plt.imshow(img_contornos, cmap='gray')
            plt.title(f"Contornos Encontrados: {len(contornos)}")
            plt.axis('off')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    

    def exibir_histograma(self):
        if self.imagem_cv is not None:
            hist = cv2.calcHist([self.imagem_cv], [0], None, [256], [0, 256])
            plt.figure(figsize=(6, 4))
            plt.plot(hist, color='black')
            plt.title("Histograma")
            plt.xlabel("Intensidade")
            plt.ylabel("Frequência")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def alargamento_contraste(self):
        if self.imagem_cv is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")
            return

        min_val = np.min(self.imagem_cv)
        max_val = np.max(self.imagem_cv)
    
        if max_val == min_val:
            stretch = np.full(self.imagem_cv.shape, 128, dtype=np.uint8)
        else:
            stretch = ((self.imagem_cv - min_val) * 255 / (max_val - min_val)).astype(np.uint8)
    
            self._mostrar_comparacao(self.imagem_cv, stretch, "Alargamento de Contraste")

    def equalizacao_histograma(self):
        if self.imagem_cv is not None:
            eq = cv2.equalizeHist(self.imagem_cv)
            self._mostrar_comparacao(self.imagem_cv, eq, "Equalização de Histograma")
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def aplicar_passabaixa(self):
        if self.imagem_cv is not None:
            media = cv2.blur(self.imagem_cv, (3, 3))
            mediana = cv2.medianBlur(self.imagem_cv, 3)
            gauss = cv2.GaussianBlur(self.imagem_cv, (3, 3), 0)
            maximo = cv2.dilate(self.imagem_cv, np.ones((3, 3), np.uint8))
            minimo = cv2.erode(self.imagem_cv, np.ones((3, 3), np.uint8))
            imagens = [media, mediana, gauss, maximo, minimo]
            titulos = ["Média", "Mediana", "Gaussiano", "Máximo", "Mínimo"]
            self._mostrar_multiplas(imagens, titulos, "Filtros Passa-Baixa")
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def aplicar_passaalta(self):
        if self.imagem_cv is not None:
            laplaciano = cv2.convertScaleAbs(cv2.Laplacian(self.imagem_cv, cv2.CV_64F))
            kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=np.float32)
            kernel_roberts_y = np.array([[0, 1], [-1, 0]], dtype=np.float32)
            roberts = cv2.addWeighted(
                cv2.filter2D(self.imagem_cv, -1, kernel_roberts_x), 0.5,
                cv2.filter2D(self.imagem_cv, -1, kernel_roberts_y), 0.5, 0)
            kernel_prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
            kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=np.float32)
            prewitt = cv2.addWeighted(
                cv2.filter2D(self.imagem_cv, -1, kernel_prewitt_x), 0.5,
                cv2.filter2D(self.imagem_cv, -1, kernel_prewitt_y), 0.5, 0)
            sobel = cv2.addWeighted(
                cv2.convertScaleAbs(cv2.Sobel(self.imagem_cv, cv2.CV_64F, 1, 0, ksize=3)), 0.5,
                cv2.convertScaleAbs(cv2.Sobel(self.imagem_cv, cv2.CV_64F, 0, 1, ksize=3)), 0.5, 0)
            imagens = [laplaciano, roberts, prewitt, sobel]
            titulos = ["Laplaciano", "Roberts", "Prewitt", "Sobel"]
            self._mostrar_multiplas(imagens, titulos, "Filtros Passa-Alta")
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def exibir_fourier(self):
        if self.imagem_cv is not None:
            fft = np.fft.fft2(self.imagem_cv)
            fft_shift = np.fft.fftshift(fft)
            espectro = 20 * np.log(np.abs(fft_shift) + 1)
            self._mostrar_comparacao(self.imagem_cv, espectro, "Espectro de Fourier", cmap2='gray')
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def aplicar_morfologia(self):
        if self.imagem_cv is not None:
            _, binaria = cv2.threshold(self.imagem_cv, 127, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            erodida = cv2.erode(binaria, kernel, iterations=1)
            dilatada = cv2.dilate(binaria, kernel, iterations=1)
            self._mostrar_multiplas([binaria, erodida, dilatada], ["Binária", "Erosão", "Dilatação"], "Morfologia")
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def segmentar_otsu(self):
        if self.imagem_cv is not None:
            _, otsu = cv2.threshold(self.imagem_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self._mostrar_comparacao(self.imagem_cv, otsu, "Segmentação Otsu")
        else:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro.")

    def _mostrar_comparacao(self, original, processada, titulo, cmap1='gray', cmap2='gray'):
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap=cmap1)
        plt.title("Original")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(processada, cmap=cmap2)
        plt.title(titulo)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    def _mostrar_multiplas(self, imagens, titulos, titulo_geral):
        plt.figure(figsize=(14, 6))
        for i in range(len(imagens)):
            plt.subplot(2, 3, i+1)
            plt.imshow(imagens[i], cmap='gray')
            plt.title(titulos[i])
            plt.axis('off')
        plt.suptitle(titulo_geral)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = EditorImagens(root)
    root.mainloop()