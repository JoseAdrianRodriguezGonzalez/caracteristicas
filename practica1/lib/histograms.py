
import cv2
import numpy as np

class Histogram:
    def __init__(self, images_path: list[str], num_size=1, quant_levels=16):
        self.path = images_path
        self.images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in images_path]
        self.num_size = num_size
        self.quant_levels = quant_levels

        # Listas de resultados
        self.media_images = []
        self.desviacion_images = []
        self.correlacion_images = []
        self.contraste_images = []
        self.homogeneidad_images = []
        self.sombra_images = []
        self.prominencia_images = []

    # ================================================================
    # 🔧 UTILIDADES BASE
    # ================================================================
    def cuantizar(self, img):
        """Reduce rango de intensidades a 0–Q."""
        Q = self.quant_levels
        return (img / (256 / Q)).astype(np.uint8)

    def Frecuencias(self, M):
        """Calcula frecuencias normalizadas (probabilidades) de una matriz."""
        vals, counts = np.unique(M, return_counts=True)
        total = np.sum(counts)
        return {int(v): c / total for v, c in zip(vals, counts)}

    def ventana_sumas(self, img, row, col):
        """Calcula histograma de sumas P_s."""
        s_win = []
        for dk in range(-self.num_size, self.num_size + 1):
            for dl in range(-self.num_size, self.num_size + 1):
                i = int(img[row, col])
                j = int(img[row + dk, col + dl])
                s_win.append(i + j)
        return self.Frecuencias(np.array(s_win))

    def ventana_diff(self, img, row, col):
        """Calcula histograma de diferencias P_d."""
        d_win = []
        for dk in range(-self.num_size, self.num_size + 1):
            for dl in range(-self.num_size, self.num_size + 1):
                i = int(img[row, col])
                j = int(img[row + dk, col + dl])
                d_win.append(i - j)
        return self.Frecuencias(np.array(d_win))

    # ================================================================
    # 🔹 MOMENTOS LOCALES SDH
    # ================================================================
    def media_calc(self, img):
        """Media local μs y μd combinadas."""
        img = self.cuantizar(img)
        K, L = img.shape
        res = np.zeros((K - 2 * self.num_size, L - 2 * self.num_size), dtype=np.float32)

        for r in range(self.num_size, K - self.num_size):
            for c in range(self.num_size, L - self.num_size):
                Ps = self.ventana_sumas(img, r, c)
                Pd = self.ventana_diff(img, r, c)
                mu_s = sum(i * p for i, p in Ps.items())
                mu_d = sum(j * p for j, p in Pd.items())
                res[r - self.num_size, c - self.num_size] = 0.5 * (mu_s + mu_d)
        return cv2.resize(res, (60, 60), interpolation=cv2.INTER_LINEAR)

    def varianza_calc(self, img, media):
        """Varianza local combinada σ²_s y σ²_d."""
        img = self.cuantizar(img)
        K, L = img.shape
        res = np.zeros((K - 2 * self.num_size, L - 2 * self.num_size), dtype=np.float32)

        for r in range(self.num_size, K - self.num_size):
            for c in range(self.num_size, L - self.num_size):
                Ps = self.ventana_sumas(img, r, c)
                Pd = self.ventana_diff(img, r, c)
                mu_s = sum(i * p for i, p in Ps.items())
                mu_d = sum(j * p for j, p in Pd.items())
                var_s = sum(((i - mu_s)**2) * p for i, p in Ps.items())
                var_d = sum(((j - mu_d)**2) * p for j, p in Pd.items())
                res[r - self.num_size, c - self.num_size] = 0.5 * (var_s + var_d)
        return cv2.resize(res, (60, 60), interpolation=cv2.INTER_LINEAR)

    def correlacion_calc(self, img, media, desviacion):
        """Correlación local normalizada."""
        img = self.cuantizar(img)
        K, L = img.shape
        res = np.zeros((K - 2 * self.num_size, L - 2 * self.num_size), dtype=np.float32)

        for r in range(self.num_size, K - self.num_size):
            for c in range(self.num_size, L - self.num_size):
                Ps = self.ventana_sumas(img, r, c)
                Pd = self.ventana_diff(img, r, c)
                mu_s = sum(i * p for i, p in Ps.items())
                mu_d = sum(j * p for j, p in Pd.items())
                sigma = desviacion[r - self.num_size, c - self.num_size]
                if sigma == 0:
                    continue
                corr_s = sum(((i - mu_s)**2) * p for i, p in Ps.items())
                corr_d = sum(((j - mu_d)**2) * p for j, p in Pd.items())
                res[r - self.num_size, c - self.num_size] = 0.5 * (corr_s + corr_d) / (sigma**2)
        return cv2.resize(res, (60, 60), interpolation=cv2.INTER_LINEAR)

    def contraste_calc(self, img):
        """Contraste local."""
        img = self.cuantizar(img)
        K, L = img.shape
        res = np.zeros((K - 2 * self.num_size, L - 2 * self.num_size), dtype=np.float32)

        for r in range(self.num_size, K - self.num_size):
            for c in range(self.num_size, L - self.num_size):
                Pd = self.ventana_diff(img, r, c)
                contraste = sum((j**2) * p for j, p in Pd.items())
                res[r - self.num_size, c - self.num_size] = contraste
        return cv2.resize(res, (60, 60), interpolation=cv2.INTER_LINEAR)

    def homogeneidad_calc(self, img):
        """Homogeneidad local."""
        img = self.cuantizar(img)
        K, L = img.shape
        res = np.zeros((K - 2 * self.num_size, L - 2 * self.num_size), dtype=np.float32)

        for r in range(self.num_size, K - self.num_size):
            for c in range(self.num_size, L - self.num_size):
                Pd = self.ventana_diff(img, r, c)
                hom = sum((1 / (1 + abs(j))) * p for j, p in Pd.items())
                res[r - self.num_size, c - self.num_size] = hom
        return cv2.resize(res, (60, 60), interpolation=cv2.INTER_LINEAR)

    def sombra_calc(self, img, media):
        """Skewness local (asimetría)."""
        img = self.cuantizar(img)
        K, L = img.shape
        res = np.zeros((K - 2 * self.num_size, L - 2 * self.num_size), dtype=np.float32)

        for r in range(self.num_size, K - self.num_size):
            for c in range(self.num_size, L - self.num_size):
                Ps = self.ventana_sumas(img, r, c)
                mu_s = sum(i * p for i, p in Ps.items())
                sombra = sum(((i - mu_s)**3) * p for i, p in Ps.items())
                res[r - self.num_size, c - self.num_size] = sombra
        return cv2.resize(res, (60, 60), interpolation=cv2.INTER_LINEAR)

    def prominencia_calc(self, img, media):
        """Kurtosis local (prominencia)."""
        img = self.cuantizar(img)
        K, L = img.shape
        res = np.zeros((K - 2 * self.num_size, L - 2 * self.num_size), dtype=np.float32)

        for r in range(self.num_size, K - self.num_size):
            for c in range(self.num_size, L - self.num_size):
                Ps = self.ventana_sumas(img, r, c)
                mu_s = sum(i * p for i, p in Ps.items())
                prom = sum(((i - mu_s)**4) * p for i, p in Ps.items())
                res[r - self.num_size, c - self.num_size] = prom
        return cv2.resize(res, (60, 60), interpolation=cv2.INTER_LINEAR)

    # ================================================================
    # 🧮 MÉTODOS PRINCIPALES
    # ================================================================
    def procesar_todo(self):
        for img in self.images:
            m = self.media_calc(img)
            print(m)
            self.media_images.append(m)
            v = self.varianza_calc(img, m)
            self.desviacion_images.append(v)
            c = self.correlacion_calc(img, m, v)
            self.correlacion_images.append(c)
            self.contraste_images.append(self.contraste_calc(img))
            self.homogeneidad_images.append(self.homogeneidad_calc(img))
            self.sombra_images.append(self.sombra_calc(img, m))
            self.prominencia_images.append(self.prominencia_calc(img, m))

