import cv2
import numpy as np
import os

class Samples:
    def __init__(self, path: str, crop: int = 60, debug: bool = False):
        self.path = path
        self.crop = crop
        self.debug = debug  # <-- nuevo parámetro
        self.folders = self.read_folders()
        self.mask_paths, self.image_paths = self.read_images()
        self.masks = [cv2.imread(m, cv2.IMREAD_GRAYSCALE) for m in self.mask_paths]
        self.images = [cv2.imread(i, cv2.IMREAD_GRAYSCALE) for i in self.image_paths]

    def read_folders(self):
        return [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]

    def read_images(self):
        mask_total, normal_total = [], []
        for folder in self.folders:
            folder_path = os.path.join(self.path, folder)
            files = os.listdir(folder_path)
            masks = [os.path.join(folder_path, f) for f in files if 'Mask' in f]
            images = [os.path.join(folder_path, f) for f in files if 'Mask' not in f]
            masks.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
            images.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))

            mask_total.extend(masks)
            normal_total.extend(images)

        return mask_total, normal_total


    def get_center(self, msk):
        # Falla = blanca, fondo = negro
        cy, cx = np.where(msk > 0)
        if len(cy) == 0 or len(cx) == 0:
            return None
        return np.mean(cx), np.mean(cy)

    def create_samples(self, img, mask):
        center = self.get_center(mask)
        if center is None:
            return []

        cx, cy = center
        h, w = img.shape[:2]
        half = self.crop // 2
        offset = [(0, 0),
                  (-half, -half),
                  (-half, half),
                  (half, -half),
                  (half, half)]

        samples = []
        for dx, dy in offset:
            x1 = int(max(0, cx + dx - half))
            x2 = int(min(w, cx + dx + half))
            y1 = int(max(0, cy + dy - half))
            y2 = int(min(h, cy + dy + half))

            patch = img[y1:y2, x1:x2]

            if patch.shape[0] == self.crop and patch.shape[1] == self.crop:
                samples.append(patch)

                if self.debug:
                    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    #cv2.circle(img_vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)  # centro
                    cv2.rectangle(img_vis, (x1, y1), (x2, y2), (0, 255, 0), 1)   # cuadro

                    # Mostrar imagen completa y patch
                    cv2.imshow("Vista general", img_vis)
                    cv2.imshow("Patch", patch)

                    key = cv2.waitKey(0)
                    if key == 27:  # tecla ESC
                        cv2.destroyAllWindows()
                        print("🟥 Salida manual por tecla ESC")
                        exit(0)
                    cv2.destroyAllWindows()


        return samples

    def save_samples(self):
        for img, mask, img_path in zip(self.images, self.masks, self.image_paths):
            patches = self.create_samples(img, mask)
            folder_name = os.path.dirname(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_dir = os.path.join(folder_name, "patches")
            os.makedirs(output_dir, exist_ok=True)

            for i, patch in enumerate(patches):
                filename = f"{base_name}_patch_{i+1}.tif"
                cv2.imwrite(os.path.join(output_dir, filename), patch)
