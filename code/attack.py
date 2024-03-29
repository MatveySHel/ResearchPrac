import os
from abc import ABC
import cv2
import math
import numpy as np
from skimage import metrics
from skimage.util import random_noise


class BaseAttack(ABC):
    def __init__(self, filename: str):
        self.filename = filename

    def execute(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, image: np.ndarray):
        if not os.path.exists('./attack-result'):
            os.mkdir('attack-result')
        cv2.imwrite(os.path.join('./attack-result', self.filename), image)


class CompressionAttack(BaseAttack):
    def __init__(self, filename, quality=.5):
        self.quality = int(quality * 100)
        super().__init__(filename)

    def execute(self, image: np.ndarray):
        compressed_file_name = 'compressed_image'+str(self.quality)+'.jpg'
        cv2.imwrite(compressed_file_name, image, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return cv2.imread(compressed_file_name)


class NoiseAttack(BaseAttack):
    def __init__(self, filename, intensity=0.05):
        self.intensity = intensity
        super().__init__(filename)

    def execute(self, image: np.ndarray):
        noise = random_noise(image, mode='s&p', amount=self.intensity)
        return np.array(255 * noise, dtype=np.uint8)


class BlurAttack(BaseAttack):
    def __init__(self, filename, size=35):
        self.size = size
        super().__init__(filename)

    def execute(self, image: np.ndarray):
        return cv2.GaussianBlur(image, (self.size, self.size), 0)


class ScaleAttack(BaseAttack):
    def __init__(self, filename, factor=.5):
        self.factor = factor
        super().__init__(filename)

    def execute(self, image: np.ndarray):
        resized = cv2.resize(image, dsize=list(map(lambda x: int(x * self.factor), image.shape))[:2],
                          interpolation=cv2.INTER_CUBIC)
        return cv2.resize(resized, dsize=image.shape[:2])

class Metrics:
    def __init__(self, images: tuple[np.ndarray, np.ndarray]):
        self.image1, self.image2 = images

    def _metrica_ssim(self):
        if self.image1.shape != self.image2.shape:
            return 'Images has different resolution'
        return metrics.structural_similarity(self.image1, self.image2, full=True, channel_axis=2)[0]

    def _metrica_mse(self):
        if self.image1.shape != self.image2.shape:
            return 'Images has different resolution'
        return metrics.mean_squared_error(self.image1, self.image2)

    def _metrica_psnr(self):
        if self.image1.shape != self.image2.shape:
            return 'Images has different resolution'
        mse = self._metrica_mse()
        if (mse == 0):
            return 100
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / self._metrica_rmse())
        return psnr

    def _metrica_rmse(self):
        if self.image1.shape != self.image2.shape:
            return 'Images has different resolution'
        return math.sqrt(self._metrica_mse())

    # def _metrica_ber(self):
    #     if self.image1.shape != self.image2.shape:
    #         return 'Images has different resolution'
    #     image1bits = np.unpackbits(np.array(self.image1, dtype='>i8').view(np.uint8))
    #     image2bits = np.unpackbits(np.array(self.image2, dtype='>i8').view(np.uint8))
    #     invalid_bits = 0
    #     total_bits = 0
    #     for b1, b2 in zip(image1bits, image2bits):
    #         if b1 == b2:
    #             invalid_bits += 1
    #         total_bits += 1
    #     return invalid_bits / total_bits

    def get(self):
        methods = [func for func in dir(self) if callable(getattr(self, func)) and not func.startswith("__") and func.startswith('_metrica_')]
        return {method.replace('_metrica_', ''): getattr(self, method)() for method in methods}

class Attacks:
    ATTACKS: tuple[BaseAttack] = (
        CompressionAttack(**{"filename": "compressed_image10.jpg", "quality": .1}),
        CompressionAttack(**{"filename": "compressed_image20.jpg", "quality": .2}),
        CompressionAttack(**{"filename": "compressed_image30.jpg", "quality": .3}),
        CompressionAttack(**{"filename": "compressed_image40.jpg", "quality": .4}),
        CompressionAttack(**{"filename": "compressed_image50.jpg", "quality": .5}),
        CompressionAttack(**{"filename": "compressed_image60.jpg", "quality": .6}),
        CompressionAttack(**{"filename": "compressed_image70.jpg", "quality": .7}),
        CompressionAttack(**{"filename": "compressed_image80.jpg", "quality": .8}),
        CompressionAttack(**{"filename": "compressed_image90.jpg", "quality": .9}),
        NoiseAttack(**{"filename": "noised_image001.png", "intensity": .01}),
        NoiseAttack(**{"filename": "noised_image005.png", "intensity": .05}),
        NoiseAttack(**{"filename": "noised_image01.png", "intensity": .1}),
        BlurAttack(**{"filename": "blurred_image3x3.png", "size": 3}),
        BlurAttack(**{"filename": "blurred_image5x5.png", "size": 5}),
        BlurAttack(**{"filename": "blurred_image4x4.png", "size": 7}),
        ScaleAttack(**{"filename": "scaled_image05.png", "factor": .5}),
        ScaleAttack(**{"filename": "scaled_image2.png", "factor": 2}),
    )

    @classmethod
    def execute(cls, image: np.ndarray):
        for attack in cls.ATTACKS:
            processed_image = attack.execute(image)
            #metrics = Metrics((image, processed_image))
            #print(metrics.get())
            attack.save(processed_image)


'''
if __name__ == "__main__":
    image = cv2.imread("image_with_watermark.png", cv2.IMREAD_UNCHANGED)
    Attacks.execute(image)
'''