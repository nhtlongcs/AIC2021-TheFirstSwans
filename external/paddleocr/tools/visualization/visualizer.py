import cv2
import random
import numpy as np
from PIL import Image
from typing import Optional
from .utils import draw_mask, draw_polylines, draw_text, get_font_size, reduce_opacity

class Visualizer():
    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.class_names = None

    def set_image(self, image: np.ndarray) -> None:
        self.image = image

    def get_image(self):
        if self.image.dtype == 'uint8':
            self.image = np.clip(self.image, 0, 255)
        elif self.image.dtype == 'float32':
            self.image = np.clip(self.image, 0.0, 1.0)
            self.image = (self.image*255).astype(np.uint8)

        return self.image

    def draw_polygon_ocr(self, polygons, texts=None, font='./doc/fonts/aachenb.ttf'):
        image = self.image.copy()/255.0
        maskIm = Image.new('L', (self.image.shape[1], self.image.shape[0]), 0)
        white_img = np.zeros(image.shape)
        if texts is not None:
            zipped = zip(polygons, texts)
        else:
            zipped = polygons

        for item in zipped: 
            if texts is not None:
                polygon, text = item
            else:
                polygon, text = item, None

            maskIm = draw_mask(polygon, maskIm) 
            image = draw_polylines(image, polygon)

            if text:
                font_size = get_font_size(image, text, polygon, font)
                color = tuple([random.randint(0,255) for _ in range(3)])
                white_img = draw_text(white_img, text, polygon, font, color, font_size)

        # Mask out polygons
        mask = np.stack([maskIm, maskIm, maskIm], axis=2)
        masked = image * mask

        # Reduce opacity of original image
        o_image = reduce_opacity(image)
        i_masked = (np.bitwise_not(mask)/255).astype(np.int)
        o_image = o_image * i_masked

        # Add two image
        new_img = o_image + masked

        new_img = new_img.astype(np.float32)

        if texts:
            white_img = white_img.astype(np.float32)
            stacked = np.concatenate([new_img, white_img], axis=1)
            self.image = stacked.copy()
        else:
            self.image = new_img.copy()

