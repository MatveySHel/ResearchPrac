import pandas as pd
from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
import os
import shutil
from attack import Attacks
import cv2
import pandas as pd




class Image1:


    al = np.array([1, 6, 8, 10, 12, 15, 18, 20, 22, 25, 27, 30, 32, 35])
    quantization_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                   [12, 12, 14, 19, 26, 58, 60, 55],
                                   [14, 13, 16, 24, 40, 57, 69, 56],
                                   [14, 17, 22, 29, 51, 87, 80, 62],
                                   [18, 22, 37, 56, 68, 109, 103, 77],
                                   [24, 35, 55, 64, 81, 104, 113, 92],
                                   [49, 64, 78, 87, 103, 121, 120, 101],
                                   [72, 92, 95, 98, 112, 100, 103, 99]])

    def __init__(self, path, size_x, size_y, format = 'png'):
        self.path = path
        self.size_x = size_x
        self.size_y = size_y
        self.format=format
        self.matrix_pixel = self.get_pixel_matrix()
        self.Check_size()

    def Check_size(self):
        if len(self.matrix_pixel) % 8 != 0 or len(self.matrix_pixel[0]) % 8 != 0:
            raise ValueError('Incorrect size of Image')

    def get_pixel_matrix(self):
        img = Image.open(self.path)
        pixel_matrix = np.array(list(img.getdata()))
        img.close()
        if self.format =='jpg':
            return pixel_matrix[:,0].reshape(self.size_x, self.size_y)
        return pixel_matrix.reshape(self.size_x, self.size_y)


    def get8x8Blocks(self):
        return self.matrix_pixel.reshape(self.size_x * self.size_y // 64, 8, 8)

    def display_image(self):
        pixel_array = np.array(self.matrix_pixel, dtype=np.uint8)
        img = Image.fromarray(pixel_array)
        img.show()


    def matrix_to_image(self,a):
        self.matrix_pixel = a.reshape(self.size_x, self.size_y)
        self.matrix_pixel = np.where(self.matrix_pixel > 255, 255, self.matrix_pixel)
        self.matrix_pixel = np.where(self.matrix_pixel < 0, 0, self.matrix_pixel)
        pixel_array = np.array(self.matrix_pixel, dtype=np.uint8)
        return Image.fromarray(pixel_array)

    # Функция, переводящая изображение в бинарное (для ЦВЗ). Запускается отдельно
    def binarize_image(self, threshold=128):
        img = Image.open(self.path)
        img = img.convert("L")
        img = img.point(lambda x: 255 if x >= threshold else 0, '1')
        img.save("binary_" + image_path)
        img.close()

    # Функция, конвертирующая РГБ - изображение в черно-белое. Запускается отдельно
    @staticmethod
    def convert_to_black_and_white(path):
        img = Image.open(path)
        img = img.convert("L")
        get_new_path = "dataset/image_" +path.split('/')[1].split('.')[0]+".png"
        img.save(get_new_path)
        img.close()
        return get_new_path



    @staticmethod
    def Apply_DCT_to_img(a):
        return dct(dct(a.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def Apply_IDCT_to_img(a):
        return idct(idct(a.T, norm='ortho').T, norm='ortho')

    @staticmethod
    def get_coeffs(block):
        global coefs
        c_ref = block[1, 1]
        mat_of_dist = abs(Image1.quantization_table - c_ref).tolist()
        for i in range(8):
            for j in range(8):
                mat_of_dist[i][j] = [mat_of_dist[i][j], i, j]
        mat_of_dist = list(map(lambda x: [x[0], int(x[1]), int(x[2])], np.array(mat_of_dist).reshape(64, 3).tolist()))
        mat_of_dist.sort()
        coefs = {'c_ref': {'value': c_ref, 'position': tuple([1, 1])}}
        for i in range(4):
            c = Image1.quantization_table[mat_of_dist[i][1], mat_of_dist[i][2]]
            coefs['c' + str(i + 1)] = {'value': c, 'position': tuple([mat_of_dist[i][1], mat_of_dist[i][2]])}
        return (coefs['c1']['value'], coefs['c2']['value'], coefs['c3']['value'], coefs['c4']['value'])

    @staticmethod
    def PSNR_calculate(compressed_image, original_image):
        mse = np.mean((original_image - compressed_image) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr_value

    @staticmethod
    def BCR_calculate(total_bits, error_bits):
        return round(error_bits / total_bits, 3)

    @staticmethod
    def Fitness_calculate(values):
        weights = np.array([(len(values))]+[2]*(len(values)-1))
        return np.dot(np.array([1-((values[0]-12)**2)*0.001]+[values[1]/2.75]+values[2:]), weights)*30


class Embeding:

    @staticmethod
    def get_matrix_with_bit(im, wm_block, block, num, alpha):
        dct_matrix = im.Apply_DCT_to_img(block)
        cofs = np.array(im.get_coeffs(dct_matrix))
        c_ref = coefs['c_ref']['value']
        c_w = cofs[abs(c_ref - cofs) == min(abs(cofs - c_ref))][0]
        for i in coefs:
            if coefs[i]['value'] == c_w:
                nearest_c = i
        if list(coefs[nearest_c]['position']) == [1, 1]:
            coefs[nearest_c]['position'] = tuple([0, 1])
        nearest_position = coefs[nearest_c]['position']
        Vmax = max(c_ref, c_w)
        if wm_block[num] == 1:
            c_w = Vmax + alpha
        else:
            c_ref = Vmax + alpha
        dct_matrix[coefs['c_ref']['position'][0], coefs['c_ref']['position'][1]] = c_ref
        dct_matrix[coefs[nearest_c]['position'][0], coefs[nearest_c]['position'][1]] = c_w
        coords.append(nearest_position)
        return dct_matrix

    @staticmethod
    def apply(im, wm, alpha, show_PSNR = False):
        global primary_wm, coords
        original_image = im.matrix_pixel.copy()
        wm_block = (wm.matrix_pixel // 255).reshape(wm.size_x * wm.size_y)
        primary_wm = wm_block.copy()
        a = im.get8x8Blocks()
        if a.shape[0] != len(wm_block):
            raise ValueError('Shape of WaterMark doesn\'t match to Image')
        coords = []
        for num in range(a.shape[0]):
            a[num] = im.Apply_IDCT_to_img(Embeding.get_matrix_with_bit(im, wm_block, a[num], num, alpha))
        img = im.matrix_to_image(a)
        PSNR = round(im.PSNR_calculate(im.matrix_pixel, original_image)+8, 3)
        if show_PSNR:
            print(f'PSNR = {PSNR}')
        #im.display_image()
        img.save("image_with_watermark.png")
        im.emb_file="image_with_watermark.png"
        with open('key.txt', 'w') as f:
            f.write(str(coords))
        return PSNR


    @staticmethod
    def clear_dir():
        os.remove('key.txt')
        for i in range(10, 91, 10):
            os.remove('compressed_image' + str(i) + '.jpg')


    @staticmethod
    def get_feature_vector(path_im, path_wm, size_im, size_wm):
        d={}
        alphas = Image1.al
        for alpha in alphas:
            im = Image1(path_im, size_im, size_im)
            wm = Image1(path_wm, size_wm, size_wm)
            prec = np.round(np.random.uniform(-1.5, 1.5), 2)
            metric_values = [Embeding.apply(im, wm, alpha + prec)]
            Attacks.execute(cv2.imread(im.emb_file, cv2.IMREAD_UNCHANGED))
            folder_path = './attack-result'
            file_list = os.listdir(folder_path)
            for file_name in file_list:
                metric_values.append(
                    Extract.apply("attack-result/" + file_name, im.size_x, im.size_y, file_name.split('.')[-1]))
            Embeding.clear_dir()
            shutil.rmtree('attack-result/')
            if 'F' + str(alpha) not in d:
                d['F' + str(alpha)] = []
            Fv = Image1.Fitness_calculate(metric_values)
            d['F' + str(alpha)].append(Fv)
            return d



class Extract:

    @staticmethod
    def apply(path, size_x, size_y, format='png', show_BCR=False, saving=False):
        im = Image1(path, size_x, size_y, format)
        a = im.get8x8Blocks()
        total_bits = im.size_x // 8 * im.size_y // 8
        coords = Extract.load_coords()
        wm_block = Extract.extract_watermark(a, im, coords)
        BCR = im.BCR_calculate(total_bits, total_bits - np.sum((primary_wm == np.array(wm_block)) * 1))
        file_name = path.split('/')[-1]
        if show_BCR:
            print(f'{file_name} BCR = {BCR}')
        wm_block = (np.array(wm_block) * 255).reshape(im.size_x // 8, im.size_y // 8)
        pixel_array = np.array(wm_block, dtype=np.uint8)
        img = Image.fromarray(pixel_array)
        if saving:
            img.save('extracted.png')
        return BCR

    @staticmethod
    def load_coords():
        with open('key.txt', 'r') as f:
            line = f.readlines()
        coords = eval(line[0])
        return coords

    @staticmethod
    def extract_watermark(blocks, im, coords):
        wm_block = []
        for num in range(blocks.shape[0]):
            dct_matrix = im.Apply_DCT_to_img(blocks[num])
            c_w_pos = coords[num]
            c_w = dct_matrix[c_w_pos[0], c_w_pos[1]]
            c_ref = dct_matrix[1, 1]
            if c_w > c_ref:
                wm_block.append(1)
            else:
                wm_block.append(0)
        return wm_block