from emb import Image1, Embeding, Extract
from knn_regression import KNNregresion, StandardScaler
from attack import Attacks
from abc_alg import abc
import pandas as pd
import numpy as np
import os
import cv2
import pickle
import time



def main():
    path_im = 'test_data/'+input('input name_file: ')  # 'test_data/'
    path_wm = 'binary_watermark.png'
    size_im = 512
    size_wm = size_im//8
    im = Image1(path_im, size_im, size_im)
    wm = Image1(path_wm, size_wm, size_wm)
    s = input('Choose optimisation method: ')
    start_time = time.time()
    if s == 'knn':
        d = Embeding.get_feature_vector(path_im, path_wm, size_im, size_wm)
        with open('KNNregression_model.pkl', 'rb') as file:
            knn = pickle.load(file)
        F_obj = pd.DataFrame(d)
        normalizer = StandardScaler()
        F_obj_norm = normalizer.transform(F_obj)
        alpha = knn.predict_alpha(F_obj_norm)
    elif s == 'abc':
        alpha = abc.get_alpha_withABC(path_im, path_wm, size_im, size_wm)
    else:
        raise ValueError ('Invalid optimization method')
    end_time = time.time()
    execution_time = end_time - start_time
    print("\nOptimization time:", np.round(execution_time, 3), "sec\n")
    print('Optimal alpha:', alpha)

    metric_values = [Embeding.apply(im, wm, abc.corr(alpha), show_PSNR=True)]
    Attacks.execute(cv2.imread(im.emb_file, cv2.IMREAD_UNCHANGED))
    print('No attacks BCR:', Extract.apply(im.emb_file, im.size_x, im.size_y, saving=True))
    folder_path = './attack-result'
    file_list = os.listdir(folder_path)
    for file_name in file_list:
        metric_values.append(
            Extract.apply("attack-result/" + file_name, im.size_x, im.size_y, format=file_name.split('.')[-1],
                          show_BCR=True))
    Embeding.clear_dir()



if __name__=='__main__':
    main()