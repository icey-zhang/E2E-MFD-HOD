import os

import tqdm
from PIL import Image
import numpy as np
from scipy.io import loadmat, savemat

if __name__ == '__main__':
    path_ir = 'to/roadscene/ir'
    path_vi = 'to/roadscene/vi'
    path_out = './datasets/M3FD/JPEGImages_roadscene'
    os.makedirs(path_out, exist_ok=True)
    # for f in os.listdir(path_vi):
    for file in tqdm.tqdm(os.listdir(path_ir)):
        # for file in tqdm.tqdm(os.listdir(path_vi + '/' + f)):
        img_ir = np.array(Image.open(path_ir + '/' + file))
        img_vi = np.array(Image.open(path_vi + '/' + file))
        # name = file.split('_')[1]
        savemat(path_out + '/' +  file.replace('png', 'mat'), {'ir': img_ir, 'vi': img_vi})
        # savemat(path_out + '/' + file.replace('png', 'mat'), {'ir': img_ir, 'vi': img_vi})
