import os

import cv2
import numpy as np
import tqdm
from prettytable import PrettyTable

from Metrics.Metric import Evaluator


def nor(data):
    data -= np.min(data)
    data = data / (np.max(data) + 1e-3)
    return data


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path)
    # img_BGR = cv2.resize(img_BGR, (int(img_BGR.shape[0]/2), int(img_BGR.shape[0]/2))).astype('float32')
    # assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb' or mode == 'V', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    elif mode == 'V':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        img = np.round(img[..., 2])
    elif mode == 'IR':
        # img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        img = np.round(img_BGR[..., 0])

    return img#.astype(np.uint8)

if __name__ == '__main__':
    path_in = to/M3FD/M3FD_Fusion'
    result_txt_name = 'results.txt'
    path_in_fi = './output0803_grad_enhance_att_loss0305_iter1000/fusion_result/fi_14999_V_detection/'
    names = os.listdir(path_in + '/Vis')
    table = PrettyTable(['name', 'EN', 'SD', 'SF', 'MI', 'SCD', 'VIF', 'Qabf', 'SSIM'])
    MIs, ENs, VIFs, SDs, SCDs, Qabfs, SSIMs, SFs = [], [], [], [], [], [], [], []
    # result_path = path_in + '/result.txt'
    for name in tqdm.tqdm(names):
        result_txt = open(result_txt_name, 'w')
        ir_file = path_in + '/Ir/' + name
        vi_file = path_in + '/Vis/' + name
        fi_file = path_in_fi + name
        ir = image_read_cv2(ir_file, 'V')
        vi = image_read_cv2(vi_file, 'V')
        fi = image_read_cv2(fi_file, 'V')
        if fi.shape[:2] != vi.shape[:2]:
            fi = cv2.resize(fi, vi.shape[:2][::-1])

        EN = Evaluator.EN(fi)
        SD = Evaluator.SD(fi)
        SF = Evaluator.SF(fi)
        MI = Evaluator.MI(fi, ir, vi)
        SCD = Evaluator.SCD(fi, ir, vi)
        VIF = Evaluator.VIFF(fi, ir, vi)
        Qabf = Evaluator.Qabf(fi, ir, vi)
        SSIM = Evaluator.SSIM(fi, ir, vi)

        table.add_row([name, EN, SD, SF, MI, SCD, VIF, Qabf, SSIM])
        ENs.append(EN)
        SDs.append(SD)
        SFs.append(SF)
        MIs.append(MI)
        SCDs.append(SCD)
        VIFs.append(VIF)
        Qabfs.append(Qabf)
        SSIMs.append(SSIM)
        result_txt.write(str(table))
    result_txt = open(result_txt_name, 'w')
    table.add_row(['mean', np.mean(ENs), np.mean(SDs), np.mean(SFs), np.mean(MIs),
                   np.mean(SCDs), np.mean(VIFs), np.mean(Qabfs), np.mean(SSIMs)])
    print('mean', np.mean(ENs), np.mean(SDs), np.mean(SFs), np.mean(MIs),
                np.mean(SCDs), np.mean(VIFs), np.mean(Qabfs), np.mean(SSIMs))
    result_txt.write(str(table))
