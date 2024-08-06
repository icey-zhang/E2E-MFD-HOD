import os

import cv2
import numpy as np
import tqdm
import scipy
import math
from prettytable import PrettyTable

from Metrics.Metric import Evaluator

def ComEntropy(img1, img2):
    if img2.shape != img1.shape:
        img2 = cv2.resize(img2, img1.shape[::-1])
    width = img1.shape[0]
    hegith = img1.shape[1]
    tmp = np.zeros((width, hegith))
    res = 0
    for i in range(width):
        for j in range(hegith):
            val1 = img1[i][j]
            val2 = img2[i][j]
            tmp[val1][val2] = float(tmp[val1][val2] + 1)
    tmp = tmp / (width * hegith)
    for i in range(width):
        for j in range(hegith):
            if tmp[i][j] == 0:
                res = res
            else:
                res = res - tmp[i][j] * (math.log(tmp[i][j] / math.log(2.0)))
    return res

def compare_vifp(ref, dist):
    if dist.shape != ref.shape:
        dist = cv2.resize(dist, ref.shape[::-1])

    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0
    for scale in range(1, 5):

        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        if (scale > 1):
            ref = scipy.ndimage.gaussian_filter(ref, sd)
            dist = scipy.ndimage.gaussian_filter(dist, sd)
            ref = ref[::2, ::2]
            dist = dist[::2, ::2]

        mu1 = scipy.ndimage.gaussian_filter(ref, sd)
        mu2 = scipy.ndimage.gaussian_filter(dist, sd)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = scipy.ndimage.gaussian_filter(ref * ref, sd) - mu1_sq
        sigma2_sq = scipy.ndimage.gaussian_filter(dist * dist, sd) - mu2_sq
        sigma12 = scipy.ndimage.gaussian_filter(ref * dist, sd) - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    if np.isnan(vifp):
        return 1.0
    else:
        return vifp

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
    path_in = '/home/data4/zjq/M3FD/M3FD_Fusion'
    result_txt_name = '/home/zjq/EfficientMFD/M3FD/results_difVIF_difMI.txt'
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
        
        MI = ComEntropy(ir, fi) + ComEntropy(vi, fi)
        SCD = Evaluator.SCD(fi, ir, vi)
        VIF = compare_vifp(ir, fi) + compare_vifp(vi, fi)
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

