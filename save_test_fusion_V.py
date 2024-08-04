import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import tqdm
import cv2
import scipy
import numpy as np
import torch
# from Metrics.Metric import Evaluator
from detectron2.data import detection_utils as utils
from diffusiondet import detector
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup
from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs
import matplotlib.pyplot as plt
from PIL import Image
from scipy.io import loadmat, savemat


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':
    mode = '_m3fd' 
    device = 'cuda'
    args = default_argument_parser().parse_args()

    cfg = setup(args)

    model = detector.DiffusionDet(cfg, return_fusion=True).to(device)
    model_data = torch.load('./output0803_grad_enhance_att_loss0305_iter1000/model_0014999.pth')
    # key = model.load_state_dict(model_data['model'], strict=True)
    key = model.load_state_dict(model_data['model'], strict=False)
    model.eval()
    iter_name = '/fi_14999_detection{}/'.format(mode)
    iter_name_V = '/fi_14999_V_detection{}/'.format(mode)
    path_in = '/home/data4/zjq/M3FD/JPEGImages{}'.format(mode)
    names = open('/home/data4/zjq/M3FD/ImageSets/Main/fusion{}.txt'.format(mode), 'r').read().splitlines()
    path_out = './output0803_grad_enhance_att_loss0305_iter1000/fusion_result/'
    # path_out_V = './output0306_grad/fusion_result/fi_09999_V/'
    # os.makedirs(path_out + '/vi', exist_ok=True)
    # os.makedirs(path_out + '/ir', exist_ok=True)
    os.makedirs(path_out + iter_name, exist_ok=True)
    os.makedirs(path_out + iter_name_V, exist_ok=True)

    time_sum = 0
    for name in tqdm.tqdm(names):
        img_in = path_in + '/' + name + '.mat'
        image_ir, image_ir_rgb, image_vi, visimage_bri, visimage_clr = utils.read_image(img_in, format=cfg.INPUT.FORMAT)
        # ir, ir_rgb, vi, visimage_bri, visimage_clr
        image = torch.as_tensor(np.ascontiguousarray(image_vi.transpose(2, 0, 1)))
        ir = torch.as_tensor(np.ascontiguousarray(image_ir.transpose(2, 0, 1)))
        ir_rgb = torch.as_tensor(np.ascontiguousarray(image_ir_rgb.transpose(2, 0, 1)))
        vi = torch.as_tensor(np.ascontiguousarray(image_vi.transpose(2, 0, 1)))
        visimage_bri = torch.as_tensor(np.ascontiguousarray(visimage_bri.transpose(2, 0, 1)))
        visimage_clr = torch.as_tensor(np.ascontiguousarray(visimage_clr.transpose(2, 0, 1)))

        with torch.no_grad():
            image = image.to(device)
            inputs = [{"image": image, "ir": ir, "ir_rgb": ir_rgb, 'vi': vi,
                       'visimage_bri': visimage_bri,
                       'visimage_clr': visimage_clr}]
            V_img,output,time_per_img = model.fusion_run(inputs)
            time_sum += time_per_img
            ############ 在这里保存V空间的图片 ###########
            bri = V_img.detach().cpu().numpy() * 255
            bri = bri.reshape([V_img.size()[2], V_img.size()[3]])
            bri = np.where(bri < 0, 0, bri)
            bri = np.where(bri > 255, 255, bri)
            im1 = Image.fromarray(bri.astype(np.uint8))

            im1.save(path_out +iter_name_V + name + '.png')
            ###############################################

        ############ 在这里保存RGB2HSV空间的图片 ###########
        vi = vi.cpu().numpy().transpose(1, 2, 0)
        if output.shape[:2] != vi.shape[:2]:
            output = cv2.resize(output, vi.shape[:2][::-1])
        output = output[..., ::-1]
        # vi = vi[..., ::-1]
        # cv2.imwrite(path_out + '/vi/' + name + '.png', vi)
        # cv2.imwrite(path_out + '/ir/' + name + '.png', ir)
        cv2.imwrite(path_out + iter_name + name + '.png', output)
        ###############################################
    print(time_sum)
