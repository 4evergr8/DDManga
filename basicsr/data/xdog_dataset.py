import cv2
import random
import time
import numpy as np
import torch
from torch.utils import data as data

from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY
from basicsr.data.fmix import sample_mask

@DATASET_REGISTRY.register()
class XDoGDataset(data.Dataset):
    """
    Dataset 用于输入灰度图+XDoG处理图，输出原始灰度图
    """

    def __init__(self, opt):
        super(XDoGDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.gt_folder = opt['dataroot_gt']

        meta_info_file = self.opt['meta_info_file']
        assert meta_info_file is not None
        if not isinstance(meta_info_file, list):
            meta_info_file = [meta_info_file]
        self.paths = []
        for meta_info in meta_info_file:
            with open(meta_info, 'r') as fin:
                self.paths.extend([line.strip() for line in fin])

        self.do_fmix = opt['do_fmix']
        self.fmix_params = {'alpha': 1., 'decay_power': 3., 'shape': (256, 256), 'max_soft': 0.0, 'reformulate': False}
        self.fmix_p = opt['fmix_p']
        self.do_cutmix = opt['do_cutmix']
        self.cutmix_params = {'alpha': 1.}
        self.cutmix_p = opt['cutmix_p']

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        gt_size = self.opt['gt_size']
        retry = 3
        while retry > 0:
            try:
                img_bytes = self.file_client.get(gt_path, 'gt')
            except Exception as e:
                logger = get_root_logger()
                logger.warn(f'File client error: {e}, remaining retry times: {retry - 1}')
                index = random.randint(0, self.__len__() - 1)
                gt_path = self.paths[index]
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1

        img_gt = imfrombytes(img_bytes, float32=True)
        img_gt = cv2.resize(img_gt, (gt_size, gt_size))

        # 转灰度图
        img_gray = cv2.cvtColor(img_gt, cv2.COLOR_BGR2GRAY)

        # -------------------------------- (Optional) CutMix & FMix -------------------------------- #
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > self.fmix_p:
            with torch.no_grad():
                lam, mask = sample_mask(**self.fmix_params)

                fmix_index = random.randint(0, self.__len__() - 1)
                fmix_img_path = self.paths[fmix_index]
                fmix_img_bytes = self.file_client.get(fmix_img_path, 'gt')
                fmix_img = imfrombytes(fmix_img_bytes, float32=True)
                fmix_img = cv2.resize(fmix_img, (gt_size, gt_size))
                fmix_gray = cv2.cvtColor(fmix_img, cv2.COLOR_BGR2GRAY)

                mask = mask.transpose(1, 2, 0)  # (1, H, W) -> (H, W, 1)
                img_gray = mask[:, :, 0] * img_gray + (1. - mask[:, :, 0]) * fmix_gray
                img_gray = img_gray.astype(np.float32)

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > self.cutmix_p:
            with torch.no_grad():
                cmix_index = random.randint(0, self.__len__() - 1)
                cmix_img_path = self.paths[cmix_index]
                cmix_img_bytes = self.file_client.get(cmix_img_path, 'gt')
                cmix_img = imfrombytes(cmix_img_bytes, float32=True)
                cmix_img = cv2.resize(cmix_img, (gt_size, gt_size))
                cmix_gray = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2GRAY)

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']), 0.3, 0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox(cmix_gray.shape, lam)

                img_gray[bbx1:bbx2, bby1:bby2] = cmix_gray[bbx1:bbx2, bby1:bby2]

        # 归一化灰度图
        img_gray_norm = img_gray.astype(np.float32) / 255.

        # 计算XDoG输入通道
        img_xdog = self.apply_xdog(img_gray_norm)

        # 输入为两通道，灰度图和XDoG图
        input_img = np.stack([img_gray_norm, img_xdog], axis=0)  # shape: (2, H, W)
        target_img = np.expand_dims(img_gray_norm, axis=0)     # shape: (1, H, W)

        input_tensor = torch.from_numpy(input_img).float()
        target_tensor = torch.from_numpy(target_img).float()

        return {
            'lq': input_tensor,
            'gt': target_tensor,
            'lq_path': gt_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

    def apply_xdog(self, gray_img):
        blur1 = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=0.3)
        blur2 = cv2.GaussianBlur(gray_img, (0, 0), sigmaX=0.48)  # 0.3 * 1.6 = 0.48
        diff = blur1 - 0.98 * blur2
        epsilon = 0.005
        phi = 20
        xdog = np.where(diff >= epsilon, 1.0, 1.0 + np.tanh(phi * (diff - epsilon)))
        return (xdog + 1) / 2


def rand_bbox(size, lam):
    '''cutmix 的 bbox 截取函数
    Args:
        size : tuple 图片尺寸 e.g (256,256)
        lam  : float 截取比例
    Returns:
        bbox 的左上角和右下角坐标
        int,int,int,int
    '''
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2
