"""
Description: 阿尔兹海默症NPZ格式数据加载器
Author: Modified for Alzheimer's NPZ data
Date: 2025/1/25
"""
import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from scipy import ndimage
from scipy.ndimage import zoom
from torchvision import transforms
from timm.data import Mixup


def random_rot_flip(image):
    """随机旋转和翻转图像"""
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image


def random_rotate(image):
    """随机旋转图像"""
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False)
    return image


class AlzheimerTransform(object):
    """阿尔兹海默症数据的变换类"""
    def __init__(self, output_size, is_train=True, normalize=True):
        self.output_size = output_size
        self.is_train = is_train
        self.normalize = normalize
        # 医学图像通常使用不同的归一化参数
        self.mean = 0.5
        self.std = 0.5

    def __call__(self, sample):
        image = sample['slice']
        label = sample['label']
        change_label = sample['change_label']

        # 数据增强（仅在训练时）
        if self.is_train:
            if random.random() > 0.5:
                image = random_rot_flip(image)
            elif random.random() > 0.5:
                image = random_rotate(image)

        # 调整图像大小
        h, w = image.shape
        if h != self.output_size[0] or w != self.output_size[1]:
            zoom_h = self.output_size[0] / h
            zoom_w = self.output_size[1] / w
            image = zoom(image, (zoom_h, zoom_w), order=3)

        # 转换为float32并归一化到[0, 1]
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image = image / image.max()

        # 扩展为3通道（复制灰度图到RGB）
        image = np.stack([image, image, image], axis=0)

        # 归一化
        if self.normalize:
            image = (image - self.mean) / self.std

        # 转换为tensor
        image = torch.from_numpy(image).float()
        label = torch.tensor(label, dtype=torch.long)
        change_label = torch.tensor(change_label, dtype=torch.long)

        return {
            'image': image,
            'label': label,
            'change_label': change_label,
            'case_name': sample['case_name']
        }


class AlzheimerNPZDataset(Dataset):
    """阿尔兹海默症NPZ数据集"""
    def __init__(self, data_dir, split='train', transform=None, file_list=None):
        """
        Args:
            data_dir: NPZ文件所在的目录
            split: 'train', 'val', 或 'test'
            transform: 数据变换
            file_list: 文件列表路径，如果为None则自动查找所有NPZ文件
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform

        # 获取文件列表
        if file_list and os.path.exists(file_list):
            # 从文件列表读取
            with open(file_list, 'r') as f:
                self.files = [line.strip() for line in f.readlines()]
                # 确保文件名以.npz结尾
                self.files = [f if f.endswith('.npz') else f + '.npz' for f in self.files]
        else:
            # 自动查找所有NPZ文件
            self.files = glob.glob(os.path.join(data_dir, '*.npz'))
            self.files = [os.path.basename(f) for f in self.files]

        if len(self.files) == 0:
            raise ValueError(f"No NPZ files found in {data_dir}")

        print(f"Found {len(self.files)} NPZ files for {split} set")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载NPZ文件
        file_name = self.files[idx]
        file_path = os.path.join(self.data_dir, file_name)

        try:
            data = np.load(file_path)
            sample = {
                'slice': data['slice'],
                'label': int(data['label'][()]) if data['label'].shape == () else int(data['label']),
                'change_label': int(data['change_label'][()]) if data['change_label'].shape == () else int(data['change_label']),
                'prior': data['prior'] if 'prior' in data else np.zeros(3),
                'case_name': os.path.splitext(file_name)[0]
            }
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            raise

        # 应用变换
        if self.transform:
            sample = self.transform(sample)

        return sample


def build_loader_finetune(config):
    """构建微调阶段的数据加载器"""
    config.defrost()

    # 设置类别数（根据您的任务调整）
    # 如果是二分类（AD vs 正常），设为2
    # 如果是多分类（AD, MCI, 正常等），相应调整
    config.MODEL.NUM_CLASSES = 2  # 可以根据实际情况调整

    config.freeze()

    # 构建训练和验证数据集
    transform_train = AlzheimerTransform(
        output_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
        is_train=True,
        normalize=True
    )

    transform_val = AlzheimerTransform(
        output_size=(config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
        is_train=False,
        normalize=True
    )

    # 数据路径设置
    train_dir = os.path.join(config.DATA.DATA_PATH, 'train')
    val_dir = os.path.join(config.DATA.DATA_PATH, 'val')

    # 如果没有train/val子目录，则使用同一目录并按比例分割
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("Train/val directories not found, using data split from single directory")
        all_files = glob.glob(os.path.join(config.DATA.DATA_PATH, '*.npz'))

        # 随机分割数据（80%训练，20%验证）
        random.shuffle(all_files)
        split_idx = int(0.8 * len(all_files))
        train_files = [os.path.basename(f) for f in all_files[:split_idx]]
        val_files = [os.path.basename(f) for f in all_files[split_idx:]]

        dataset_train = AlzheimerNPZDataset(
            data_dir=config.DATA.DATA_PATH,
            split='train',
            transform=transform_train,
            file_list=None
        )
        dataset_train.files = train_files

        dataset_val = AlzheimerNPZDataset(
            data_dir=config.DATA.DATA_PATH,
            split='val',
            transform=transform_val,
            file_list=None
        )
        dataset_val.files = val_files
    else:
        # 使用分离的目录
        dataset_train = AlzheimerNPZDataset(
            data_dir=train_dir,
            split='train',
            transform=transform_train
        )

        dataset_val = AlzheimerNPZDataset(
            data_dir=val_dir,
            split='val',
            transform=transform_val
        )

    # 分布式采样器
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()

    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    # 数据加载器
    data_loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # Mixup设置
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP,
            cutmix_alpha=config.AUG.CUTMIX,
            cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB,
            switch_prob=config.AUG.MIXUP_SWITCH_PROB,
            mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING,
            num_classes=config.MODEL.NUM_CLASSES
        )

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


# 测试函数
if __name__ == "__main__":
    # 测试数据集加载
    import matplotlib.pyplot as plt

    # 创建一个简单的配置对象用于测试
    class SimpleConfig:
        class DATA:
            IMG_SIZE = 256
            DATA_PATH = r"D:\codebase\Swin-Transformer\examples"
            BATCH_SIZE = 4
            NUM_WORKERS = 0
            PIN_MEMORY = True

        class AUG:
            MIXUP = 0
            CUTMIX = 0
            CUTMIX_MINMAX = None
            MIXUP_PROB = 0
            MIXUP_SWITCH_PROB = 0
            MIXUP_MODE = 'batch'

        class MODEL:
            LABEL_SMOOTHING = 0.0
            NUM_CLASSES = 2

    config = SimpleConfig()

    # 测试数据集
    transform = AlzheimerTransform(output_size=(256, 256), is_train=True)
    dataset = AlzheimerNPZDataset(
        data_dir=config.DATA.DATA_PATH,
        split='test',
        transform=transform
    )

    # 加载一个样本
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Label: {sample['label']}")
        print(f"Change label: {sample['change_label']}")
        print(f"Case name: {sample['case_name']}")

        # 可视化
        img = sample['image'].numpy()
        # 反归一化用于显示
        img = img * 0.5 + 0.5
        img = np.transpose(img, (1, 2, 0))

        plt.figure(figsize=(8, 8))
        plt.imshow(img[:, :, 0], cmap='gray')
        plt.title(f"Sample: {sample['case_name']}\nLabel: {sample['label']}, Change: {sample['change_label']}")
        plt.colorbar()
        plt.show()