import os
import math
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from utils.camera_utils import loadCam
from utils.loss_utils import l1_loss, ssim
from concurrent.futures import ThreadPoolExecutor

class GSDataset(Dataset):
    def __init__(self, cameras, scene, args, pipe=None, scale=1):
        self.cameras = cameras
        self.scale = scale
        self.args = args

          # initialise the linalg module for lazy loading
        torch.inverse(torch.ones((1, 1), device="cuda:0"))

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        c = self.cameras[idx]
        viewpoint_cam = loadCam(self.args, idx, c, self.scale)
        x = {
            "FoVx": viewpoint_cam.FoVx,
            "FoVy": viewpoint_cam.FoVy,
            "image_name": viewpoint_cam.image_name,
            "image_height": viewpoint_cam.image_height,
            "image_width": viewpoint_cam.image_width,
            "camera_center": viewpoint_cam.camera_center,
            "world_view_transform": viewpoint_cam.world_view_transform,
            "full_proj_transform": viewpoint_cam.full_proj_transform,
        }
        y = viewpoint_cam.original_image
        
        return x, y

class CacheDataLoader(torch.utils.data.DataLoader):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            max_cache_num: int,
            shuffle: bool,
            seed: int = -1,
            distributed: bool = False,
            world_size: int = -1,
            global_rank: int = -1,
            **kwargs,
    ):
        assert kwargs.get("batch_size", 1) == 1, "only batch_size=1 is supported"

        self.dataset = dataset

        super().__init__(dataset=dataset, **kwargs)

        self.shuffle = shuffle
        self.max_cache_num = max_cache_num

        # image indices to use
        self.indices = list(range(len(self.dataset)))
        if distributed is True and self.max_cache_num != 0:
            assert world_size > 0
            assert global_rank >= 0
            image_num_to_use = math.ceil(len(self.indices) / world_size)
            start = global_rank * image_num_to_use
            end = start + image_num_to_use
            indices = self.indices[start:end]
            indices += self.indices[:image_num_to_use - len(indices)]
            self.indices = indices

            print("#{} distributed indices (total: {}): {}".format(os.getpid(), len(self.indices), self.indices))

        # cache all images if max_cache_num > len(dataset)
        if self.max_cache_num >= len(self.indices):
            self.max_cache_num = -1

        self.num_workers = kwargs.get("num_workers", 0)

        if self.max_cache_num < 0:
            # cache all data
            print("cache all images")
            self.cached = self._cache_data(self.indices)

        # use dedicated random number generator foreach dataloader
        if self.shuffle is True:
            assert seed >= 0, "seed must be provided when shuffle=True"
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
            print("#{} dataloader seed to {}".format(os.getpid(), seed))

    def _cache_data(self, indices: list):
        # TODO: speedup image loading
        cached = []
        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as e:
                for i in tqdm(
                        e.map(self.dataset.__getitem__, indices),
                        total=len(indices),
                        desc="#{} caching images (1st: {})".format(os.getpid(), indices[0]),
                ):
                    cached.append(i)
        else:
            for i in tqdm(indices, desc="#{} loading images (1st: {})".format(os.getpid(), indices[0])):
                cached.append(self.dataset.__getitem__(i))

        return cached

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)

    def __iter__(self):
        # TODO: support batching
        if self.max_cache_num < 0:
            if self.shuffle is True:
                indices = torch.randperm(len(self.cached), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            elif self.shuffle is None and self.sampler is not None:
                indices = list(self.sampler)
            else:
                indices = list(range(len(self.cached)))

            for i in indices:
                yield self.cached[i]
        else:
            if self.shuffle is True:
                indices = torch.randperm(len(self.indices), generator=self.generator).tolist()  # shuffle for each epoch
                # print("#{} 1st index: {}".format(os.getpid(), indices[0]))
            elif self.shuffle is None and self.sampler is not None:
                indices = list(self.sampler)
            else:
                indices = self.indices.copy()

            # print("#{} self.max_cache_num={}, indices: {}".format(os.getpid(), self.max_cache_num, indices))

            if self.max_cache_num == 0:
                # no cache
                for i in indices:
                    yield self.__getitem__(i)
            else:
                # cache
                # the list contains the data have not been cached
                not_cached = indices.copy()

                while not_cached:
                    # select self.max_cache_num images
                    to_cache = not_cached[:self.max_cache_num]
                    del not_cached[:self.max_cache_num]

                    # cache
                    try:
                        del cached
                    except:
                        pass
                    cached = self._cache_data(to_cache)

                    for i in cached:
                        yield i