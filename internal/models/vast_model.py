from typing import Union
import torch
import torch.nn.functional as F

from torch import nn

class UpsampleBlock(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(UpsampleBlock, self).__init__()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv = nn.Conv2d(num_input_channels // (2 * 2), num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pixel_shuffle(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class AppearanceNetwork(nn.Module):
    def __init__(self, num_input_channels, num_output_channels):
        super(AppearanceNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(num_input_channels, 256, 3, stride=1, padding=1)
        self.up1 = UpsampleBlock(256, 128)
        self.up2 = UpsampleBlock(128, 64)
        self.up3 = UpsampleBlock(64, 32)
        self.up4 = UpsampleBlock(32, 16)
        
        self.conv2 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, num_output_channels, 3, stride=1, padding=1)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        # bilinear interpolation
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        # x = self.sigmoid(x)
        return x

class VastModel(nn.Module):
    def __init__(
            self,
            n_appearance_count: int=6000,
            n_appearance_dims: int = 64,
            n_rgb_dims: int = 3,
            std: float = 1e-4,
    ) -> None:
        super().__init__()

        self._appearance_embeddings = nn.Parameter(torch.empty(n_appearance_count, n_appearance_dims).cuda())
        self._appearance_embeddings.data.normal_(0, std)
        self.appearance_network = AppearanceNetwork(n_rgb_dims+n_appearance_dims, n_rgb_dims).cuda()

    def forward(self, image, gt_image, view_idx):
        appearance_embedding = self.get_appearance(view_idx)

        origH, origW = image.shape[1:]
        H = origH // 32 * 32
        W = origW // 32 * 32
        left = origW // 2 - W // 2
        top = origH // 2 - H // 2
        crop_image = image[:, top:top+H, left:left+W]
        crop_gt_image = gt_image[:, top:top+H, left:left+W]
        
        # down sample the image
        crop_image_down = torch.nn.functional.interpolate(crop_image[None], size=(H//32, W//32), mode="bilinear", align_corners=True)[0]
        
        crop_image_down = torch.cat([crop_image_down, appearance_embedding[None].repeat(H//32, W//32, 1).permute(2, 0, 1)], dim=0)[None]
        mapping_image = self.appearance_network(crop_image_down)
        transformed_image = mapping_image * crop_image

        return transformed_image, crop_gt_image 

    def get_appearance(self, view_idx: Union[float, torch.Tensor]):
        return self._appearance_embeddings[view_idx]
