import os
import random
import time

import numpy as np
import pandas as pd

import torch
import torchvision
from torchvision.transforms import v2

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch.use_deterministic_algorithms(True)

if torch.cuda.is_available():
    torch.set_default_device("cuda:0")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class SqueezeAndExcitation(torch.nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.projection = torch.nn.Conv2d(
            num_features,
            num_features // 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        torch.nn.init.xavier_uniform_(self.projection.weight, gain=np.sqrt(2.0))
        torch.nn.init.zeros_(self.projection.bias)

        self.expansion = torch.nn.Conv2d(
            num_features // 2,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        torch.nn.init.xavier_uniform_(self.expansion.weight)
        torch.nn.init.zeros_(self.expansion.bias)

    def forward(self, x):
        y = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        y = self.projection(y)
        y = torch.nn.functional.relu(y)
        y = self.expansion(y)
        y = torch.nn.functional.hardsigmoid(y)
        return x * y


class InvertedBottleneckBlock(torch.nn.Module):

    def __init__(
        self,
        num_blocks,
        num_features,
        norm,
        activation,
        expansion_norm,
        expansion_activation,
        depthwise_filter_size,
        depthwise_norm,
        depthwise_activation,
        attention,
        projection_norm
    ):
        super().__init__()

        if expansion_norm is True:
            self.expansion = torch.nn.Conv2d(
                num_features,
                num_features * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True if norm is None else False
            )
            if expansion_activation:
                torch.nn.init.xavier_uniform_(self.expansion.weight, gain=np.sqrt(2.0))
            else:
                torch.nn.init.xavier_uniform_(self.expansion.weight)
            if self.expansion.bias is not None:
                torch.nn.init.zeros_(self.expansion.bias)

            if norm == "batch":
                self.expansion_norm = torch.nn.BatchNorm2d(num_features * 2, eps=1.0e-3)
            elif norm == "layer":
                self.expansion_norm = torch.nn.GroupNorm(1, num_features * 2, eps=1.0e-3)
            elif norm == "group":
                self.expansion_norm = torch.nn.GroupNorm(
                    num_features * 2 // 4,
                    num_features * 2,
                    eps=1.0e-3
                )
            elif norm == "instance":
                self.expansion_norm = torch.nn.GroupNorm(
                    num_features * 2,
                    num_features * 2,
                    eps=1.0e-3
                )
            elif norm is None:
                self.expansion_norm = None
            else:
                raise
        elif expansion_norm is False:
            self.expansion = torch.nn.Conv2d(
                num_features,
                num_features * 2,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
            if expansion_activation:
                torch.nn.init.xavier_uniform_(self.expansion.weight, gain=np.sqrt(2.0))
            else:
                torch.nn.init.xavier_uniform_(self.expansion.weight)
            torch.nn.init.zeros_(self.expansion.bias)

            self.expansion_norm = None
        else:
            raise

        self.expansion_activation = expansion_activation

        if depthwise_norm is True:
            self.depthwise = torch.nn.Conv2d(
                num_features * 2,
                num_features * 2,
                groups=num_features * 2,
                kernel_size=depthwise_filter_size,
                stride=1,
                padding="same",
                bias=True if norm is None else False
            )
            if depthwise_activation:
                torch.nn.init.xavier_uniform_(self.depthwise.weight, gain=np.sqrt(2.0))
            else:
                torch.nn.init.xavier_uniform_(self.depthwise.weight)
            if self.depthwise.bias is not None:
                torch.nn.init.zeros_(self.depthwise.bias)

            if norm == "batch":
                self.depthwise_norm = torch.nn.BatchNorm2d(num_features * 2, eps=1.0e-3)
            elif norm == "layer":
                self.depthwise_norm = torch.nn.GroupNorm(1, num_features * 2, eps=1.0e-3)
            elif norm == "group":
                self.depthwise_norm = torch.nn.GroupNorm(
                    num_features * 2 // 4,
                    num_features * 2,
                    eps=1.0e-3
                )
            elif norm == "instance":
                self.depthwise_norm = torch.nn.GroupNorm(
                    num_features * 2,
                    num_features * 2,
                    eps=1.0e-3
                )
            elif norm is None:
                self.depthwise_norm = None
            else:
                raise
        elif depthwise_norm is False:
            self.depthwise = torch.nn.Conv2d(
                num_features * 2,
                num_features * 2,
                groups=num_features * 2,
                kernel_size=depthwise_filter_size,
                stride=1,
                padding="same",
                bias=True
            )
            if depthwise_activation:
                torch.nn.init.xavier_uniform_(self.depthwise.weight, gain=np.sqrt(2.0))
            else:
                torch.nn.init.xavier_uniform_(self.depthwise.weight)
            torch.nn.init.zeros_(self.depthwise.bias)

            self.depthwise_norm = None
        else:
            raise

        self.depthwise_activation = depthwise_activation

        if attention == "squeeze":
            self.attention = SqueezeAndExcitation(num_features * 2)
        elif attention is None:
            self.attention = None
        else:
            raise

        if projection_norm is True:
            self.projection = torch.nn.Conv2d(
                num_features * 2,
                num_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True if norm is None else False
            )
            torch.nn.init.xavier_uniform_(self.projection.weight, gain=1.0/np.sqrt(num_blocks))
            if self.projection.bias is not None:
                torch.nn.init.zeros_(self.projection.bias)

            if norm == "batch":
                self.projection_norm = torch.nn.BatchNorm2d(num_features, eps=1.0e-3)
            elif norm == "layer":
                self.projection_norm = torch.nn.GroupNorm(1, num_features, eps=1.0e-3)
            elif norm == "group":
                self.projection_norm = torch.nn.GroupNorm(
                    num_features // 4,
                    num_features,
                    eps=1.0e-3
                )
            elif norm == "instance":
                self.projection_norm = torch.nn.GroupNorm(num_features, num_features, eps=1.0e-3)
            elif norm is None:
                self.projection_norm = None
            else:
                raise
        elif projection_norm is False:
            self.projection = torch.nn.Conv2d(
                num_features * 2,
                num_features,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )
            torch.nn.init.xavier_uniform_(self.projection.weight, gain=1.0/np.sqrt(num_blocks))
            torch.nn.init.zeros_(self.projection.bias)

            self.projection_norm = None
        else:
            raise

        if activation == "hard-swish":
            self.activation = torch.nn.functional.hardswish
        elif activation == "relu":
            self.activation = torch.nn.functional.relu
        else:
            raise

    def forward(self, x):
        y = self.expansion(x)
        if self.expansion_norm is not None:
            y = self.expansion_norm(y)
        if self.expansion_activation:
            y = self.activation(y)

        y = self.depthwise(y)
        if self.depthwise_norm is not None:
            y = self.depthwise_norm(y)
        if self.depthwise_activation:
            y = self.activation(y)

        if self.attention:
            y = self.attention(y)

        y = self.projection(y)
        if self.projection_norm is not None:
            y = self.projection_norm(y)

        return x + y


class HierarchicalIntroBlock(torch.nn.Module):

    def __init__(self, num_features, norm):
        super().__init__()

        self.strided = torch.nn.Conv2d(
            1,
            num_features,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True if norm is None else False
        )
        torch.nn.init.xavier_uniform_(self.strided.weight)
        if self.strided.bias is not None:
            torch.nn.init.zeros_(self.strided.bias)

        if norm == "batch":
            self.norm = torch.nn.BatchNorm2d(num_features, eps=1.0e-3)
        elif norm == "layer":
            self.norm = torch.nn.GroupNorm(1, num_features, eps=1.0e-3)
        elif norm == "group":
            self.norm = torch.nn.GroupNorm(num_features // 4, num_features, eps=1.0e-3)
        elif norm == "instance":
            self.norm = torch.nn.GroupNorm(num_features, num_features, eps=1.0e-3)
        elif norm is None:
            self.norm = None
        else:
            raise

    def forward(self, x):
        y = self.strided(x)
        if self.norm is not None:
            y = self.norm(y)
        return y


class HierarchicalDownsamplingBlock(torch.nn.Module):

    def __init__(self, num_features, norm):
        super().__init__()

        self.conv = torch.nn.Conv2d(
            num_features,
            num_features * 2,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True if norm is None else False
        )
        torch.nn.init.xavier_uniform_(self.conv.weight)
        if self.conv.bias is not None:
            torch.nn.init.zeros_(self.conv.bias)

        if norm == "batch":
            self.norm = torch.nn.BatchNorm2d(num_features * 2, eps=1.0e-3)
        elif norm == "layer":
            self.norm = torch.nn.GroupNorm(1, num_features * 2, eps=1.0e-3)
        elif norm == "group":
            self.norm = torch.nn.GroupNorm(num_features * 2 // 4, num_features * 2, eps=1.0e-3)
        elif norm == "instance":
            self.norm = torch.nn.GroupNorm(num_features * 2, num_features * 2, eps=1.0e-3)
        elif norm is None:
            self.norm = None
        else:
            raise

    def forward(self, x):
        y = self.conv(x)
        if self.norm is not None:
            y = self.norm(y)
        return y


class HierarchicalUpsamplingBlock(torch.nn.Module):

    def __init__(self, num_blocks, num_features, norm):
        super().__init__()

        self.transposed_conv = torch.nn.ConvTranspose2d(
            num_features,
            num_features // 2,
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True if norm is None else False
        )
        torch.nn.init.xavier_uniform_(self.transposed_conv.weight)
        if self.transposed_conv.bias is not None:
            torch.nn.init.zeros_(self.transposed_conv.bias)

        if norm == "batch":
            self.norm = torch.nn.BatchNorm2d(num_features // 2, eps=1.0e-3)
        elif norm == "layer":
            self.norm = torch.nn.GroupNorm(1, num_features // 2, eps=1.0e-3)
        elif norm == "group":
            self.norm = torch.nn.GroupNorm(num_features // 2 // 4, num_features // 2, eps=1.0e-3)
        elif norm == "instance":
            self.norm = torch.nn.GroupNorm(num_features // 2, num_features // 2, eps=1.0e-3)
        elif norm is None:
            self.norm = None
        else:
            raise

    def forward(self, x):
        y = self.transposed_conv(x)
        if self.norm is not None:
            y = self.norm(y)
        return y


class HierarchicalOutroBlock(torch.nn.Module):

    def __init__(self, num_features, channels_out, activation):
        super().__init__()

        self.expansion = torch.nn.Conv2d(
            num_features,
            num_features * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        torch.nn.init.xavier_uniform_(self.expansion.weight, gain=np.sqrt(2.0))
        torch.nn.init.zeros_(self.expansion.bias)

        if activation == "hard-swish":
            self.activation = torch.nn.functional.hardswish
        elif activation == "relu":
            self.activation = torch.nn.functional.relu
        else:
            raise

        self.dropout = torch.nn.Dropout2d(p = 0.05)

        self.linear = torch.nn.Conv2d(
            num_features * 2,
            channels_out,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        y = self.expansion(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear(y)
        return y


class HierarchicalModel(torch.nn.Module):

    def __init__(
        self,
        num_blocks,
        num_features,
        num_keypoints,
        norm,
        activation,
        expansion_norm,
        expansion_activation,
        depthwise_filter_size,
        depthwise_norm,
        depthwise_activation,
        attention,
        projection_norm
    ):
        super().__init__()

        self.intro_block = HierarchicalIntroBlock(num_features // 4, norm)

        self.middle_block_1 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 4,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )
        self.middle_block_2 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 4,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )

        self.downsampling_block_1 = HierarchicalDownsamplingBlock(num_features // 4, norm)

        self.middle_block_3 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 2,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )
        self.middle_block_4 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 2,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )

        self.downsampling_block_2 = HierarchicalDownsamplingBlock(num_features // 2, norm)

        self.middle_block_5 = InvertedBottleneckBlock(
            num_blocks,
            num_features,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )
        self.middle_block_6 = InvertedBottleneckBlock(
            num_blocks,
            num_features,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )

        self.upsampling_block_1 = HierarchicalUpsamplingBlock(num_blocks, num_features, norm)

        self.middle_block_7 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 2,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )
        self.middle_block_8 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 2,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )

        self.upsampling_block_2 = HierarchicalUpsamplingBlock(num_blocks, num_features // 2, norm)

        self.middle_block_9 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 4,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )
        self.middle_block_10 = InvertedBottleneckBlock(
            num_blocks,
            num_features // 4,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )

        self.outro_block = HierarchicalOutroBlock(
            num_features // 4,
            num_keypoints,
            activation
        )

    def forward(self, x):
        intro = self.intro_block(x)

        middle_1 = self.middle_block_1(intro)
        middle_2 = self.middle_block_2(middle_1)

        downsampling_1 = self.downsampling_block_1(middle_2)

        middle_3 = self.middle_block_3(downsampling_1)
        middle_4 = self.middle_block_4(middle_3)

        downsampling_2 = self.downsampling_block_2(middle_4)

        middle_5 = self.middle_block_5(downsampling_2)
        middle_6 = self.middle_block_6(middle_5)

        upsampling_1 = self.upsampling_block_1(middle_6)

        middle_7 = self.middle_block_7(upsampling_1)
        middle_8 = self.middle_block_8(middle_7)

        upsampling_2 = self.upsampling_block_2(middle_8)

        middle_9 = self.middle_block_9(upsampling_2)
        middle_10 = self.middle_block_10(middle_9)

        outro = self.outro_block(middle_10)

        return outro


class IsotropicIntroBlock(torch.nn.Module):

    def __init__(self, patch_size, num_features, norm):
        super().__init__()

        self.pixel_unshuffle = torch.nn.PixelUnshuffle(patch_size)

        self.pointwise = torch.nn.Conv2d(
            1 * patch_size * patch_size,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True if norm is None else False
        )
        torch.nn.init.xavier_uniform_(self.pointwise.weight)
        if self.pointwise.bias is not None:
            torch.nn.init.zeros_(self.pointwise.bias)

        if norm == "batch":
            self.norm = torch.nn.BatchNorm2d(num_features, eps=1.0e-3)
        elif norm == "layer":
            self.norm = torch.nn.GroupNorm(1, num_features, eps=1.0e-3)
        elif norm == "group":
            self.norm = torch.nn.GroupNorm(num_features // 4, num_features, eps=1.0e-3)
        elif norm == "instance":
            self.norm = torch.nn.GroupNorm(num_features, num_features, eps=1.0e-3)
        elif norm is None:
            self.norm = None
        else:
            raise

    def forward(self, x):
        y = self.pixel_unshuffle(x)
        y = self.pointwise(y)
        if self.norm is not None:
            y = self.norm(y)
        return y


class IsotropicOutroBlock(torch.nn.Module):

    def __init__(self, patch_size, num_features, channels_out, activation):
        super().__init__()

        self.expansion = torch.nn.Conv2d(
            num_features,
            num_features * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        torch.nn.init.xavier_uniform_(self.expansion.weight, gain=np.sqrt(2.0))
        torch.nn.init.zeros_(self.expansion.bias)

        if activation == "hard-swish":
            self.activation = torch.nn.functional.hardswish
        elif activation == "relu":
            self.activation = torch.nn.functional.relu
        else:
            raise

        self.dropout = torch.nn.Dropout2d(p = 0.05)

        self.linear = torch.nn.Conv2d(
            num_features * 2,
            channels_out * (patch_size // 2 * patch_size // 2),
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)

        self.pixel_shuffle = torch.nn.PixelShuffle(patch_size // 2)

    def forward(self, x):
        y = self.expansion(x)
        y = self.activation(y)
        y = self.dropout(y)
        y = self.linear(y)
        y = self.pixel_shuffle(y)
        return y


class IsotropicModel(torch.nn.Module):

    def __init__(
        self,
        patch_size,
        num_blocks,
        num_features,
        num_keypoints,
        norm,
        activation,
        expansion_norm,
        expansion_activation,
        depthwise_filter_size,
        depthwise_norm,
        depthwise_activation,
        attention,
        projection_norm
    ):
        super().__init__()

        self.intro_block = IsotropicIntroBlock(patch_size, num_features, norm)

        self.blocks = torch.nn.ParameterList()
        for i in range(num_blocks):
            self.blocks.append(
                InvertedBottleneckBlock(
                    num_blocks,
                    num_features,
                    norm,
                    activation,
                    expansion_norm,
                    expansion_activation,
                    depthwise_filter_size,
                    depthwise_norm,
                    depthwise_activation,
                    attention,
                    projection_norm
                )
            )

        self.outro_block = IsotropicOutroBlock(
            patch_size,
            num_features,
            num_keypoints,
            activation
        )

    def forward(self, x):
        y = self.intro_block(x)
        for block in self.blocks:
            y = block(y)
        y = self.outro_block(y)
        return y


def split_dataset(path, training_validation_split = 0.75, undersampling = None):
    paths = []
    for filename in os.listdir(path):
        base, extension = os.path.splitext(filename)
        if extension == ".png":
            paths.append(os.path.join(path, base))

    paths = sorted(paths)
    np.random.shuffle(paths)

    num_training = round(training_validation_split * len(paths))
    num_validation = len(paths) - num_training

    training = paths[:num_training]
    validation = paths[num_training:]

    if undersampling is None:
        pass
    elif undersampling == "half":
        subsample_size = len(training) // 2
        training = training[:subsample_size]
    elif undersampling == "three-quarters":
        subsample_size = len(training) // 4
        training = training[:subsample_size]
    else:
        raise

    return training, validation


def read_coords(path):
    with torch.no_grad():
        dataframe = pd.read_csv(path, index_col=0, dtype=np.float32)
        return torch.tensor(dataframe.to_numpy())


def draw_gaussians(height, width, coords, sigma):
    with torch.no_grad():
        x_grid, y_grid = torch.meshgrid(torch.arange(width), torch.arange(height), indexing="xy")

        gaussians = torch.zeros([coords.shape[0], height, width], dtype=torch.float32)
        for c in range(coords.shape[0]):
            y = torch.square(y_grid + 0.5 - coords[c, 0])
            x = torch.square(x_grid + 0.5 - coords[c, 1])
            gaussians[c, :, :] = torch.exp(-(y + x) / (2.0 * sigma**2.0))

        return gaussians


class Dataset(torch.utils.data.Dataset):

    def __init__(
        self,
        paths,
        input_size,
        output_size,
        num_keypoints,
        gaussian_sigma,
        image_mean = None,
        image_std = None,
        augmentation = False
    ):
        with torch.no_grad():
            self.paths = paths
            self.input_size = input_size
            self.output_size = output_size
            self.num_keypoints = num_keypoints
            self.gaussian_sigma = gaussian_sigma
            self.image_mean = image_mean
            self.image_std = image_std
            self.augmentation = augmentation

            # Cache the mean and std of an example gaussian
            test_gaussian = draw_gaussians(
                self.output_size,
                self.output_size,
                torch.tensor(
                    [[self.output_size / 2.0, self.output_size / 2.0]],
                    dtype=torch.float32
                ),
                self.gaussian_sigma
            )
            self.gaussian_mean = test_gaussian.mean().item()
            self.gaussian_std = test_gaussian.std().item()

            # Estimate the mean and std across images (if precomputed values not supplied)
            if self.image_mean is None and self.image_std is None:
                image_means = torch.zeros((len(self.paths)), dtype=torch.float32)
                image_stds = torch.zeros((len(self.paths)), dtype=torch.float32)
                for i in range(len(self.paths)):
                    image = torchvision.io.read_image(
                        self.paths[i] + ".png",
                        torchvision.io.ImageReadMode.GRAY
                    )
                    image = torchvision.transforms.v2.functional.convert_image_dtype(
                        image,
                        dtype=torch.float32
                    )
                    image_means[i] = torch.mean(image, axis=(1, 2))
                    image_stds[i] = torch.std(image, axis=(1, 2))
                self.image_mean = torch.mean(image_means)
                self.image_std = torch.mean(image_stds)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        with torch.no_grad():
            # Retrieve image and coords
            image = torchvision.io.read_image(
                self.paths[index] + ".png",
                torchvision.io.ImageReadMode.GRAY
            ).to(device="cuda:0" if torch.cuda.is_available() else "cpu")
            coords = read_coords(self.paths[index] + ".csv")

            # Convert image to floating point
            image = torchvision.transforms.v2.functional.convert_image_dtype(
                image,
                dtype=torch.float32
            )

            # Cache original image dimensions
            original_height = image.shape[1]
            original_width = image.shape[2]

            if not self.augmentation:
                # Resize image
                coords[:, 0] *= self.input_size / original_height
                coords[:, 1] *= self.input_size / original_width
                image = torchvision.transforms.v2.functional.resize(
                    image,
                    (self.input_size, self.input_size),
                    torchvision.transforms.v2.InterpolationMode.BILINEAR
                )

                # Convert coordinates from input space to output space
                coords *= self.output_size / self.input_size

                # Draw gaussians
                gaussians = draw_gaussians(
                    self.output_size,
                    self.output_size,
                    coords,
                    self.gaussian_sigma
                )

                # Standardize image and gaussians
                image -= self.image_mean
                image /= self.image_std
                gaussians -= self.gaussian_mean
                gaussians /= self.gaussian_std

                return image, gaussians, coords
            else:
                # Randomly shift gamma
                image = torch.pow(image, np.random.uniform(3.0/4.0, 4.0/3.0))

                # Randomly shift brightness
                image += np.random.uniform(-0.1, 0.1)

                # Cache image mean to use as fill value during padding and rotation
                mean = image.mean().item()

                # Randomly resize image dimensions
                scale = np.random.uniform(0.9375, 1.0625)

                resized_height = round(self.input_size * scale)
                resized_width = round(self.input_size * scale)

                image = torchvision.transforms.v2.functional.resize(
                    image,
                    (resized_height, resized_width),
                    torchvision.transforms.v2.InterpolationMode.BILINEAR
                )
                coords[:, 0] *= resized_height / original_height
                coords[:, 1] *= resized_width / original_width

                # Randomly flip vertically
                if np.random.randint(0, 2):
                    image = torchvision.transforms.v2.functional.vflip(image)
                    coords[:, 0] = image.shape[1] - coords[:, 0]

                # Randomly flip horizontally
                if np.random.randint(0, 2):
                    image = torchvision.transforms.v2.functional.hflip(image)
                    coords[:, 1] = image.shape[2] - coords[:, 1]

                # Randomly rotate
                degrees = np.random.uniform(-45.0, 45.0)
                image = torchvision.transforms.v2.functional.rotate(
                    image,
                    degrees,
                    torchvision.transforms.v2.InterpolationMode.BILINEAR,
                    expand = True,
                    fill = mean
                )

                radians = np.deg2rad(degrees)
                cos_radians = np.cos(-radians)
                sin_radians = np.sin(-radians)

                coords[:, 0] -= resized_height / 2.0
                coords[:, 1] -= resized_width / 2.0

                rotated_y = cos_radians * coords[:, 0] + sin_radians * coords[:, 1]
                rotated_x = -sin_radians * coords[:, 0] + cos_radians * coords[:, 1]

                coords[:, 0] = rotated_y
                coords[:, 1] = rotated_x

                coords[:, 0] += image.shape[1] / 2.0
                coords[:, 1] += image.shape[2] / 2.0

                # Randomly pad as needed
                vertical_padding = max(0, self.input_size - image.shape[1])
                top_padding = np.random.randint(0, vertical_padding + 1)
                bottom_padding = vertical_padding - top_padding

                horizontal_padding = max(0, self.input_size - image.shape[2])
                left_padding = np.random.randint(0, horizontal_padding + 1)
                right_padding = horizontal_padding - left_padding

                if vertical_padding or horizontal_padding:
                    image = torchvision.transforms.v2.functional.pad(
                        image,
                        padding=[left_padding, top_padding, right_padding, bottom_padding],
                        fill=mean
                    )
                    coords[:, 0] += top_padding
                    coords[:, 1] += left_padding

                # Randomly crop as needed
                vertical_excess = max(0, image.shape[1] - self.input_size)
                vertical_crop_start = np.random.randint(0, vertical_excess + 1)

                horizontal_excess = max(0, image.shape[2] - self.input_size)
                horizontal_crop_start = np.random.randint(0, horizontal_excess + 1)

                if vertical_excess or horizontal_excess:
                    image = torchvision.transforms.v2.functional.crop(
                        image,
                        vertical_crop_start,
                        horizontal_crop_start,
                        min(self.input_size, image.shape[1] - vertical_crop_start),
                        min(self.input_size, image.shape[2] - horizontal_crop_start)
                    )
                    coords[:, 0] -= vertical_crop_start
                    coords[:, 1] -= horizontal_crop_start

                # Convert coordinates from input space to output space
                coords *= self.output_size / self.input_size

                # Randomly "wiggle" coordinates
                angles = torch.distributions.uniform.Uniform(0.0, 360.0).sample((coords.shape[0],))
                thetas = torch.deg2rad(angles)
                y_scales = torch.distributions.uniform.Uniform(0.0, 0.25).sample((coords.shape[0],))
                x_scales = torch.distributions.uniform.Uniform(0.0, 0.25).sample((coords.shape[0],))
                coords[:, 0] += y_scales * torch.sin(thetas)
                coords[:, 1] += x_scales * torch.cos(thetas)

                # Draw gaussians
                gaussians = draw_gaussians(
                    self.output_size,
                    self.output_size,
                    coords,
                    self.gaussian_sigma
                )

                # Standardize image and gaussians
                image -= self.image_mean
                image /= self.image_std
                gaussians -= self.gaussian_mean
                gaussians /= self.gaussian_std

                # Randomly add Gaussian noise to image
                image += torch.distributions.normal.Normal(0.0, 0.1).sample(image.shape)

                return image, gaussians, coords


def training_loop(
    data_loader,
    model,
    model_ema,
    loss_function,
    optimizer,
    epoch,
    batch_size,
    accumulation_size,
    scheduler
):
    losses = np.zeros(len(data_loader) * accumulation_size // batch_size, dtype=np.float32)

    elapsed_time = 0.0

    model.train(True)
    batch_index = 0
    accumulation_losses = np.zeros(batch_size // accumulation_size, dtype=np.float32)
    for accumulation_index, (x, y, c) in enumerate(data_loader):
        start = time.time()

        prediction = model(x)
        loss = loss_function(prediction, y)

        accumulation_losses[
            accumulation_index % (batch_size // accumulation_size)
        ] = loss.detach().cpu().item()

        loss.backward()
        if (accumulation_index + 1) * accumulation_size % batch_size == 0:
            optimizer.step()

            for parameter in model.parameters():
                parameter.grad = None

            if scheduler is not None:
                scheduler.step()

            losses[batch_index] = np.mean(accumulation_losses)
            batch_index += 1

        finish = time.time()
        elapsed_time += finish - start

        if (accumulation_index + 1) * accumulation_size % batch_size == 0:
            model_ema.update_parameters(model)

    torch.optim.swa_utils.update_bn(data_loader, model_ema)

    return np.mean(losses), elapsed_time


def evaluation_loop(data_loader, model, loss_function, batch_size):
    losses = np.zeros(len(data_loader), dtype=np.float32)

    elapsed_time = 0.0

    model.train(False)
    with torch.no_grad():
        for i, (x, y, c) in enumerate(data_loader):
            start = time.time()
            prediction = model(x)
            finish = time.time()
            elapsed_time += finish - start

            loss = loss_function(prediction, y)

            losses[i] = loss.detach().cpu().item()

    return np.mean(losses), elapsed_time


def distance_loop(data_loader, model, loss_function, batch_size, num_keypoints):
    distances = np.zeros((batch_size * len(data_loader), num_keypoints), dtype=np.float32)

    model.train(False)
    with torch.no_grad():
        for i, (x, y, c) in enumerate(data_loader):
            prediction = model(x)

            c = torch.floor(c) + 0.5

            predicted_coords = torch.zeros(c.shape, dtype=torch.float32)
            for ii in range(prediction.shape[0]):
                for jj in range(prediction.shape[1]):
                    argmax = torch.argmax(prediction[ii, jj, :, :])
                    predicted_coords[ii, jj, 0] = argmax // y.shape[3]
                    predicted_coords[ii, jj, 1] = argmax % y.shape[3]
            predicted_coords += 0.5
            distances[i * batch_size:(i * batch_size) + batch_size, :] = torch.hypot(
                c[:, :, 0] - predicted_coords[:, :, 0],
                c[:, :, 1] - predicted_coords[:, :, 1]
            ).detach().cpu().numpy()

    return distances


def set_random_seed(nominal):
    random.seed(nominal)
    effective = random.randint(0, 2**32 - 1)

    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)


def train_model(
    seed,
    dataset,
    training_data_ablation,
    architecture,
    patch_size,
    num_blocks,
    num_features,
    norm,
    activation,
    expansion_norm,
    expansion_activation,
    depthwise_filter_size,
    depthwise_norm,
    depthwise_activation,
    attention,
    projection_norm,
    training_batch_size,
    gradient_accumulation_size,
    evaluation_batch_size,
    learning_rate,
    num_epochs
):
    print("Configuration:")
    print(f"seed: {seed}")
    print(f"dataset: {dataset}")
    print(f"training_data_ablation: {training_data_ablation}")
    print(f"architecture: {architecture}")
    print(f"patch_size: {patch_size}")
    print(f"num_blocks: {num_blocks}")
    print(f"num_features: {num_features}")
    print(f"norm: {norm}")
    print(f"activation: {activation}")
    print(f"expansion_norm: {expansion_norm}")
    print(f"expansion_activation: {expansion_activation}")
    print(f"depthwise_filter_size: {depthwise_filter_size}")
    print(f"depthwise_norm: {depthwise_norm}")
    print(f"depthwise_activation: {depthwise_activation}")
    print(f"attention: {attention}")
    print(f"projection_norm: {projection_norm}")
    print(f"training_batch_size: {training_batch_size}")
    print(f"gradient_accumulation_size: {gradient_accumulation_size}")
    print(f"evaluation_batch_size: {evaluation_batch_size}")
    print(f"learning_rate: {learning_rate}")
    print(f"num_epochs: {num_epochs}")

    path = "generated/"
    path += f"{seed};"
    path += f"{dataset};"
    path += f"{training_data_ablation};"
    path += f"{architecture};"
    path += f"{patch_size};"
    path += f"{num_blocks};"
    path += f"{num_features};"
    path += f"{norm};"
    path += f"{activation};"
    path += f"{expansion_norm};"
    path += f"{expansion_activation};"
    path += f"{depthwise_filter_size};"
    path += f"{depthwise_norm};"
    path += f"{depthwise_activation};"
    path += f"{attention};"
    path += f"{projection_norm};"
    path += f"{training_batch_size};"
    path += f"{gradient_accumulation_size};"
    path += f"{evaluation_batch_size};"
    path += f"{learning_rate};"
    path += f"{num_epochs}"

    if not os.path.exists(path):
        os.makedirs(path)

    print("\nPath:")
    print(f"{path}")

    set_random_seed(seed)

    # Partition dataset into training and validation splits, and save splits to disk
    if dataset == "visuomotor":
        training_data, validation_data = split_dataset(
            "visuomotor-dataset",
            0.75,
            training_data_ablation
        )
        input_size = 256
        output_size = input_size // 2
        num_keypoints = 1
        gaussian_sigma = 2.0
    elif dataset == "touch-evoked":
        training_data, validation_data = split_dataset(
            "touch-evoked-dataset",
            0.75,
            training_data_ablation
        )
        input_size = 512
        output_size = input_size // 2
        num_keypoints = 7
        gaussian_sigma = 2.0
    else:
        raise

    with open(os.path.join(path, "training_images.csv"), "w") as csv:
        csv.write("index,path\n")
        for i in range(len(training_data)):
            csv.write(f"{i},{training_data[i]}.png\n")
    with open(os.path.join(path, "validation_images.csv"), "w") as csv:
        csv.write("index,path\n")
        for i in range(len(validation_data)):
            csv.write(f"{i},{validation_data[i]}.png\n")

    # Set up model and its exponential moving average
    if architecture == "hierarchical":
        model = HierarchicalModel(
            num_blocks,
            num_features,
            num_keypoints,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )
    elif architecture == "isotropic":
        model = IsotropicModel(
            patch_size,
            num_blocks,
            num_features,
            num_keypoints,
            norm,
            activation,
            expansion_norm,
            expansion_activation,
            depthwise_filter_size,
            depthwise_norm,
            depthwise_activation,
            attention,
            projection_norm
        )
    else:
        raise

    model = model.to(device="cuda:0" if torch.cuda.is_available() else "cpu")
    model_ema = torch.optim.swa_utils.AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99)
    )

    print("\nParameters:")
    print(f"{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Set up datasets and data loaders
    training_dataset = Dataset(
        training_data,
        input_size,
        output_size,
        num_keypoints,
        gaussian_sigma,
        image_mean=None,
        image_std=None,
        augmentation=True
    )
    training_data_loader = torch.utils.data.DataLoader(
        training_dataset,
        batch_size=gradient_accumulation_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator(device="cuda:0" if torch.cuda.is_available() else "cpu")
    )

    training_eval_dataset = Dataset(
        training_data,
        input_size,
        output_size,
        num_keypoints,
        gaussian_sigma,
        image_mean=training_dataset.image_mean,
        image_std=training_dataset.image_std,
        augmentation=False
    )
    training_eval_data_loader = torch.utils.data.DataLoader(
        training_eval_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator(device="cuda:0" if torch.cuda.is_available() else "cpu")
    )

    validation_eval_dataset = Dataset(
        validation_data,
        input_size,
        output_size,
        num_keypoints,
        gaussian_sigma,
        image_mean=training_dataset.image_mean,
        image_std=training_dataset.image_std,
        augmentation=False
    )
    validation_eval_data_loader = torch.utils.data.DataLoader(
        validation_eval_dataset,
        batch_size=evaluation_batch_size,
        shuffle=False,
        drop_last=False,
        generator=torch.Generator(device="cuda:0" if torch.cuda.is_available() else "cpu")
    )

    # Set up AdamW optimizer with biases and norm weights excluded from weight decay
    beta_1 = 0.9
    beta_2 = 0.95
    eps = 1.0e-6
    weight_decay = 1.0e-5

    parameters = []
    for key, value in model.named_parameters():
        if "bias" in key or "norm" in key:
            parameters.append({"params": value, "weight_decay": 0.0})
        else:
            parameters.append({"params": value, "weight_decay": weight_decay / learning_rate})

    optimizer = torch.optim.AdamW(
        parameters,
        lr = learning_rate,
        betas = (beta_1, beta_2),
        eps = eps,
        weight_decay = weight_decay / learning_rate
    )

    # Set up learning rate warmup
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0e-6,
        end_factor=1.0,
        total_iters=np.ceil(2.0 / (1.0 - beta_2)),
        last_epoch=-1
    )

    # Training loop
    training_losses = np.full(num_epochs, np.nan)
    training_eval_losses = np.full(num_epochs, np.nan)
    validation_eval_losses = np.full(num_epochs, np.nan)
    training_eval_losses_ema = np.full(num_epochs, np.nan)
    validation_eval_losses_ema = np.full(num_epochs, np.nan)

    training_times = np.full(num_epochs, np.nan)
    training_eval_times = np.full(num_epochs, np.nan)
    validation_eval_times = np.full(num_epochs, np.nan)
    training_eval_times_ema = np.full(num_epochs, np.nan)
    validation_eval_times_ema = np.full(num_epochs, np.nan)

    loss_function = torch.nn.MSELoss()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch}")

        start = time.time()

        training_loss, training_time = training_loop(
            training_data_loader,
            model,
            model_ema,
            loss_function,
            optimizer,
            epoch,
            training_batch_size,
            gradient_accumulation_size,
            scheduler
        )
        print(
            f"training_loss:            "
            f"{training_loss:>11.8f} [t = {training_time:>12.8f}]"
        )

        training_eval_loss, training_eval_time = evaluation_loop(
            training_eval_data_loader, model, loss_function, evaluation_batch_size
        )
        print(
            f"training_eval_loss:       "
            f"{training_eval_loss:>11.8f} [t = {training_eval_time:>12.8f}]"
        )

        validation_eval_loss, validation_eval_time = evaluation_loop(
            validation_eval_data_loader, model, loss_function, evaluation_batch_size
        )
        print(
            f"validation_eval_loss:     "
            f"{validation_eval_loss:>11.8f} [t = {validation_eval_time:>12.8f}]"
        )

        training_eval_loss_ema, training_eval_time_ema = evaluation_loop(
            training_eval_data_loader, model_ema, loss_function, evaluation_batch_size
        )
        print(
            f"training_eval_loss_ema:   "
            f"{training_eval_loss_ema:>11.8f} [t = {training_eval_time_ema:>12.8f}]"
        )

        validation_eval_loss_ema, validation_eval_time_ema = evaluation_loop(
            validation_eval_data_loader, model_ema, loss_function, evaluation_batch_size
        )
        print(
            f"validation_eval_loss_ema: "
            f"{validation_eval_loss_ema:>11.8f} [t = {validation_eval_time_ema:>12.8f}]"
        )

        finish = time.time()
        elapsed_time = finish - start
        print(f"[t = {elapsed_time:>12.8f}]")

        training_losses[epoch] = training_loss
        training_eval_losses[epoch] = training_eval_loss
        validation_eval_losses[epoch] = validation_eval_loss
        training_eval_losses_ema[epoch] = training_eval_loss_ema
        validation_eval_losses_ema[epoch] = validation_eval_loss_ema

        training_times[epoch] = training_time
        training_eval_times[epoch] = training_eval_time
        validation_eval_times[epoch] = validation_eval_time
        training_eval_times_ema[epoch] = training_eval_time_ema
        validation_eval_times_ema[epoch] = validation_eval_time_ema

        # Save recorded losses to disk
        with open(f"{path}/losses.csv", "w") as csv:
            csv.write(
                "epoch,"
                "training_loss,"
                "training_eval_loss,"
                "validation_eval_loss,"
                "training_eval_loss_ema,"
                "validation_eval_loss_ema\n"
            )

            for i, j, k, l, m, n in zip(
                range(num_epochs),
                training_losses,
                training_eval_losses,
                validation_eval_losses,
                training_eval_losses_ema,
                validation_eval_losses_ema
            ):
                csv.write(f"{i},{j:.8f},{k:.8f},{l:.8f},{m:.8f},{n:.8f}\n")

        # Save recorded times to disk
        with open(f"{path}/times.csv", "w") as csv:
            csv.write("epoch,training_time,validation_eval_time\n")
            for i, j, k, l, m, n in zip(
                range(num_epochs),
                training_times,
                training_eval_times,
                validation_eval_times,
                training_eval_times_ema,
                validation_eval_times_ema
            ):
                csv.write(f"{i},{j:.8f},{k:.8f},{l:.8f},{m:.8f},{n:.8f}\n")

        # Save model(s) to disk if best or only ones so far
        if epoch == 0 or validation_eval_losses[epoch] <= validation_eval_losses[epoch - 1]:
            torch.save(model.state_dict(), os.path.join(path, "model.pt"))
        if epoch == 0 or validation_eval_losses_ema[epoch] <= validation_eval_losses_ema[epoch - 1]:
            torch.save(model_ema.state_dict(), os.path.join(path, "model_ema.pt"))

    # Reload best models, evaluate distances from predictions to ground truths, and write to disk
    model.load_state_dict(torch.load(os.path.join(path, "model.pt"), weights_only=True))
    model_ema.load_state_dict(torch.load(os.path.join(path, "model_ema.pt"), weights_only=True))

    training_eval_distances = distance_loop(
        training_eval_data_loader, model, loss_function, evaluation_batch_size, num_keypoints
    )
    validation_eval_distances = distance_loop(
        validation_eval_data_loader, model, loss_function, evaluation_batch_size, num_keypoints
    )
    with open(os.path.join(path, "training_distances.csv"), "w") as csv:
        csv.write("image,keypoint,distance\n")
        for i in range(training_eval_distances.shape[0]):
            for j in range(training_eval_distances.shape[1]):
                csv.write(f"{i},{j},{training_eval_distances[i, j]:.8f}\n")
    with open(os.path.join(path, "validation_distances.csv"), "w") as csv:
        csv.write("image,keypoint,distance\n")
        for i in range(validation_eval_distances.shape[0]):
            for j in range(validation_eval_distances.shape[1]):
                csv.write(f"{i},{j},{validation_eval_distances[i, j]:.8f}\n")

    training_eval_distances_ema = distance_loop(
        training_eval_data_loader, model_ema, loss_function, evaluation_batch_size, num_keypoints
    )
    validation_eval_distances_ema = distance_loop(
        validation_eval_data_loader, model_ema, loss_function, evaluation_batch_size, num_keypoints
    )
    with open(os.path.join(path, "training_distances_ema.csv"), "w") as csv:
        csv.write("image,keypoint,distance\n")
        for i in range(training_eval_distances_ema.shape[0]):
            for j in range(training_eval_distances_ema.shape[1]):
                csv.write(f"{i},{j},{training_eval_distances_ema[i, j]:.8f}\n")
    with open(os.path.join(path, "validation_distances_ema.csv"), "w") as csv:
        csv.write("image,keypoint,distance\n")
        for i in range(validation_eval_distances_ema.shape[0]):
            for j in range(validation_eval_distances_ema.shape[1]):
                csv.write(f"{i},{j},{validation_eval_distances_ema[i, j]:.8f}\n")

    print("\nDone!")

    return model, model_ema
