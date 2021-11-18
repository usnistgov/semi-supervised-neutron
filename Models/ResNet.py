#Credit to Haotong Liang (AuroraLHT)

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import OrderedDict

class ModelOutput:
    def __init__(self, logits, loss=None):
        self.logits=logits
        self.loss=loss
        self.predictions=torch.flatten(torch.argmax(logits, dim=-1))
    def __str__(self):
        return str({"loss":self.loss, "predictions":self.predictions, "logits": self.logits})
    def accuracy(self, labels):
        assert labels.shape==self.predictions.shape, "Predictions and labels do not have the same shape"
        accuracy=(torch.sum((self.predictions==labels))/len(self.predictions)).item()
        return round(accuracy, 4)*100
    def top_k_preds(self, k):
        return torch.topk(self.logits, dim=-1, k=5).indices
    def top_k_acc(self, labels, k):
        labels=torch.unsqueeze(labels,dim=-1)
        labels=torch.cat(tuple(labels for _ in range(k)),-1)
        labels=torch.unsqueeze(labels,dim=1)
        preds=self.top_k_preds(k)
        acc=(torch.sum(preds==labels)/len(labels)).item()
        return round(acc, 4)*100
        
class CNN1dlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p=0.1, bias=False, padding=0):
        super().__init__()
        self.in_conv_dim = in_channels
        self.out_conv_dim = out_channels

        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size, stride, bias=bias, padding=padding)
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        self.act = nn.GELU()

    def forward(self, inputs):
        hidden = self.conv(inputs)
        hidden = self.dropout(hidden)

        hidden = hidden.transpose(-2, -1)
        hidden = self.layer_norm(hidden)
        hidden = hidden.transpose(-2, -1)

        hidden = self.act(hidden)

        return hidden

class CNN1dlayerNoAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p=0.1, bias=False, padding=0):
        super().__init__()
        self.in_conv_dim = in_channels
        self.out_conv_dim = out_channels

        self.conv = nn.Conv1d(self.in_conv_dim, self.out_conv_dim, kernel_size, stride, bias=bias, padding=padding)
        self.dropout = nn.Dropout(p=dropout_p)
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)

    def forward(self, inputs):
        hidden = self.conv(inputs) 
        hidden = self.dropout(hidden)

        hidden = hidden.transpose(-2, -1)
        hidden = self.layer_norm(hidden)
        hidden = hidden.transpose(-2, -1)
        return hidden

class ResnetConfig:
    def __init__(
        self,
        input_dim = 1,
        output_dim = 14,
        res_dims=[32, 64, 64, 64],
        res_kernel=[5, 7, 17, 13],
        res_stride=[4, 4, 5, 3],
        num_blocks=[2, 2, 2, 2],
        first_kernel_size = 13,
        first_stride = 1,
        first_pool_kernel_size = 7,
        first_pool_stride = 7,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.res_dims = res_dims
        self.res_kernel = res_kernel
        self.res_stride = res_stride
        self.num_blocks = num_blocks
        self.first_kernel_size = first_kernel_size
        self.first_stride = first_stride
        self.first_pool_kernel_size = first_pool_kernel_size
        self.first_pool_stride = first_pool_stride

class Resnet1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dropout_p=0.05, downsample=False, **kargs):
        super().__init__()
        
        # padding ensure the hidden and residual is in the same size
        self.conv1 = CNN1dlayer(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dropout_p=dropout_p,
            padding=(kernel_size)//2
        )

        self.conv2 = CNN1dlayerNoAct(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1, 
            dropout_p=dropout_p,
            padding=(kernel_size)//2
        )

        if downsample is not None:
            self.downsample = CNN1dlayerNoAct(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                dropout_p=0
            )
        else:
            self.downsample = None

        self.act = nn.GELU()

    def forward(self, inputs):
        if self.downsample:
            residual = self.downsample(inputs)
        else:
            residual = inputs 

        hidden = self.conv1(inputs)
        hidden = self.conv2(hidden)

        hidden = residual + hidden
        hidden = self.act(hidden)

        return hidden

class Resnet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.input_dim = config.input_dim

        self.conv = CNN1dlayer(
            self.input_dim,
            config.res_dims[0],
            kernel_size=config.first_kernel_size,
            stride=config.first_stride,
            dropout_p=0.05,
        )

        self.maxpool = nn.MaxPool1d(
            config.first_pool_kernel_size,
            stride=config.first_pool_stride,
            padding=config.first_pool_kernel_size//2,
        )

        self.layer1 = self._make_resnet_layer(
            config.res_dims[0],
            config.res_dims[0],
            config.num_blocks[0],
            config.res_kernel[0],
            config.res_stride[0],
        )

        self.layer2 = self._make_resnet_layer(
            config.res_dims[1],
            config.res_dims[0],
            config.num_blocks[1],
            config.res_kernel[1],
            config.res_stride[1],
        )

        self.layer3 = self._make_resnet_layer(
            config.res_dims[2],
            config.res_dims[1],
            config.num_blocks[2],
            config.res_kernel[2],
            config.res_stride[2],
        )
        self.layer4 = self._make_resnet_layer(
            config.res_dims[3],
            config.res_dims[2],
            config.num_blocks[3],
            config.res_kernel[3],
            config.res_stride[3],    
        )

    def _make_resnet_layer(self, dim, prev_dim, num_blocks, kernel_size=3, stride=1, dropout_p=0.05):
        layers = []

        layers.append(
            Resnet1dBlock(
                in_channels = prev_dim,
                out_channels = dim,
                kernel_size = kernel_size,
                stride = stride,
                dropout_p = dropout_p,
                downsample=True
            )
        )

        for i in range(1, num_blocks):
            layers.append(
                Resnet1dBlock(
                    in_channels = dim,
                    out_channels = dim,
                    kernel_size = kernel_size,
                    stride = 1,
                    dropout_p = dropout_p,
                    downsample=False
                )
            )

        return nn.Sequential(*layers)

    def forward(self, inputs):
        hidden = self.conv(inputs)

        hidden = self.layer1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.layer3(hidden)
        hidden = self.layer4(hidden)

        return hidden

class ResnetClassifier(nn.Module):
    def __init__(self, config=None, num_classes=None):
        super().__init__()
        if config==None:
            if num_classes is not None:
                config=ResnetConfig(output_dim=num_classes)
            else:
                config=ResnetConfig()
        self.resnet = Resnet(config)
        self.avgpool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(config.res_dims[-1], config.output_dim)
        self.num_labels = config.output_dim
        self.adv = nn.Linear(config.res_dims[-1], 1)
    def forward(self, inputs, labels=None, s=True, loss_func=None):
        hidden=self.resnet(inputs)
        pooled_output=self.avgpool(hidden)
        pooled_output=pooled_output.transpose(-2, -1)
        s_logits = self.classifier(pooled_output)
        u_logits = self.adv(pooled_output)

        if labels is not None:
            if s:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(s_logits.view(-1, self.num_labels), labels.view(-1))
                return ModelOutput(logits=s_logits, loss=loss)
            loss=loss_func(u_logits, labels)
            return ModelOutput(logits=u_logits, loss=loss)
        return ModelOutput(logits=s_logits, loss=None)
    
