#Credit to Haotong Liang (AuroraLHT)

import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

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

class ResnetConfig(PretrainedConfig):
    def __init__(
        self,
        input_dim,
        output_dim,
        res_dims=[64,128,256,512],
        res_kernel=[3,3,3,3],
        res_stride=[2,2,2,2],
        num_blocks=[3,4,6,3],
        first_kernel_size = 7,
        first_stride = 2,
        first_pool_kernel_size = 3,
        first_pool_stride = 1,
        **kargs
    ):
        super().__init__(**kargs)

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
    """
        This block is similar to the BasicBlock in the fastai 2dresnet implementation
    """
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

class ResnetPreTrained(PreTrainedModel):
    config_class = ResnetConfig
    base_model_prefix = "resnet"

class ResnetClassifier(ResnetPreTrained):
    def __init__(self, config):
        super().__init__(config)
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
                return SequenceClassifierOutput(logits=s_logits, loss=loss)
            loss=loss_func(u_logits, labels)
            return SequenceClassifierOutput(logits=u_logits, loss=loss)
        return SequenceClassifierOutput(logits=s_logits, loss=None)
