import torch
import torch.nn as nn
from lib.models.backbones.vit.transformer import TransformerModel
from lib.models.backbones.vit.position_encoding import (
    FixedPositionalEncoding,
    LearnedPositionalEncoding,
)
from lib.models.backbones.vit.helper import IntermediateSequential

__all__ = [
    'SETR_Naive_S',
    'SETR_Naive_L',
    'SETR_Naive_H',
    'SETR_PUP_S',
    'SETR_PUP_L',
    'SETR_PUP_H',
    'SETR_MLA_S',
    'SETR_MLA_L',
    'SETR_MLA_H',
]

class SegmentationTransformer(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(SegmentationTransformer, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0

        self.img_dim = img_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.transformer = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        if self.conv_patch_representation:
            self.conv_x = nn.Conv2d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=(self.patch_dim, self.patch_dim),
                stride=(self.patch_dim, self.patch_dim),
                padding=self._get_padding(
                    'VALID', (self.patch_dim, self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

    def _init_decode(self):
        raise NotImplementedError("Should be implemented in child class!!")

    def encode(self, x):
        n, c, h, w = x.shape
        if self.conv_patch_representation:
            # combine embedding w/ conv patch distribution
            x = self.conv_x(x)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.embedding_dim)
        else:
            x = (
                x.unfold(2, self.patch_dim, self.patch_dim)
                .unfold(3, self.patch_dim, self.patch_dim)
                .contiguous()
            )
            x = x.view(n, c, -1, self.patch_dim ** 2)
            x = x.permute(0, 2, 3, 1).contiguous()
            x = x.view(x.size(0), -1, self.flatten_dim)
            x = self.linear_encoding(x)

        x = self.position_encoding(x)
        x = self.pe_dropout(x)

        # apply transformer
        x, intmd_x = self.transformer(x)
        x = self.pre_head_ln(x)

        return x, intmd_x

    def decode(self, x):
        raise NotImplementedError("Should be implemented in child class!!")

    def forward(self, x, auxillary_output_layers=None):
        encoder_output, intmd_encoder_outputs = self.encode(x)
        decoder_output = self.decode(
            encoder_output, intmd_encoder_outputs, auxillary_output_layers
        )

        if auxillary_output_layers is not None:
            auxillary_outputs = {}
            for i in auxillary_output_layers:
                val = str(2 * i - 1)
                _key = 'Z' + str(i)
                auxillary_outputs[_key] = intmd_encoder_outputs[val]

            return decoder_output, auxillary_outputs

        return decoder_output

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)

    def _reshape_output(self, x):
        x = x.view(
            x.size(0),
            int(self.img_dim / self.patch_dim),
            int(self.img_dim / self.patch_dim),
            self.embedding_dim,
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class SETR_Naive(SegmentationTransformer):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(SETR_Naive, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes
        self._init_decode()

    def _init_decode(self):
        self.conv1 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.embedding_dim,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )
        self.bn1 = nn.BatchNorm2d(self.embedding_dim)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=self.embedding_dim,
            out_channels=self.num_classes,
            kernel_size=1,
            stride=1,
            padding=self._get_padding('VALID', (1, 1),),
        )
        self.upsample = nn.Upsample(
            scale_factor=self.patch_dim, mode='bilinear'
        )

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class SETR_PUP(SegmentationTransformer):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(SETR_PUP, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes
        self._init_decode()

    def _init_decode(self):
        extra_in_channels = int(self.embedding_dim / 4)
        in_channels = [
            self.embedding_dim,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
        ]
        out_channels = [
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            extra_in_channels,
            self.num_classes,
        ]

        modules = []
        for i, (in_channel, out_channel) in enumerate(
            zip(in_channels, out_channels)
        ):
            modules.append(
                nn.Conv2d(
                    in_channels=in_channel,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=1,
                    padding=self._get_padding('VALID', (1, 1),),
                )
            )
            if i != 4:
                modules.append(nn.Upsample(scale_factor=2, mode='bilinear'))
        self.decode_net = IntermediateSequential(
            *modules, return_intermediate=False
        )

    def decode(self, x, intmd_x, intmd_layers=None):
        x = self._reshape_output(x)
        x = self.decode_net(x)
        return x


class SETR_MLA(SegmentationTransformer):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        dropout_rate=0.0,
        attn_dropout_rate=0.0,
        conv_patch_representation=False,
        positional_encoding_type="learned",
    ):
        super(SETR_MLA, self).__init__(
            img_dim=img_dim,
            patch_dim=patch_dim,
            num_channels=num_channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
            attn_dropout_rate=attn_dropout_rate,
            conv_patch_representation=conv_patch_representation,
            positional_encoding_type=positional_encoding_type,
        )

        self.num_classes = num_classes
        self._init_decode()

    def _init_decode(self):
        self.net1_in, self.net1_intmd, self.net1_out = self._define_agg_net()
        self.net2_in, self.net2_intmd, self.net2_out = self._define_agg_net()
        self.net3_in, self.net3_intmd, self.net3_out = self._define_agg_net()
        self.net4_in, self.net4_intmd, self.net4_out = self._define_agg_net()

        # fmt: off
        self.output_net = IntermediateSequential(return_intermediate=False)
        self.output_net.add_module(
            "conv_1",
            nn.Conv2d(
                in_channels=self.embedding_dim, out_channels=self.num_classes,
                kernel_size=1, stride=1,
                padding=self._get_padding('VALID', (1, 1),),
            )
        )
        self.output_net.add_module(
            "upsample_1",
            nn.Upsample(scale_factor=4, mode='bilinear')
        )
        # fmt: on

    def decode(self, x, intmd_x, intmd_layers=None):
        assert intmd_layers is not None, "pass the intermediate layers for MLA"

        encoder_outputs = {}
        all_keys = []
        for i in intmd_layers:
            val = str(2 * i - 1)
            _key = 'Z' + str(i)
            all_keys.append(_key)
            encoder_outputs[_key] = intmd_x[val]
        all_keys.reverse()

        temp_x = encoder_outputs[all_keys[0]]
        temp_x = self._reshape_output(temp_x)
        key0_intmd_in = self.net1_in(temp_x)
        key0_out = self.net1_out(key0_intmd_in)

        temp_x = encoder_outputs[all_keys[1]]
        temp_x = self._reshape_output(temp_x)
        key1_in = self.net2_in(temp_x)
        key1_intmd_in = key1_in + key0_intmd_in
        key1_intmd_out = self.net2_intmd(key1_intmd_in)
        key1_out = self.net2_out(key1_intmd_out)

        temp_x = encoder_outputs[all_keys[2]]
        temp_x = self._reshape_output(temp_x)
        key2_in = self.net3_in(temp_x)
        key2_intmd_in = key2_in + key1_intmd_in
        key2_intmd_out = self.net3_intmd(key2_intmd_in)
        key2_out = self.net3_out(key2_intmd_out)

        temp_x = encoder_outputs[all_keys[3]]
        temp_x = self._reshape_output(temp_x)
        key3_in = self.net4_in(temp_x)
        key3_intmd_in = key3_in + key2_intmd_in
        key3_intmd_out = self.net4_intmd(key3_intmd_in)
        key3_out = self.net4_out(key3_intmd_out)

        out = torch.cat((key0_out, key1_out, key2_out, key3_out), dim=1)
        out = self.output_net(out)
        return out

    # fmt: off
    def _define_agg_net(self):
        model_in = IntermediateSequential(return_intermediate=False)
        model_in.add_module(
            "layer_1",
            nn.Conv2d(
                self.embedding_dim, int(self.embedding_dim / 2), 1, 1,
                padding=self._get_padding('VALID', (1, 1),),
            ),
        )

        model_intmd = IntermediateSequential(return_intermediate=False)
        model_intmd.add_module(
            "layer_intmd",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )

        model_out = IntermediateSequential(return_intermediate=False)
        model_out.add_module(
            "layer_2",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 2), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "layer_3",
            nn.Conv2d(
                int(self.embedding_dim / 2), int(self.embedding_dim / 4), 3, 1,
                padding=self._get_padding('SAME', (3, 3),),
            ),
        )
        model_out.add_module(
            "upsample", nn.Upsample(scale_factor=4, mode='bilinear')
        )
        return model_in, model_intmd, model_out
    # fmt: on


def SETR_Naive_S(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = None
    model = SETR_Naive(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_Naive_L(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [10, 15, 20]
    model = SETR_Naive(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_Naive_H(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = None
    model = SETR_Naive(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1280,
        num_heads=16,
        num_layers=32,
        hidden_dim=5120,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_PUP_S(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = None
    model = SETR_PUP(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_PUP_L(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [10, 15, 20, 24]
    model = SETR_PUP(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_PUP_H(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [10, 15, 20, 24]
    model = SETR_PUP(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1280,
        num_heads=16,
        num_layers=32,
        hidden_dim=5120,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_MLA_S(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = None
    model = SETR_MLA(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=768,
        num_heads=12,
        num_layers=12,
        hidden_dim=3072,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_MLA_L(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [6, 12, 18, 24]
    model = SETR_MLA(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1024,
        num_heads=16,
        num_layers=24,
        hidden_dim=4096,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model


def SETR_MLA_H(dataset='cityscapes', _conv_repr=False, _pe_type="learned"):
    if dataset.lower() == 'cityscapes':
        img_dim = 768
        num_classes = 19
    elif dataset.lower() == 'ade20k':
        img_dim = 512
        num_classes = 150
    elif dataset.lower() == 'pascal':
        img_dim = 480
        num_classes = 59

    num_channels = 3
    patch_dim = 16
    aux_layers = [8, 16, 24, 32]
    model = SETR_MLA(
        img_dim,
        patch_dim,
        num_channels,
        num_classes,
        embedding_dim=1280,
        num_heads=16,
        num_layers=32,
        hidden_dim=5120,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
        conv_patch_representation=_conv_repr,
        positional_encoding_type=_pe_type,
    )

    return aux_layers, model