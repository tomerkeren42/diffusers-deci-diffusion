from typing import Any

from diffusers import UNet2DConditionModel
from diffusers.configuration_utils import register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import nn


import itertools
from typing import Any, Optional, Dict, Tuple

import torch
from diffusers import Transformer2DModel, ModelMixin, ConfigMixin
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.resnet import ResnetBlock2D, Downsample2D, Upsample2D
from diffusers.models.transformer_2d import Transformer2DModelOutput
from torch import nn


def custom_sort_order(obj):
    """
    Key function for sorting order of execution in forward methods
    """
    return {ResnetBlock2D: 0, Transformer2DModel: 1, FlexibleTransformer2DModel: 1}.get(obj.__class__)


class FlexibleCrossAttnDownBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_resnets: int = 1,
            num_attentions: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads: int = 1,
            cross_attention_dim: int = 1280,
            output_scale_factor: float = 1.0,
            downsample_padding: int = 1,
            add_downsample: bool = True,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            upcast_attention: bool = False,
            last_block: bool = False,
            mix_block_in_forward: bool = True,
    ):
        super().__init__()

        self.last_block = last_block
        self.mix_block_in_forward = mix_block_in_forward
        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        modules = []
        add_resnets = [True] * num_resnets
        add_cross_attentions = [True] * num_attentions
        for i, (add_resnet, add_cross_attention) in enumerate(
                itertools.zip_longest(add_resnets, add_cross_attentions, fillvalue=False)):
            in_channels = in_channels if i == 0 else out_channels
            if add_resnet:
                modules.append(
                    ResnetBlock2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                )
            if add_cross_attention:
                modules.append(
                    FlexibleTransformer2DModel(
                        num_attention_heads=num_attention_heads,
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                )

        if not mix_block_in_forward:
            modules = sorted(modules, key=custom_sort_order)

        self.modules_list = nn.ModuleList(modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample2D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):
        output_states = ()

        for module in self.modules_list:
            if isinstance(module, ResnetBlock2D):
                hidden_states = module(hidden_states, temb)
            elif isinstance(module, (FlexibleTransformer2DModel, Transformer2DModel)):
                hidden_states = module(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            else:
                raise ValueError(f'Got an unexpected module in modules list! {type(module)}')
            if isinstance(module, ResnetBlock2D):
                output_states = output_states + (hidden_states,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

            if not self.last_block:
                output_states = output_states + (hidden_states,)

        return hidden_states, output_states


class FlexibleCrossAttnUpBlock2D(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            prev_output_channel: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_resnets: int = 1,
            num_attentions: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            cross_attention_dim=1280,
            output_scale_factor=1.0,
            add_upsample=True,
            use_linear_projection=False,
            only_cross_attention=False,
            upcast_attention=False,
            mix_block_in_forward: bool = True
    ):
        super().__init__()
        modules = []

        # WARNING: This parameter is filled with number of resnets and used within diffusers
        self.resnets = []

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads

        add_resnets = [True] * num_resnets
        add_cross_attentions = [True] * num_attentions
        for i, (add_resnet, add_cross_attention) in enumerate(
                itertools.zip_longest(add_resnets, add_cross_attentions, fillvalue=False)):
            res_skip_channels = in_channels if (i == len(add_resnets) - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            if add_resnet:
                self.resnets += [True]
                modules.append(
                    ResnetBlock2D(
                        in_channels=resnet_in_channels + res_skip_channels,
                        out_channels=out_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                )
            if add_cross_attention:
                modules.append(
                    FlexibleTransformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                    )
                )

        if not mix_block_in_forward:
            modules = sorted(modules, key=custom_sort_order)

        self.modules_list = nn.ModuleList(modules)

        self.upsamplers = None
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, use_conv=True, out_channels=out_channels)])

        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            res_hidden_states_tuple: Tuple[torch.FloatTensor, ...],
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            upsample_size: Optional[int] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ):

        for module in self.modules_list:
            if isinstance(module, ResnetBlock2D):
                res_hidden_states = res_hidden_states_tuple[-1]
                res_hidden_states_tuple = res_hidden_states_tuple[:-1]
                hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
                hidden_states = module(hidden_states, temb)
            if isinstance(module, (FlexibleTransformer2DModel, Transformer2DModel)):
                hidden_states = module(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, upsample_size)

        return hidden_states


class FlexibleUNetMidBlock2DCrossAttn(nn.Module):
    def __init__(
            self,
            in_channels: int,
            temb_channels: int,
            dropout: float = 0.0,
            num_resnets: int = 1,
            num_attentions: int = 1,
            transformer_layers_per_block: int = 1,
            resnet_eps: float = 1e-6,
            resnet_time_scale_shift: str = "default",
            resnet_act_fn: str = "swish",
            resnet_groups: int = 32,
            resnet_pre_norm: bool = True,
            num_attention_heads=1,
            output_scale_factor=1.0,
            cross_attention_dim=1280,
            use_linear_projection=False,
            upcast_attention=False,
            mix_block_in_forward: bool = True,
            add_upsample: bool = True,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.num_attention_heads = num_attention_heads
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        # There is always at least one resnet
        modules = [ResnetBlock2D(
            in_channels=in_channels,
            out_channels=in_channels,
            temb_channels=temb_channels,
            eps=resnet_eps,
            groups=resnet_groups,
            dropout=dropout,
            time_embedding_norm=resnet_time_scale_shift,
            non_linearity=resnet_act_fn,
            output_scale_factor=output_scale_factor,
            pre_norm=resnet_pre_norm,
        )]

        add_resnets = [True] * num_resnets
        add_cross_attentions = [True] * num_attentions
        for i, (add_resnet, add_cross_attention) in enumerate(
                itertools.zip_longest(add_resnets, add_cross_attentions, fillvalue=False)):
            if add_cross_attention:
                modules.append(
                    FlexibleTransformer2DModel(
                        num_attention_heads,
                        in_channels // num_attention_heads,
                        in_channels=in_channels,
                        num_layers=transformer_layers_per_block,
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        upcast_attention=upcast_attention,
                    )
                )

            if add_resnet:
                modules.append(
                    ResnetBlock2D(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        temb_channels=temb_channels,
                        eps=resnet_eps,
                        groups=resnet_groups,
                        dropout=dropout,
                        time_embedding_norm=resnet_time_scale_shift,
                        non_linearity=resnet_act_fn,
                        output_scale_factor=output_scale_factor,
                        pre_norm=resnet_pre_norm,
                    )
                )
        if not mix_block_in_forward:
            modules = sorted(modules, key=custom_sort_order)

        self.modules_list = nn.ModuleList(modules)

        self.upsamplers = nn.ModuleList([nn.Identity()])
        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample2D(in_channels, use_conv=True, out_channels=in_channels)])

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        hidden_states = self.modules_list[0](hidden_states, temb)

        for module in self.modules_list:
            if isinstance(module, (FlexibleTransformer2DModel, Transformer2DModel)):
                hidden_states = module(
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                    return_dict=False,
                )[0]
            elif isinstance(module, ResnetBlock2D):
                hidden_states = module(hidden_states, temb)

        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)

        return hidden_states


class FlexibleTransformer2DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            only_cross_attention: bool = False,
            use_linear_projection: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.in_channels = in_channels
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        self.use_linear_projection = use_linear_projection
        if self.use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                )
                for _ in range(num_layers)
            ]
        )

        # Define output layers
        self.out_channels = in_channels if out_channels is None else out_channels
        if self.use_linear_projection:
            self.proj_out = nn.Linear(inner_dim, in_channels)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = False
    ):
        # 1. Input
        batch, _, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=class_labels,
            )

        # 3. Output
        if not self.use_linear_projection:
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

        output = hidden_states + residual
        if return_dict:
            return (output, )
        return Transformer2DModelOutput(sample=output)


class FlexibleUNet2DConditionModel(UNet2DConditionModel, ModelMixin):
    @register_to_config
    def __init__(self, configurations):
        super().__init__(sample_size=configurations.get('sample_size', 512),
                         cross_attention_dim=configurations.get("cross_attention_dim", 1280))

        mid_block_add_upsample = configurations.get("add_upsample_mid_block")
        mix_block_in_forward = configurations.get("mix_block_in_forward")
        down_blocks_in_channels = configurations.get("down_blocks_in_channels")
        down_blocks_out_channels = configurations.get("down_blocks_out_channels")
        down_blocks_num_resnets = configurations.get("down_blocks_num_resnets")
        down_blocks_num_attentions = configurations.get("down_blocks_num_attentions")
        add_downsample = configurations.get("add_downsample")
        prev_output_channels = configurations.get("prev_output_channels")
        up_blocks_num_resnets = configurations.get("up_blocks_num_resnets")
        up_blocks_num_attentions = configurations.get("up_blocks_num_attentions")
        up_upsample = configurations.get("add_upsample")
        temb_dim = configurations.get("temb_dim")
        resnet_eps = configurations.get("resnet_eps")
        resnet_act_fn = configurations.get("resnet_act_fn")
        num_attention_heads = configurations.get("num_attention_heads")
        cross_attention_dim = configurations.get("cross_attention_dim")
        mid_num_resnets = configurations.get("mid_num_resnets")
        mid_num_attentions = configurations.get("mid_num_attentions")

        ###############
        # Down blocks #
        ###############
        self.down_blocks = nn.ModuleList()

        for i, (in_c, out_c, n_res, n_att, add_down) in enumerate(zip(down_blocks_in_channels, down_blocks_out_channels,
                                                                      down_blocks_num_resnets,
                                                                      down_blocks_num_attentions,
                                                                      add_downsample)):
            last_block = i == len(down_blocks_in_channels) - 1
            self.down_blocks.append(FlexibleCrossAttnDownBlock2D(in_channels=in_c,
                                                                 out_channels=out_c,
                                                                 temb_channels=temb_dim,
                                                                 num_resnets=n_res,
                                                                 num_attentions=n_att,
                                                                 resnet_eps=resnet_eps,
                                                                 resnet_act_fn=resnet_act_fn,
                                                                 num_attention_heads=num_attention_heads,
                                                                 cross_attention_dim=cross_attention_dim,
                                                                 add_downsample=add_down,
                                                                 last_block=last_block,
                                                                 mix_block_in_forward=mix_block_in_forward))

        ###############
        # Mid blocks  #
        ###############

        self.mid_block = FlexibleUNetMidBlock2DCrossAttn(in_channels=down_blocks_out_channels[-1],
                                                         temb_channels=temb_dim,
                                                         resnet_act_fn=resnet_act_fn,
                                                         resnet_eps=resnet_eps,
                                                         cross_attention_dim=cross_attention_dim,
                                                         num_attention_heads=num_attention_heads,
                                                         num_resnets=mid_num_resnets,
                                                         num_attentions=mid_num_attentions,
                                                         mix_block_in_forward=mix_block_in_forward,
                                                         add_upsample=mid_block_add_upsample
                                                         )

        ###############
        #  Up blocks  #
        ###############

        self.up_blocks = nn.ModuleList()
        for in_c, out_c, prev_out, n_res, n_att, add_up in zip(reversed(down_blocks_in_channels),
                                                               reversed(down_blocks_out_channels),
                                                               prev_output_channels,
                                                               up_blocks_num_resnets, up_blocks_num_attentions,
                                                               up_upsample):
            self.up_blocks.append(FlexibleCrossAttnUpBlock2D(in_channels=in_c,
                                                             out_channels=out_c,
                                                             prev_output_channel=prev_out,
                                                             temb_channels=temb_dim,
                                                             num_resnets=n_res,
                                                             num_attentions=n_att,
                                                             resnet_eps=resnet_eps,
                                                             resnet_act_fn=resnet_act_fn,
                                                             num_attention_heads=num_attention_heads,
                                                             cross_attention_dim=cross_attention_dim,
                                                             add_upsample=add_up,
                                                             mix_block_in_forward=mix_block_in_forward
                                                             ))
