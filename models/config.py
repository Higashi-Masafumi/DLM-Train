from typing import Any, Literal, Type

import torch
from pydantic import (
    BaseModel,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)
from typing_extensions import Self

import models.model
from models.rmsnorm import FusedRMSNorm, RMSNorm


class Config(BaseModel):
    """Configuration class for the model.

    Args:
        BaseModel (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    name: StrictStr = Field(description="The name of the config")
    block_size: StrictInt = Field(
        description="The block size of the config", default=4096
    )
    vocab_size: StrictInt = Field(
        description="The vocabulary size of the config", default=50254
    )
    padding_multiple: StrictInt = Field(
        description="The padding multiple of the config", default=512
    )
    padded_vocab_size: StrictInt | None = Field(
        description="The padded vocabulary size of the config", default=None
    )
    n_layer: StrictInt = Field(
        description="The number of layer of the config", default=16
    )
    n_head: StrictInt = Field(
        description="The number of head of the config", default=32
    )
    n_embed: StrictInt = Field(
        description="The number of embed of the config", default=4096
    )
    rotary_percentage: StrictFloat = Field(
        description="The rotary percentage of the config", default=0.25
    )
    parallel_residual: StrictBool = Field(
        description="The parallel residual of the config", default=True
    )
    bias: StrictBool = Field(description="The bias of the config", default=True)
    # to use multi-head attention (MHA), set this to `n_head` (default)
    # to use multi-query attention (MQA), set this to 1
    # to use grouped-query attention (GQA), set this to a value in between
    # Example with `n_head=4`
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ v ││ v ││ v ││ v │     │ v │    │ v │             │ v │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │         │        │                 │
    # ┌───┐┌───┐┌───┐┌───┐     ┌───┐    ┌───┐             ┌───┐
    # │ k ││ k ││ k ││ k │     │ k │    │ k │             │ k │
    # └───┘└───┘└───┘└───┘     └───┘    └───┘             └───┘
    #   │    │    │    │      ┌──┴──┐  ┌──┴──┐      ┌────┬──┴─┬────┐
    # ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐  ┌───┐┌───┐┌───┐┌───┐
    # │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │  │ q ││ q ││ q ││ q │
    # └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘  └───┘└───┘└───┘└───┘
    # ◀──────────────────▶  ◀──────────────────▶  ◀──────────────────▶
    #         MHA                    GQA                   MQA
    #   n_query_groups=4       n_query_groups=2      n_query_groups=1
    #
    # credit https://arxiv.org/pdf/2305.13245.pdf
    n_query_groups: StrictInt | None = Field(
        description="The number of query groups of the config", default=None
    )
    shared_attention_norm: StrictBool = Field(
        description="The shared attention norm of the config", default=False
    )
    _norm_class: Literal["LayerNorm", "RMSNorm"] = Field(
        description="The norm class of the config", default="LayerNorm"
    )
    norm_eps: StrictFloat = Field(
        description="The norm epsilon of the config", default=1e-5
    )
    _mlp_class: Literal["GptNeoXMLP", "LLaMAMLP"] = Field(
        description="The MLP class of the config", default="GptNeoXMLP"
    )
    intermediate_size: StrictInt | None = Field(
        description="The intermediate size of the config", default=None
    )
    condense_ratio: StrictInt = Field(
        description="The condense ratio of the config", default=1
    )

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        """Validate the configuration.

        Raises:
            ValueError: If the configuration is invalid.

        Returns:
            Self: The validated configuration.
        """
        assert self.n_embed % self.n_head == 0
        if self.padded_vocab_size is None:
            # padded_vocab_sizeをpadding_multipleの倍数にする
            self.padded_vocab_size = (
                self.vocab_size
                if self.vocab_size % self.padding_multiple == 0
                else self.vocab_size
                + self.padding_multiple
                - self.vocab_size % self.padding_multiple
            )
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        if self.intermediate_size is None:
            if self._mlp_class == "LLaMAMLP":
                raise ValueError("intermediate_size is required for LLaMAMLP")
            self.intermediate_size = self.n_embed * 4
        return self

    @property
    def head_size(self) -> StrictInt:
        """The size of each attention head.

        Returns:
            StrictInt: The size of each attention head.
        """
        return self.n_embed // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs: Any) -> Self:
        conf_dict = name_to_config[name].copy()
        conf_dict.update(kwargs)
        return cls(**conf_dict)

    @property
    def mlp_class(self) -> Type:
        # `self._mlp_class` cannot be the type to keep the config json serializable
        return getattr(models.model, self._mlp_class)

    @property
    def norm_class(self) -> Type:
        # `self._norm_class` cannot be the type to keep the config json serializable
        if self._norm_class == "RMSNorm":

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)


configs = []

Diff_LLaMA = [
    dict(
        name="Diff_LLaMA_6M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=6,
        n_head=4,
        n_embed=256,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1024,
        n_query_groups=4,
    ),
    dict(
        name="Diff_LLaMA_19M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=8,
        n_head=6,
        n_embed=384,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1536,
        n_query_groups=6,
    ),
    dict(
        name="Diff_LLaMA_34M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=8,
        n_head=8,
        n_embed=512,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2048,
        n_query_groups=8,
    ),
    dict(
        name="Diff_LLaMA_48M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=9,
        n_head=9,
        n_embed=576,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2304,
        n_query_groups=9,
    ),
    dict(
        name="Diff_LLaMA_66M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=10,
        n_head=10,
        n_embed=640,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2560,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_85M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=13,
        n_head=10,
        n_embed=640,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=2560,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_75M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=16,
        n_head=8,
        n_embed=640,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=1600,
        n_query_groups=8,
    ),
    dict(
        name="Diff_LLaMA_113M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=12,
        n_head=12,
        n_embed=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3072,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_142M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=15,
        n_head=12,
        n_embed=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3072,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_170M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=18,
        n_head=12,
        n_embed=768,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3072,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_180M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=14,
        n_head=14,
        n_embed=896,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3584,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_206M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=16,
        n_head=14,
        n_embed=896,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3584,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_231M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=18,
        n_head=14,
        n_embed=896,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=3584,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_268M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=16,
        n_head=16,
        n_embed=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_302M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=18,
        n_head=16,
        n_embed=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_336M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=20,
        n_head=16,
        n_embed=1024,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=4096,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_472M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=18,
        n_head=10,
        n_embed=1280,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5120,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_551M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=21,
        n_head=10,
        n_embed=1280,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5120,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_571M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=18,
        n_head=11,
        n_embed=1408,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=11,
    ),
    dict(
        name="Diff_LLaMA_629M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=24,
        n_head=10,
        n_embed=1280,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,
        _mlp_class="LLaMAMLP",
        intermediate_size=5120,
        n_query_groups=10,
    ),
    dict(
        name="Diff_LLaMA_666M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=21,
        n_head=11,
        n_embed=1408,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=11,
    ),
    dict(
        name="Diff_LLaMA_717M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=19,
        n_head=12,
        n_embed=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_761M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=24,
        n_head=11,
        n_embed=1408,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=5632,
        n_query_groups=11,
    ),
    dict(
        name="Diff_LLaMA_831M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=12,
        n_embed=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_944M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=25,
        n_head=12,
        n_embed=1536,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=6144,
        n_query_groups=12,
    ),
    dict(
        name="Diff_LLaMA_1028M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=20,
        n_head=14,
        n_embed=1792,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=7168,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_1233M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=24,
        n_head=14,
        n_embed=1792,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=7168,
        n_query_groups=14,
    ),
    dict(
        name="Diff_LLaMA_1476M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=22,
        n_head=16,
        n_embed=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=8192,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_1678M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=25,
        n_head=16,
        n_embed=2048,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=8192,
        n_query_groups=16,
    ),
    dict(
        name="Diff_LLaMA_2121M",
        block_size=2048,
        vocab_size=32000,
        padding_multiple=64,
        n_layer=28,
        n_head=17,
        n_embed=2176,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        _norm_class="FusedRMSNorm",
        norm_eps=1e-5,  # Llama 2 use 1e-5. Llama 1 use 1e-6
        _mlp_class="LLaMAMLP",
        intermediate_size=8704,
        n_query_groups=17,
    ),
]
configs.extend(Diff_LLaMA)

name_to_config = {config["name"]: config for config in configs}
