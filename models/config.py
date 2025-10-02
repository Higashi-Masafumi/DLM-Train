from typing import Literal

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


class Config(BaseModel):
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
        return self.n_embed // self.n_head
