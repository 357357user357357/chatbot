"""TileScale chatbot — PyTorch chatbot tutorial rewritten with TileLang kernels.

Original tutorial: https://docs.pytorch.org/tutorials/beginner/chatbot_tutorial.html
Kernel language:   https://github.com/tile-ai/tilescale  (``import tilelang``)
"""

__version__ = "1.0.0"

# Must run before the submodule imports below pull in torch and the first
# CUDA call happens: hides sub-sm_75 GPUs (e.g. a GTX 950 kept for displays)
# because cuDNN >= 9.11 aborts in any process that can see one.
from .cuda_guard import filter_cuda_devices as _filter_cuda_devices  # noqa: E402

_filter_cuda_devices()

from .data import (  # noqa: E402,F401
    EOS_token,
    MAX_LENGTH,
    PAD_token,
    SOS_token,
    Voc,
    batch2TrainData,
    loadPrepareData,
    trimRareWords,
)
from .kernels import (  # noqa: F401
    TiledAttention,
    TiledGRUStep,
    reference_attention,
    reference_gru_step,
    tilelang_available,
)
from .models import (  # noqa: F401
    EncoderRNN,
    GreedySearchDecoder,
    LuongAttnDecoderRNN,
)
