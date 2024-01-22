import torch
from transformers import PretrainedConfig, BitsAndBytesConfig
import math
from typing import Optional

class VisionProjectorConfig(PretrainedConfig):
    def __init__(
            self,
            input_dim=768,
            hidden_dim=256,
            num_tokens=1,
            output_dim=2560,
            **kwargs
    ):
        #super.__init__(**kwargs)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        self.kwargs = kwargs


class CustomPhiConfig(PretrainedConfig):
    model_type = "phi-msft"
    attribute_map = {
        "max_position_embeddings": "n_positions",
        "hidden_size": "n_embd",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }

    def __init__(
            self,
            vocab_size: int = 51200,
            n_positions: int = 2048,
            n_embd: int = 2560,
            n_layer: int = 32,
            n_inner: Optional[int] = None,
            n_head: int = 32,
            n_head_kv: Optional[int] = None,
            rotary_dim: Optional[int] = 32,
            activation_function: Optional[str] = "gelu_new",
            flash_attn: bool = False,
            flash_rotary: bool = False,
            fused_dense: bool = False,
            attn_pdrop: float = 0.0,
            embd_pdrop: float = 0.0,
            resid_pdrop: float = 0.1,
            layer_norm_epsilon: float = 1e-05,
            initializer_range: float = 0.02,
            tie_word_embeddings: bool = False,
            pad_vocab_size_multiple: int = 64,
            **kwargs
    ) -> None:
        self.vocab_size = int(math.ceil(vocab_size / pad_vocab_size_multiple) * pad_vocab_size_multiple)
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_inner = n_inner
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.rotary_dim = min(rotary_dim, n_embd // n_head)
        self.activation_function = activation_function
        self.flash_attn = flash_attn
        self.flash_rotary = flash_rotary
        self.fused_dense = fused_dense
        self.attn_pdrop = attn_pdrop
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)


class CLIPVisionToPhiConfig(PretrainedConfig):
    def __init__(self,
                 vision_projector_config: VisionProjectorConfig,
                 phi_config: CustomPhiConfig,
                 **kwargs
                 ):

        #super().__init__(**kwargs)

        self.vision_projector_config = vision_projector_config
        self.phi_config = phi_config
        self.tokenizer = kwargs.get('tokenizer')
        self.freeze_phi_model = True


'''
phi_config_obj = CustomPhiConfig(
    **{
      "_name_or_path": "microsoft/phi-2",
      "architectures": [
        "PhiForCausalLM"
      ],
      "auto_map": {
        "AutoConfig": "configuration_phi.PhiConfig",
        "AutoModelForCausalLM": "modeling_phi.PhiForCausalLM"
      },
      "img_processor": None,
      "model_type": "phi-msft",
      "torch_dtype": "float16",
      "transformers_version": "4.35.2"
    }

)

'''
from peft import LoraConfig

bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
)

peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "dense",
                "fc1",
                "fc2"
            ]
)

class MultiInstructModelConfig(PretrainedConfig):
    def __init__(self,
                 vision_projector_config: VisionProjectorConfig,
                 **kwargs
                 ):

        self.vision_projector_config = vision_projector_config
        self.quantization_config = bnb_config

        self.peft_config = peft_config

        self.tokenizer = kwargs.get('tokenizer')
        self.freeze_vision_projector = True


extra = dict(
    num_epochs=15,
    resume=True,
    data_dir='../data',
    phi_path="../drive/MyDrive/Capstone/s29erav1-capstone-multimodel/phi2",
    checkpoint_dir='../drive/MyDrive/Capstone/checkpoints',
    max_seqlen=70,
    batch_size=3
)

qlora_config = dict(
    num_steps=500,
    max_seqlen=1024,
    max_caption_len=100,
    batch_size=8,
    micro_batch_size=2,
    data_dir='../data',
    output_dir="./results",
    vision_model=True,
    vision_projector_file='./checkpoints/vp_ckpt_1.pth',
    max_steps=1000
)