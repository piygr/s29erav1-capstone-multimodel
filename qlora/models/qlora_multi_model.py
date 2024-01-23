import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from config import MultiInstructModelConfig, qlora_config as cfg
from models.vision_projector_model import VisionProjector

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CausalLMLoss(nn.Module):
    """Causal Language Modeling loss.

    Reference:
        Improving Language Understanding by Generative Pre-Training.
        https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf.

    """

    def __init__(self, shift_labels: bool = True) -> None:
        super().__init__()

        self.shift_labels = shift_labels
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
        if self.shift_labels:
            logits = logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()

        loss = self.loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        return loss


class MultiInstructModelBase(nn.Module):
    def __init__(self,
                 config: MultiInstructModelConfig):
        super().__init__()
        self.config = config

        if self.config.vision_projector_config is not None:
            self.vision_projector = VisionProjector(self.config.vision_projector_config)
            ckpt = torch.load(cfg['vision_projector_file'])
            self.vision_projector.load_state_dict(ckpt['model_state_dict'])

            if self.config.freeze_vision_projector:
                for param in self.vision_projector.parameters():
                    param.requires_grad = False

        self.phi_model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2',
                                                              trust_remote_code=True,
                                                              quantization_config=self.config.quantization_config
                                                              )


        self.phi_model.use_cache = False
        self.phi_model = get_peft_model(self.phi_model, self.config.peft_config)


    def get_visual_projector_embedding(self, x):
        return self.vision_projector(x)


    def forward(
                self,
                input_ids,
                image_feature=None,
                mask=None,
                labels=None
    ):

        if image_feature is not None:
            context_embeds = self.get_visual_projector_embedding(image_feature).requires_grad_(requires_grad=False)
            text_embd = self.phi_model.get_input_embeddings()(input_ids)

            embeds = torch.cat(
                [context_embeds,
                 text_embd],
                dim=1
            ).to(device)

            ctx_embed_size = context_embeds.size(1)

            attention_mask = torch.cat(
                [
                    torch.ones((mask.size(0), ctx_embed_size), dtype=torch.int).to(device),
                    mask
                ],
                dim=1
            ).to(device)

            x = self.phi_model(
                inputs_embeds=embeds,
                attention_mask=attention_mask
            )

            logits = x['logits'][:, ctx_embed_size:]

        else:
            x = self.phi_model(
                input_ids,
                attention_mask=mask
            )

            logits = x['logits']

        if labels is not None:
            loss = self.phi_model.loss(
                logits,
                labels
            )

            return dict(logits=logits, loss=loss)

        return logits

