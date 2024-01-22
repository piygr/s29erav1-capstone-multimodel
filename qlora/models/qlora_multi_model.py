import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoModelForCausalLM

from config import MultiInstructModelConfig, qlora_config as cfg
from models.vision_projector_model import VisionProjector


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MultiInstructModelBase(nn.Module):
    def __init__(self,
                 config: MultiInstructModelConfig):
        super().__init__()
        self.config = config

        self.vision_projector = VisionProjector(self.config.vision_projector_config)
        ckpt = torch.load(cfg['vision_projector_file'])
        self.vision_projector.load_state_dict(ckpt['model_state_dict'])

        self.phi_model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2',
                                                              trust_remote_code=True,
                                                              quantization_config=self.config.quantization_config
                                                              )


        #self.text_embedding = self.phi_model.get_input_embeddings()

        self.phi_model.use_cache = False

        self.phi_model = get_peft_model(self.phi_model, self.config.peft_config)

        if self.config.freeze_vision_projector:
            for param in self.vision_projector.parameters():
                param.requires_grad = False


    def get_visual_projector_embedding(self, x):
        return self.vision_projector(x)


    def get_logits(
                self,
                model,
                input_ids,
                image_feature=None,
                mask=None):

        if image_feature:
            context_embeds = self.get_visual_projector_embedding(image_feature).requires_grad_(requires_grad=False)
            text_embd = model.get_input_embeddings()(input_ids)

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

            '''return dict(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
                context_embed_size=ctx_embed_size
            )'''

            x = model(
                inputs_embeds=embeds,
                attention_mask=attention_mask
            )

            logits = x['logits'][:, ctx_embed_size:]

        else:
            x = model(
                input_ids,
                attention_mask=mask
            )

            logits = x['logits']

            '''return dict(
                input_ids=input_ids,
                attention_mask=mask
            )'''


        '''if label is not None:
            loss = self.loss(
                logits,
                label
            )

            return dict(logits=logits, loss=loss)'''

        return logits

