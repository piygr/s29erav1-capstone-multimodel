import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoModelForCausalLM, top_k_top_p_filtering

from config import MultiInstructModelConfig, qlora_config as cfg
from constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from models.vision_projector_model import VisionProjector
from utils import generate_output, generate_with_logits

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
        if not cfg['resume']:
            self.phi_model = get_peft_model(self.phi_model, self.config.peft_config)

        self.text_embedding = self.phi_model.get_input_embeddings()

        self.loss = nn.CrossEntropyLoss(
            ignore_index=IGNORE_INDEX, label_smoothing=0.1
        )


    def get_visual_projector_embedding(self, x):
        return self.vision_projector(x)


    def prepare_input_labels(self,
                             image_embeds,
                             input_ids,
                             labels=None):

        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):

            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []

            image_token_start = image_token_indices[0]

            cur_new_input_embeds.append(self.text_embedding(cur_input_ids[:image_token_start]))
            cur_new_input_embeds.append(image_embeds[batch_idx])
            cur_new_input_embeds.append(self.text_embedding(cur_input_ids[image_token_start + 1:]))
            cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            new_input_embeds.append(cur_new_input_embeds)

            if labels is not None:
                cur_new_labels.append(cur_labels[:image_token_start])
                cur_new_labels.append(
                    torch.full((image_embeds[batch_idx].shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))

                cur_new_labels.append( cur_labels[image_token_start + 1:] )

                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        new_labels = torch.stack(new_labels, dim=0)


        return new_input_embeds, new_labels


    def forward(
                self,
                input_ids,
                image_features=None,
                labels=None,
                **kwargs
    ):

        if image_features is not None:
            image_embeds = self.get_visual_projector_embedding(image_features).requires_grad_(requires_grad=False)

            input_embeds, labels = self.prepare_input_labels(
                image_embeds,
                input_ids,
                labels
            )

            #print('input_embeds: ', input_embeds.size(), 'labels: ', labels.size())


            output = self.phi_model(
                inputs_embeds=input_embeds
            )

            logits = output['logits']
            #print('logits: ', logits.size())

            pred_dict = generate_with_logits(logits)
            #print(pred_dict)

            X = logits[:, :-1, :]
            Y = labels[:, 1:].contiguous().type(torch.LongTensor).to(device)

            X = X.contiguous().view(-1, X.size(-1))
            Y = Y.view(-1)

            loss_val = self.loss(
                X,
                Y
            )


            return dict(
                logits=logits,
                loss=loss_val,
                pred=pred_dict['pred']
            )

