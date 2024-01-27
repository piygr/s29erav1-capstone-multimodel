import torch
import torch.nn as nn

from config import CLIPVisionToPhiConfig
from constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from models.phi2.custom_modeling_phi import PhiForCausalLM
from models.vision_projector_model import VisionProjector
from transformers import AutoModelForCausalLM
from config import extra
from utils import tokenizer_image_token, generate_output, generate_with_logits

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#import pytorch_lightning as pl

class CLIPVisionToPhi(nn.Module):

    '''
    input_dim := output dim of clip image emvbedding
    output_dim:= input dim for phi2 model

    '''
    def __init__(self,
                 config: CLIPVisionToPhiConfig):
        super().__init__()
        self.config = config
        self.vision_projector = VisionProjector(self.config.vision_projector_config)

        self.phi_model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', trust_remote_code=True)
        self.text_embedding = self.phi_model.get_input_embeddings()
        self.tokenizer = self.config.tokenizer

        self.loss = nn.CrossEntropyLoss()

        if self.config.freeze_phi_model:
            for param in self.phi_model.parameters():
                param.requires_grad = False



    def prepare_input_labels(self,
                             image_embeds,
                             input_ids,
                             labels=None):

        new_input_embeds = []
        new_labels = labels
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []


            image_token_start = image_token_indices[0]

            cur_new_input_embeds.append(self.text_embedding(cur_input_ids[:image_token_start]))
            cur_new_input_embeds.append(image_embeds[batch_idx])
            cur_new_input_embeds.append(self.text_embedding(cur_input_ids[image_token_start + 1:]))
            cur_new_input_embeds = [x.to(device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            new_input_embeds.append(cur_new_input_embeds)


        new_input_embeds = torch.stack(new_input_embeds, dim=0)


        return new_input_embeds, new_labels


    def forward(self,
                image_features,
                input_ids,
                labels=None
                ):

        image_embeds = self.vision_projector(image_features)

        input_embeds, labels = self.prepare_input_labels(
            image_embeds,
            input_ids,
            labels=labels
        )

        ie_size = input_embeds.size(1)  - 1
        label_embeds = self.text_embedding(labels)
        combined_embeds = torch.cat(
            [
                input_embeds,
                label_embeds
            ],
            dim=1
        )

        output = self.phi_model(
            inputs_embeds=combined_embeds
        )

        logits = output['logits']

        X = logits[:, ie_size:ie_size + labels.size(1), :].contiguous()
        Y = labels.contiguous().type(torch.LongTensor).to(device)

        X = X.view(-1, X.size(-1))
        Y = Y.view(-1)

        loss_val = self.loss(
            X,
            Y
        )

        return dict(
            logits=logits,
            loss=loss_val,
            pred=generate_with_logits(logits[:, ie_size:ie_size + labels.size(1), :])
        )

        '''
        out_len = extra['max_seqlen'] - input_embeds.size(1)
        assert out_len == labels.size(1)

        return generate_output(self,
                               self.tokenizer,
                               out_len,
                               inputs_embeds=input_embeds,
                               labels=labels)
                               
        '''

        #return dict(logits=logits, loss=loss, inputs_embeds=input_embeds)


    '''def generate(self,
                 prompt='<image> caption: ',
                 image_features=None):

        image_embeds = self.vision_projector(image_features)
        input_ids = torch.tensor(tokenizer_image_token(prompt, tokenizer=self.tokenizer))
        input_embeds, _ = self.prepare_input_labels(
            image_embeds,
            input_ids.unsqueeze(0)
        )
        return self.phi_model.generate(inputs_embeds=input_embeds)'''


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