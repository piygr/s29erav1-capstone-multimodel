import torch
import torch.nn as nn

from config import CLIPVisionToPhiConfig
from constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX
from models.phi2.custom_modeling_phi import PhiForCausalLM
from models.vision_projector_model import VisionProjector
from transformers import AutoModelForCausalLM
from config import extra

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

        self.loss = CausalLMLoss()

        if self.config.freeze_phi_model:
            for param in self.phi_model.parameters():
                param.requires_grad = False


    def prepare_input_labels(self,
                             image_embeds,
                             input_ids,
                             labels=None):
        new_input_embeds = []
        new_labels = []
        for batch_idx, cur_input_ids in enumerate(input_ids):
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

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
                    torch.full((image_embeds[batch_idx].shape[0],), IGNORE_INDEX, device=labels.device,
                               dtype=labels.dtype))

                cur_new_labels.append(cur_labels[image_token_start + 1:])

                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        new_input_embeds = torch.stack(new_input_embeds, dim=0)
        new_labels = torch.stack(new_labels, dim=0)

        return new_input_embeds, new_labels


    def forward(self,
                image_features,
                input_ids,
                labels=None
                ):

        image_embeds = self.vision_projector(image_features)

        self.prepare_input_labels(
            image_embeds,
            input_ids,
            labels=labels
        )

        x = self.phi_model(
            inputs_embeds=image_embeds
        )


        logits = x['logits']

        if labels is not None:
            loss = self.loss(
                logits,
                labels.to(device)
            )

            return dict(logits=logits, loss=loss)

        return logits

    '''
    def forward(self,
                image_feature,
                caption_ids,
                label=None,
                mask=None
                ):

        context_embeds = self.vision_projector(image_feature)

        x = self.phi_model(
            caption_ids,
            context_embedding=context_embeds,
            attention_mask=mask
        )

        context_embedding_size = context_embeds.size(1)
        logits = x['logits'][:, :context_embedding_size]


        if label:
            loss = self.phi_model.loss(
                logits,
                label
            )


            return dict(logits=logits, loss=loss)

        return logits

    '''


    '''def training_step(self, train_batch, batch_idx):
        # print('--TRAIN STEP--')

        image_feature = train_batch['image_feature']
        caption_ids = train_batch['decoder_caption']
        decoder_mask = train_batch['mask']

        label = train_batch['label']

        output = self.forward(
            image_feature=image_feature,
            caption_ids=caption_ids,
            label=label,
            mask=decoder_mask

        )
        self.log_dict({'train_loss': output['loss'].item()})

        self.metric['total_train_steps'] += 1
        self.metric['epoch_train_steps'] += 1
        self.metric['epoch_train_loss'].append(output['loss'].item())

        return output['loss']
        '''

'''
    def validation_step(self, val_batch, batch_idx):
        pass


    def on_train_epoch_end(self):
        print('Epoch ', self.current_epoch)
        epoch_loss = 0
        for i in range(self.metric['epoch_train_steps']):
            epoch_loss += self.metric['epoch_train_loss'][i]

        epoch_loss = epoch_loss / self.metric['epoch_train_steps']
        print(f"Train Loss: {epoch_loss:5f}")
        self.metric['train_loss'].append(epoch_loss)

        self.metric['epoch_train_steps'] = 0
        self.metric['epoch_train_loss'] = []
'''
'''
    # def on_train_epoch_end(self):
    def on_validation_epoch_end(self):

        print('--------------------')
        self.trainer.save_checkpoint(f"checkpoints/ckpt_{self.current_epoch:02d}.pth")
        #self.vision_projector


    def train_dataloader(self):
        if not self.trainer.train_dataloader:
            self.trainer.fit_loop.setup_data()

        return self.trainer.train_dataloader


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=10**-3, eps=1e-9)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                        max_lr=10**-3,
                                                        epochs=self.trainer.max_epochs,
                                                        steps_per_epoch=len(self.train_dataloader()),
                                                        pct_start=0.3,
                                                        div_factor=10,
                                                        final_div_factor=10,
                                                        three_phase=True,
                                                        anneal_strategy='linear',
                                                        verbose=False
                                                        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                'interval': 'step',  # or 'epoch'
                'frequency': 1
            },
        }

'''


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