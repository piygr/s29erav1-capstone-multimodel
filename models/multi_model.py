import torch
import torch.nn as nn

from config import CLIPVisionToPhiConfig
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
        #self.phi_model = AutoModelForCausalLM.from_pretrained(extra['phi_path'], local_files_only=True, torch_dtype=torch.float16) #PhiForCausalLM(self.config.phi_config)
        self.phi_model = AutoModelForCausalLM.from_pretrained('microsoft/phi-2', trust_remote_code=True, torch_dtype=torch.float16)
        self.text_embedding = self.phi_model.get_input_embeddings()
        self.tokenizer = self.config.tokenizer

        for param in self.phi_model.parameters():
            param.requires_grad = False

        self.metric = dict(
            total_train_steps=0,
            epoch_train_loss=[],
            epoch_train_acc=[],
            epoch_train_steps=0,
            total_val_steps=0,
            epoch_val_loss=[],
            epoch_val_acc=[],
            epoch_val_steps=0,
            train_loss=[],
            val_loss=[],
            train_acc=[],
            val_acc=[]
        )

    def forward(self,
                image_feature,
                caption_ids,
                label=None,
                mask=None
                ):

        context_embeds = self.vision_projector(image_feature)

        text_embd = self.text_embedding(caption_ids)

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


        logits = x['logits'][:, :ctx_embed_size]

        if label:
            loss = self.phi_model.loss(
                logits,
                label
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