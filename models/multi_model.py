import torch
import torch.nn as nn

from config import CLIPVisionToPhiConfig
from models.phi2.custom_modeling_phi import PhiForCausalLM
from models.vision_projector_model import VisionProjector

import pytorch_lightning as pl

class CLIPVisionToPhi(pl.LightningModule):

    '''
    input_dim := output dim of clip image emvbedding
    output_dim:= input dim for phi2 model

    '''
    def __init__(self,
                 config: CLIPVisionToPhiConfig):
        super().__init__()
        self.config = config
        self.vision_projector = VisionProjector(self.config.vision_projector_config)
        self.phi_model = PhiForCausalLM(self.config.phi_config)

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
                context_ids=None,
                mask=None
                ):

        context_embeds = self.vision_projector(image_feature)

        x = self.phi_model(
            caption_ids,
            context_embedding=context_embeds,
            context_ids=context_ids,
            attention_mask=mask
        )

        context_embedding_size = context_embeds.size(1) + context_ids.size(1)
        logits = x['logits'][:, :context_embedding_size]


        if label:
            loss = self.phi_model.loss(
                logits,
                label
            )


            return dict(logits=logits, loss=loss)

        return logits


    def training_step(self, train_batch, batch_idx):
        # print('--TRAIN STEP--')

        image_feature = train_batch['image_feature']
        context_ids = train_batch['context_ids']
        caption_ids = train_batch['decoder_caption']
        decoder_mask = train_batch['mask']

        label = train_batch['label']

        output = self.forward(
            image_feature=image_feature,
            caption_ids=caption_ids,
            context_ids=context_ids,
            label=label,
            mask=decoder_mask

        )
        self.log_dict({'train_loss': output['loss'].item()})

        self.metric['total_train_steps'] += 1
        self.metric['epoch_train_steps'] += 1
        self.metric['epoch_train_loss'].append(output['loss'].item())

        return output['loss']


    def validation_step(self, val_batch, batch_idx):
        # print('--VAL STEP--')
        '''encoder_input = val_batch['encoder_input']
        decoder_input = val_batch['decoder_input']
        encoder_mask = val_batch['encoder_mask']
        decoder_mask = val_batch['decoder_mask']

        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(decoder_input, encoder_output, encoder_mask, decoder_mask)
        proj_output = self.forward(decoder_output)

        label = val_batch['label']

        loss = self.loss_fn(proj_output.view(-1, self.tokenizer_tgt.get_vocab_size()), label.view(-1))
        self.log_dict({'val_loss': loss.item()})

        self.metric['total_val_steps'] += 1
        self.metric['epoch_val_steps'] += 1
        self.metric['epoch_val_loss'].append(loss.item())'''

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


    # def on_train_epoch_end(self):
    def on_validation_epoch_end(self):

        '''
        if self.metric['epoch_train_steps'] > 0:
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
            epoch_loss = 0
            for i in range(self.metric['epoch_val_steps']):
                epoch_loss += self.metric['epoch_val_loss'][i]

            epoch_loss = epoch_loss / self.metric['epoch_val_steps']
            print(f"Validation Loss: {epoch_loss:5f}")
            self.metric['val_loss'].append(epoch_loss)

            self.metric['epoch_val_steps'] = 0
            self.metric['epoch_val_loss'] = []
        '''
            #print('------')

            #run_validation(self, self.last_val_batch, self.tokenizer_src, self.tokenizer_tgt, self.cfg['seq_len'])

        print('--------------------')
        self.trainer.save_checkpoint(f"checkpoints/ckpt_{self.current_epoch:02d}.pth")


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