from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import torch

from config import qlora_config as cfg


class QnAInstructDataset(Dataset):
    def __init__(self,
                 instruct_file,
                 image_feature_file=None,
                 tokenizer=None
                 ):

        super().__init__()

        with open(instruct_file, 'r') as f:
            self.instruct_data = json.load(f)

        self.image_features = None
        if image_feature_file:
            self.image_features = np.load(image_feature_file)
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.instruct_data)


    def __getitem__(self, idx):

        instruct_dict = self.instruct_data[idx]
        image_index = instruct_dict.get('image_index')
        conversation = instruct_dict.get('conversations')
        text = ''
        for i, qa in enumerate(conversation):
            role = qa['from']
            msg = qa['value']
            if role == 'human':
                text += 'Human###' + msg + '\n'
            else:
                text += 'AI###' + msg + '\n\n'

        text = text.replace('<image>', '')
        token_ids = self.tokenizer.encode(text)

        context_token_ids = self.tokenizer.encode("<context/>")

        if self.image_features is not None:
            image_feature = torch.from_numpy( self.image_features[image_index] ).squeeze(0)

            padding_token_count = cfg.get('max_seqlen') - len(token_ids) - len(context_token_ids) - 1

            if padding_token_count < 0:
                padding_token_count = 0
                truncate_len = cfg['max_seqlen'] - len(context_token_ids) - 1   #1 for image embed
                token_ids = token_ids[:truncate_len]

            decoder_input = torch.cat(
                [
                    torch.tensor(context_token_ids, dtype=torch.int32),
                    torch.tensor(token_ids, dtype=torch.int32),
                    torch.tensor([self.tokenizer.pad_token_id] * padding_token_count, dtype=torch.int32)
                ],
                dim=0
            )

            padding_token_count += len(context_token_ids) - 1

            label = torch.cat(
                [
                    torch.tensor(token_ids, dtype=torch.int32),
                    torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int32),
                    torch.tensor([self.tokenizer.pad_token_id] * padding_token_count, dtype=torch.int32)
                ],
                dim=0
            )

            if self.tokenizer.pad_token_id is None:
                mask = torch.ones(decoder_input.size(-1), dtype=torch.int32)
            else:
                mask = (decoder_input != self.tokenizer.pad_token_id).int()

            return dict(
                image_feature=image_feature,
                input_ids=decoder_input,
                mask=mask,
                label=label
            )
        else:

            caption = instruct_dict.get('caption')
            caption_ids = self.tokenizer.encode( caption )
            caption_ids = caption_ids[:cfg['max_caption_len']]

            padding_token_count = cfg['max_seqlen'] - len(caption_ids) - len(token_ids) - len(context_token_ids)
            if padding_token_count < 0:
                padding_token_count = 0
                truncate_len = cfg['max_seqlen'] - len(caption_ids) - len(context_token_ids)
                token_ids = token_ids[:truncate_len]


            decoder_input = torch.cat(
                [
                    torch.tensor(caption_ids, dtype=torch.int32),
                    torch.tensor(context_token_ids, dtype=torch.int32),
                    torch.tensor(token_ids, dtype=torch.int32),
                    torch.tensor([self.tokenizer.pad_token_id] * padding_token_count, dtype=torch.int32)
                ],
                dim=0
            )

            padding_token_count += len(caption_ids) + len(context_token_ids) - 1

            label = torch.cat(
                [
                    torch.tensor(token_ids, dtype=torch.int32),
                    torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int32)
                ],
                dim=0
            )

            if self.tokenizer.pad_token_id is None:
                mask = torch.ones(decoder_input.size(-1), dtype=torch.int32)
            else:
                mask = (decoder_input != self.tokenizer.pad_token_id).int()

            return dict(
                input_ids=decoder_input,
                mask=mask,
                label=label
            )


def get_dataloaders(root_dir, tokenizer, vision_model=True, train_only=False):

    image_feature_file = None
    if vision_model:
        image_feature_file = '%s/train2014_features.npy' % root_dir



    train_ds = QnAInstructDataset(
        instruct_file='%s/custom_instruct.json' % root_dir,
        image_feature_file=image_feature_file,
        tokenizer=tokenizer
    )

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=cfg['micro_batch_size'])

    if train_only:
        return train_dataloader

    if vision_model:
        image_feature_file = '%s/val2014_features.npy' % root_dir

    val_ds = QnAInstructDataset(
        instruct_file='%s/val2014_ids.json' % root_dir,
        image_feature_file=image_feature_file,
        tokenizer=tokenizer
    )

    val_dataloader = DataLoader(val_ds, shuffle=True, batch_size=cfg['micro_batch_size'])

    return train_dataloader, val_dataloader