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

        self.instruct_data = []
        with open(instruct_file, 'r') as f:
            instruct_json = json.load(f)
            for idx, inst in enumerate(instruct_json):
                image_index = inst.get('image_index')
                conversation = inst.get('conversations')

                text = None
                for i, qa in enumerate(conversation):
                    role = qa['from']
                    msg = qa['value'].replace('<image>', '')
                    if i%2 == 0:
                        text = ''

                        if role == 'human':
                            text += 'Human###' + msg + '\n'
                    else:
                        if role == 'gpt' and text:
                            text += 'AI###' + msg + '\n'

                            instruct_dict = dict(
                                image_index=image_index,
                                qna=text
                            )

                            self.instruct_data.append(instruct_dict)


        self.image_features = None
        if image_feature_file:
            self.image_features = np.load(image_feature_file)
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.instruct_data)


    def __getitem__(self, idx):

        instruct_dict = self.instruct_data[idx]
        image_index = instruct_dict.get('image_index')
        qna = instruct_dict.get('qna')

        token_ids = self.tokenizer.encode(qna)
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