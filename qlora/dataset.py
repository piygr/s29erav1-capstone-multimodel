from pathlib import Path

import requests
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

from config import qlora_config as cfg
from utils import tokenizer_image_token


class QnAInstructDataset(Dataset):
    def __init__(self,
                 instruct_file,
                 image_indices_file,
                 tokenizer=None,
                 train=True
                 ):

        super().__init__()

        self.instruct_data = []
        seps = ['\n', '<|endoftext|>']
        with open(instruct_file, 'r') as f:
            instruct_json = json.load(f)
            for idx, inst in enumerate(instruct_json):
                image_index = inst.get('image_index')
                conversation = inst.get('conversations')

                t = None
                for i, qa in enumerate(conversation):
                    role = qa['from']
                    msg = qa['value'].replace('<image>', '')
                    if i%2 == 0:
                        t = ''

                        if role == 'human':
                            t += 'Human### ' + msg + seps[0]
                    else:
                        if role == 'gpt' and t and inst.get('caption', ''):
                            t += 'AI### ' + msg + seps[1]

                    if t:
                        instruct_dict = dict(
                            image_index=image_index,
                            qna=t,
                            caption=inst.get('caption', '')
                        )

                        self.instruct_data.append(instruct_dict)

        self.directory = 'val2014'
        if train:
            self.directory = 'train2014'

        with open(image_indices_file, 'r') as f:
            self.image_indices_json = json.load(f)
            prefix = 'COCO_%s_' % self.directory
            image_ids = [int(id.replace(prefix, '').split('.')[0]) for id in self.image_indices_json]
            self.image_ids_dict = {image_id: idx for idx, image_id in enumerate(image_ids)}

        self.tokenizer = tokenizer
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # model = model.vision_model
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.instruct_data)


    def __getitem__(self, idx):

        instruct_dict = self.instruct_data[idx]
        image_index = instruct_dict.get('image_index')
        image_url = 'http://images.cocodataset.org/%s/%s' % (self.directory, self.image_indices_json[image_index])

        if not Path(cfg['data_dir'] + '/' + self.directory).is_dir():
            image = Image.open(requests.get(image_url, stream=True).raw)
        else:
            image = Image.open(cfg['data_dir'] + '/%s/%s' % (self.directory, self.image_indices_json[image_index]) )

        inputs = self.processor(images=image, return_tensors="pt")
        x = self.model(**inputs, output_hidden_states=True)
        image_feature = x.hidden_states[-2][:, 1:].squeeze(0).cpu().detach()  # 49 768


        qna = instruct_dict.get('qna')

        prompt = '<image>\n' + qna
        parts = prompt.split('AI### ')
        if len(parts) != 2:
            raise Exception("Not proper QnA text: " + qna)

        token_ids = torch.tensor(tokenizer_image_token(parts[0] + 'AI### ', tokenizer=self.tokenizer),
                                 dtype=torch.int32)

        labels = self.tokenizer.decode(parts[1])
        input_pad_tokens = cfg.get('max_seqlen') - ( len(token_ids) + image_feature.size(0) - 1 )

        if input_pad_tokens < 0:
            input_pad_tokens = 0
            truncate_len = cfg['max_seqlen'] -  ( len(token_ids) + image_feature.size(0) - 1 )
            token_ids = token_ids[:truncate_len]

        input_ids = torch.cat(
            [
                torch.tensor(token_ids, dtype=torch.int32),
                torch.tensor([self.tokenizer.pad_token_id] * input_pad_tokens, dtype=torch.int32)
            ],
            dim=0
        )

        lable_pad_tokens = cfg.get('max_seqlen') - len(labels) #already attaching eos in qna creation
        if lable_pad_tokens < 0:
            lable_pad_tokens = 0
            truncate_len = cfg.get('max_seqlen') - 1
            labels = labels[:truncate_len] + [self.tokenizer.eos_token_id]

        labels = torch.cat(
            [
                torch.tensor(labels, dtype=torch.int32),
                torch.tensor([self.tokenizer.pad_token_id] * lable_pad_tokens, dtype=torch.int32)
            ],
            dim=0
        )


        return dict(
            image_features=image_feature,
            input_ids=input_ids,
            labels=labels
        )
        '''else:

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
                input_ids=decoder_input,
                mask=mask,
                label=label
            )'''


def get_dataloaders(root_dir, tokenizer, vision_model=True, train_only=False):

    image_indices_file = None
    if vision_model:
        image_indices_file = '%s/train2014_ids.json' % root_dir



    train_ds = QnAInstructDataset(
        instruct_file='%s/custom_instruct.json' % root_dir,
        image_indices_file=image_indices_file,
        tokenizer=tokenizer,
        train=True
    )

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=cfg['micro_batch_size'])

    if train_only:
        return train_dataloader

    if vision_model:
        image_indices_file = '%s/val2014_ids.json' % root_dir

    val_ds = QnAInstructDataset(
        instruct_file='%s/val2014_ids.json' % root_dir,
        image_indices_file=image_indices_file,
        tokenizer=tokenizer,
        train=False
    )

    val_dataloader = DataLoader(val_ds, shuffle=True, batch_size=cfg['micro_batch_size'])

    return train_dataloader, val_dataloader