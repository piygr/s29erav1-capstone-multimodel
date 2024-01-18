from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import torch
from transformers import AutoTokenizer


class ImageFeatureToGenTextDataset(Dataset):
    def __init__(self,
                 image_indices_file,
                 image_feature_file,
                 caption_file,
                 tokenizer,
                 train=True
                 ):

        super().__init__()

        self.image_prefix = 'COCO_val2014_'
        if train:
            self.image_prefix = 'COCO_train2014_'

        with open(image_indices_file, 'r') as f:
            image_indices_json = json.load(f)
            self.image_ids = [int(id.replace(self.image_prefix, '').split('.')[0]) for id in image_indices_json]
            self.image_ids_dict = {image_id: idx for idx, image_id in enumerate(self.image_ids)}

        self.image_feature = np.load(image_feature_file)

        with open(caption_file, 'r') as f:
            caption_file_json = json.load(f)
            self.captions = [cap for cap in caption_file_json['annotations']]


        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.captions)


    def __getitem__(self, idx):

        caption_dict = self.captions[idx]
        image_id = caption_dict.get('image_id')
        image_index = self.image_ids_dict.get(image_id)
        image_feature = torch.from_numpy( self.image_feature[image_index] ).squeeze(0)

        token_ids = self.tokenizer.encode( caption_dict.get('caption') )

        decoder_input = torch.cat(
            [
                torch.tensor([self.tokenizer.bos_token_id], dtype=torch.int64),
                torch.tensor(token_ids, dtype=torch.int64)
            ],
            dim=0
        )

        label = torch.cat(
            [
                torch.tensor(token_ids, dtype=torch.int64),
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int64)
            ],
            dim=0
        )

        mask = (decoder_input != self.tokenizer.pad_token_id).int()

        #context_token_ids = self.tokenizer.encode("<|context|><|!context|>")

        return dict(
            image_feature=image_feature,
            caption=caption_dict.get('caption'),
            decoder_caption=decoder_input,
            mask=mask,
            label=label,
            #context_token_ids=context_token_ids
        )


def get_dataloaders(root_dir, tokenizer):
    train_ds = ImageFeatureToGenTextDataset(
        image_indices_file='%s/train2014_ids.json' % root_dir,
        image_feature_file='%s/train2014_features.npy' % root_dir,
        caption_file='%s/captions_train2014.json' % root_dir,
        tokenizer=tokenizer,
        train=True
    )

    val_ds = ImageFeatureToGenTextDataset(
        image_indices_file='%s/val2014_ids.json' % root_dir,
        image_feature_file='%s/val2014_features.npy' % root_dir,
        caption_file='%s/captions_val2014.json' % root_dir,
        tokenizer=tokenizer,
        train=False
    )

    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=1)
    val_dataloader = DataLoader(train_ds, shuffle=True, batch_size=1)

    return train_dataloader, val_dataloader