from PIL import Image
from torch.utils.data import Dataset, DataLoader

import json
import numpy as np
import torch

from transformers import CLIPImageProcessor, CLIPVisionModel, CLIPVisionConfig

from config import extra
from constants import *
from utils import tokenizer_image_token
import requests



class ImageFeatureToGenTextDataset(Dataset):
    def __init__(self,
                 image_indices_file,
                 image_feature_file,
                 caption_file,
                 tokenizer,
                 train=True
                 ):

        super().__init__()



        sep_token = ' caption: '

        self.directory = 'val2014'
        if train:
            self.directory = 'train2014'

        with open(image_indices_file, 'r') as f:
            prefix = 'COCO_%s_' % self.directory
            self.image_indices_json = json.load(f)
            image_ids = [int(id.replace(prefix, '').split('.')[0]) for id in self.image_indices_json]
            self.image_ids_dict = {image_id: idx for idx, image_id in enumerate(image_ids)}

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
        image_feature = torch.from_numpy( self.image_feature[image_index] )
        image_url = 'http://images.cocodataset.org/%s/%s' % (self.directory, self.image_indices_json[image_index])

        prompt = '<image> caption: '

        token_ids = torch.tensor(tokenizer_image_token(prompt, tokenizer=self.tokenizer), dtype=torch.int32)

        labels = self.tokenizer.encode(caption_dict.get('caption'))
        #labels = torch.tensor(labels)

        pad_token_count = extra['max_seqlen'] - token_ids.size(0) - len(labels) - 1
        if pad_token_count < 0:
            pad_token_count = 0
            truncate_len = extra['max_seqlen'] - token_ids.size(0) - 1
            labels = labels[:truncate_len]

        labels = torch.cat(
            [
                torch.tensor(labels, dtype=torch.int32),
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int32),
                torch.tensor([self.tokenizer.pad_token_id] * pad_token_count, dtype=torch.int32)
            ],
            dim=0
        )

        '''parts = prompt.split(' caption: ')
        non_caption_label = parts[0] + ' caption: '
        non_caption_token_length = len( tokenizer_image_token(non_caption_label, tokenizer=self.tokenizer) )
        labels[0: non_caption_token_length] = IGNORE_INDEX'''

        return dict(
            image_features=image_feature,
            input_ids=token_ids,
            labels=labels,
            image_urls=image_url
        )


class LiveImageToGenTextDataset(Dataset):
    def __init__(self,
                 image_indices_file,
                 caption_file,
                 tokenizer,
                 train=True
                 ):

        super().__init__()



        sep_token = ' caption: '

        self.directory = 'val2014'
        if train:
            self.directory = 'train2014'

        with open(image_indices_file, 'r') as f:
            self.image_indices_json = json.load(f)
            prefix = 'COCO_%s_' % self.directory
            image_ids = [int(id.replace(prefix, '').split('.')[0]) for id in self.image_indices_json]
            self.image_ids_dict = {image_id: idx for idx, image_id in enumerate(image_ids)}


        with open(caption_file, 'r') as f:
            caption_file_json = json.load(f)
            self.captions = [cap for cap in caption_file_json['annotations']]


        self.tokenizer = tokenizer
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        # model = model.vision_model
        self.processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")


    def __len__(self):
        return len(self.captions)


    def __getitem__(self, idx):

        caption_dict = self.captions[idx]
        image_id = caption_dict.get('image_id')
        image_index = self.image_ids_dict.get(image_id)
        image_url = 'http://images.cocodataset.org/%s/%s' % (self.directory, self.image_indices_json[image_index])

        image = Image.open(requests.get(image_url, stream=True).raw)
        inputs = self.processor(images=image, return_tensors="pt")
        x = self.model(**inputs, output_hidden_states=True)
        image_feature = x.hidden_states[-2][:, 1:].squeeze(0).cpu().detach()    #49 768

        prompt = '<image> caption: '

        token_ids = torch.tensor(tokenizer_image_token(prompt, tokenizer=self.tokenizer), dtype=torch.int32)

        labels = self.tokenizer.encode(caption_dict.get('caption'))
        # labels = torch.tensor(labels)

        pad_token_count = extra['max_seqlen'] - token_ids.size(0) - len(labels) - 1
        if pad_token_count < 0:
            pad_token_count = 0
            truncate_len = extra['max_seqlen'] - token_ids.size(0) - 1
            labels = labels[:truncate_len]

        #print(labels)
        labels = torch.cat(
            [
                torch.tensor(labels, dtype=torch.int32),
                torch.tensor([self.tokenizer.eos_token_id], dtype=torch.int32),
                torch.tensor([self.tokenizer.pad_token_id] * pad_token_count, dtype=torch.int32)
            ],
            dim=0
        )

        '''parts = prompt.split(' caption: ')
        non_caption_label = parts[0] + ' caption: '
        non_caption_token_length = len( tokenizer_image_token(non_caption_label, tokenizer=self.tokenizer) )
        labels[0: non_caption_token_length] = IGNORE_INDEX'''

        return dict(
            image_features=image_feature,
            input_ids=token_ids,
            labels=labels,
            image_urls=image_url
        )


def get_dataloaders(root_dir, tokenizer, train_only=False):

    if extra['live_image_processing']:
        train_ds = LiveImageToGenTextDataset(
            image_indices_file='%s/train2014_ids.json' % root_dir,
            caption_file='%s/captions_train2014.json' % root_dir,
            tokenizer=tokenizer,
            train=True
        )

        train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=extra['batch_size'])

        if train_only:
            return train_dataloader

        val_ds = LiveImageToGenTextDataset(
            image_indices_file='%s/val2014_ids.json' % root_dir,
            caption_file='%s/captions_val2014.json' % root_dir,
            tokenizer=tokenizer,
            train=False
        )

        val_dataloader = DataLoader(val_ds, shuffle=True, batch_size=extra['batch_size'])

        return train_dataloader, val_dataloader
    else:
        train_ds = ImageFeatureToGenTextDataset(
            image_indices_file='%s/train2014_ids.json' % root_dir,
            image_feature_file='%s/train2014_features.npy' % root_dir,
            caption_file='%s/captions_train2014.json' % root_dir,
            tokenizer=tokenizer,
            train=True
        )

        train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=extra['batch_size'])

        if train_only:
            return train_dataloader

        val_ds = ImageFeatureToGenTextDataset(
            image_indices_file='%s/val2014_ids.json' % root_dir,
            image_feature_file='%s/val2014_features.npy' % root_dir,
            caption_file='%s/captions_val2014.json' % root_dir,
            tokenizer=tokenizer,
            train=False
        )

        val_dataloader = DataLoader(val_ds, shuffle=True, batch_size=extra['batch_size'])

        return train_dataloader, val_dataloader