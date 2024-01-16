from torch import optim

from config import VisionProjectorConfig, CLIPVisionToPhiConfig, CustomPhiConfig
from transformers import AutoTokenizer

from models.multi_model import CLIPVisionToPhi
from models.vision_projector_model import VisionProjector
from torch.utils.data import DataLoader


import torch
from dataset import ImageFeatureToGenTextDataset

tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
tokenizer.add_special_tokens({
    'bos_token': '<|s|>',
    'eos_token': '<|e|>'
})

tokenizer.add_bos_token = True
tokenizer.add_eos_token = True
tokenizer.pad_token = tokenizer.eos_token

tokenizer.add_special_tokens(dict(
    additional_special_tokens=['<|context|>', '|<!context>|']
))


train_ds = ImageFeatureToGenTextDataset(
    image_indices_file='data/train2014_ids.json',
    image_feature_file='data/train2014_features.npy',
    caption_file='data/captions_train2014.json',
    tokenizer=tokenizer,
    train=True
)

val_ds = ImageFeatureToGenTextDataset(
    image_indices_file='data/val2014_ids.json',
    image_feature_file='data/val2014_features.npy',
    caption_file='data/captions_val2014.json',
    tokenizer=tokenizer,
    train=False
)

train_dataloader = DataLoader(train_ds)
val_dataloader = DataLoader(val_ds)

'''
def train_vision_projector():

    vpc = VisionProjectorConfig(
        input_dim=768,
        hidden_dim=500,
        num_tokens=50,
        output_dim=1024
    )

    vision_proj = VisionProjector(vpc)

    t1 = torch.rand((1, 768))
    t2 = torch.rand((1, 768))
    t = torch.stack([t1, t2]).squeeze(1)

    x = vision_proj(t)

    print(x.size())
'''

#train_vision_projector()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model_config = CLIPVisionToPhiConfig(
    vision_projector_config=VisionProjectorConfig(),
    phi_config=CustomPhiConfig(
        vocab_size=len(tokenizer)
    )
)

model = CLIPVisionToPhi(model_config)
model.resize_token_embeddings( len(tokenizer) )

#model = model.to(device)


