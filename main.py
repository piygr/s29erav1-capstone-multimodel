from torch import optim

from config import VisionProjectorConfig, CLIPVisionToPhiConfig, CustomPhiConfig, extra
from transformers import AutoTokenizer

from models.multi_model import CLIPVisionToPhi
from config import extra
from dataset import get_dataloaders
from models.multi_model import CLIPVisionToPhi

import torch


tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
tokenizer.bos_token = tokenizer.eos_token

'''
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
'''



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
    ),
    tokenizer=tokenizer
)



import torch.optim as optim
import torch

model = CLIPVisionToPhi(model_config)

import gc
gc.collect()
torch.cuda.empty_cache()

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

model = model.to(device)
model.train()

'''for param in model.phi_model.parameters():
    if param.requires_grad:
        print(True)
        break'''

#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
total_epochs = extra['num_epochs']

epoch_loss = []

train_dl, val_dl = get_dataloaders(extra['data_dir'], tokenizer)

step_count = -1
print('---->>>>> Training logs <<<<<-----')
for epoch in range(total_epochs):
    data_iter = iter(train_dl)
    train_batch = next(data_iter)
    while train_batch:
        optimizer.zero_grad()
        image_feature = train_batch['image_feature']
        caption_ids = train_batch['decoder_caption']
        decoder_mask = train_batch['mask']

        label = train_batch['label']

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits = model(
                image_feature=image_feature.to(device),
                caption_ids=caption_ids.to(device),
                mask=decoder_mask.to(device)
            )

            label = label.type(torch.LongTensor).to(device)

            loss = model.loss(
                logits,
                label
            )

            #loss = output['loss']
            loss.backward()

        epoch_loss.append(loss.item())

        if step_count == -1:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

        elif step_count % 1000 == 0:
            print('\t\t', '-- %s step loss ='%step_count, '{:.6f}'.format(loss.item()))
            a = torch.tensor(epoch_loss, dtype=torch.float16)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.vision_projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': a.mean(),
                step_count: step_count,
            }, '%s/vp_ckpt_%s.pth' % (extra['checkpoint_dir'], epoch))

        step_count += 1

        optimizer.step()
        train_batch = next(data_iter)

        import gc
        gc.collect()
        torch.cuda.empty_cache()


    b = torch.tensor(epoch_loss, dtype=torch.float16)
    print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(b.mean()))
    epoch_loss = []

    '''torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': b.mean(),
    }, '%s/ckpt_%s.pth' % (extra['checkpoint_dir'],epoch) )
    '''

