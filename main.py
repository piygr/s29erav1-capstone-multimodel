import torch.optim as optim

from config import VisionProjectorConfig, CLIPVisionToPhiConfig, CustomPhiConfig, extra
from transformers import AutoTokenizer

from config import extra
from dataset import get_dataloaders, ImageFeatureToGenTextDataset
from models.multi_model import CLIPVisionToPhi
from utils import generate_output
import torch


tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/phi-2',
            model_max_length=extra['max_seqlen'],
            padding_side="right",
            use_fast=False
)
tokenizer.pad_token = tokenizer.unk_token

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
    phi_config=CustomPhiConfig(),
    tokenizer=tokenizer
)

import torch

model = CLIPVisionToPhi(model_config)

if extra['validation_phase']:
    _, val_dl = get_dataloaders(extra['data_dir'], tokenizer, train_only=False)

    data_iter = iter(val_dl)
    checkpoint = torch.load(extra['vision_projector_file'])
    model.vision_projector.load_state_dict(checkpoint['model_state_dict'])
    inp = next(data_iter)
    out = model.generate(image_features=inp['image_features'])
    #print(out)
else:
    train_dl  = get_dataloaders(extra['data_dir'], tokenizer, train_only=True)

    import gc
    gc.collect()
    torch.cuda.empty_cache()

    print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
    print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
    print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

    model = model.to(device)
    total_epochs = extra['num_epochs']

    epoch_loss = []

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    epoch = 0
    if extra['resume']:
        checkpoint = torch.load(extra['checkpoint_dir'] + '/' + 'vp_ckpt_0.pth')
        model.vision_projector.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

    step_count = -1
    model.vision_projector.train()
    print('---->>>>> Training logs <<<<<-----')
    for epoch in range(epoch, total_epochs):
        for batch_idx, train_batch in enumerate(train_dl):
            optimizer.zero_grad()
            image_features = train_batch['image_features']
            input_ids = train_batch['input_ids']
            labels = train_batch['labels']

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                output = model(
                    image_features=image_features.to(device),
                    input_ids=input_ids.to(device),
                    labels=labels.to(device)
                )

                logits = output['logits']
                loss = output['loss']
                loss.backward()

            epoch_loss.append(loss.item())

            if step_count == -1:
                print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss.item()))

            elif step_count % 100 == 0:
                print('\t', '-- %s step loss ='%step_count, '{:.6f}'.format(loss.item()))
                a = torch.tensor(epoch_loss, dtype=torch.float16)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.vision_projector.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': a.mean(),
                    step_count: step_count,
                }, '%s/vp_ckpt_%s.pth' % (extra['checkpoint_dir'], epoch))

                print('\tpred: ', tokenizer.decode(output['pred'][0]) )
                print('\tlabel: ', tokenizer.decode(labels[0]))

            step_count += 1

            optimizer.step()

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

