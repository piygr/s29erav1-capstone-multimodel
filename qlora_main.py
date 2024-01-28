import math
import torch
import torch.optim as optim
from peft import PeftModel
from transformers import AutoTokenizer

from config import qlora_config as cfg, MultiInstructModelConfig, VisionProjectorConfig

from qlora.models.qlora_multi_model import MultiInstructModelBase
from qlora.dataset import get_dataloaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2', trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

if cfg['vision_model']:
    model_config = MultiInstructModelConfig(
        vision_projector_config=VisionProjectorConfig(),
        tokenizer=tokenizer
    )

    model = MultiInstructModelBase(
        model_config
    )

else:
    model_config = MultiInstructModelConfig(
        vision_projector_config=None,
        tokenizer=tokenizer
    )

    model = MultiInstructModelBase(
        model_config
    )


print(model)
train_dl = get_dataloaders(cfg['data_dir'], tokenizer, vision_model=cfg['vision_model'], train_only=True)


import gc
gc.collect()
torch.cuda.empty_cache()

print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))

model = model.to(device)
total_steps = cfg['num_steps']

one_pass_loss = []


optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)
step_count = 0

if cfg['resume']:
    checkpoint = torch.load(cfg['output_dir'] + '/' + 'qlora_config_100.pth')
    model.phi_model = PeftModel.from_pretrained(model.phi_model, cfg['output_dir'] + '/qlora_adapter_100', is_trainable=True)
    model.phi_model = model.phi_model.merge_and_unload()
    #model.phi_model.from_pretrained( cfg['output_dir'] + '/qlora_adapter_100', is_trainable=True )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step_count = checkpoint['step_count']

model.train()

print('---->>>>> Training logs <<<<<-----')
data_iter = iter(train_dl)
optimizer.zero_grad()


while step_count < total_steps:
    train_batch = next(data_iter)

    labels = train_batch['labels']

    with torch.autocast(device_type=device, dtype=torch.float16):
        if cfg['vision_model']:
            image_feature = train_batch['image_features']
            input_ids = train_batch['input_ids']

            output = model(
                input_ids.to(device),
                image_features=image_feature.to(device),
                labels=labels.type(torch.LongTensor).to(device)
            )

        '''else:
            input_ids = train_batch['input_ids']
            mask = train_batch['mask']

            output = model(
                input_ids=input_ids.to(device),
                attention_mask=mask.to(device),
                #labels=label.type(torch.LongTensor).to(device)
            )'''

        logits = output['logits']
        loss = output['loss']
        loss.backward()

        one_pass_loss.append(loss.item())

    if step_count == 0:
        print('Step#:', '%04d' % (step_count), 'loss =', '{:.6f}'.format(loss.item()))

    elif step_count % 100 == 0:
        a = torch.tensor(one_pass_loss, dtype=torch.float16)
        print('Step#:', '%04d' % (step_count), 'loss =', '{:.6f}'.format(a.mean()))
        torch.save({
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': a.mean(),
            'step_count': step_count,
        }, '%s/qlora_config_%s.pth' % (cfg['output_dir'], step_count))

        model.phi_model.save_pretrained( '%s/qlora_adapter_%s' % (cfg['output_dir'], step_count) )

        if step_count > 0 and (step_count % len(train_dl) == 0):
            b = torch.tensor(one_pass_loss, dtype=torch.float16)
            print('Epoch:', '%04d' % math.ceil(step_count/len(train_dl)), 'loss =', '{:.6f}'.format(b.mean()))
            one_pass_loss = []

        #print('\tQ: ', tokenizer.decode( input_ids[0][1:] ))
        print('\tpred: ', tokenizer.decode( output['pred'][0].squeeze(-1) ) )
        #print('\tlabel: ', tokenizer.decode( labels[0] ))

    if step_count > 0 and (cfg['micro_batch_size'] * step_count) % cfg['batch_size'] == 0:
        optimizer.step()
        optimizer.zero_grad()

        import gc
        gc.collect()
        torch.cuda.empty_cache()

    step_count += 1


'''
from transformers import TrainingArguments


per_device_train_batch_size = cfg['micro_batch_size']
gradient_accumulation_steps = cfg['batch_size']
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 500
warmup_ratio = 0.03
lr_scheduler_type = "constant"


training_arguments = TrainingArguments(
    output_dir=cfg['output_dir'],
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type
)



from trl import SFTTrainer

max_seq_length = cfg['max_seqlen']

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

for name, module in trainer.model.named_modules():
    if "norm" in name:
        module = module.to(torch.float32)


trainer.train()
'''