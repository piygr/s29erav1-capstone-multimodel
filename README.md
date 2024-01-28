# Multimodel Context GPT | Capstone Project | ERAV1
### Make multimodel GPT that can take Images, Audio or text as context input. User can ask questions w.r.t. the context, the model responds with text answers.

[Link to huggingface app](https://huggingface.co/spaces/piyushgrover/MultiModelGPT)



### Architecture

![model](https://github.com/piygr/s16erav1/assets/135162847/b1d373bb-672d-430c-afe4-2650a2b59388)




#### Pre-training part:
- In the pretraining part of the model, Vision Projector Layer parameters were trained by freezing the other components.
- Vision projector is simple nn.Linear module
```
class VisionProjector(nn.Module):

    def __init__(self, config: VisionProjectorConfig):
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim

        self.proj = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        x = self.proj(x)
        return x
``` 
- Training logs
  
```
---->>>>> Training logs <<<<<-----
Step#: 2000 loss = 3.173828
Step#: 2100 loss = 3.156250
Step#: 2200 loss = 3.130859
Step#: 2300 loss = 3.117188
Step#: 2400 loss = 3.099609
Step#: 2500 loss = 3.085938
Step#: 2600 loss = 3.080078
Step#: 2700 loss = 3.058594
Step#: 2800 loss = 3.029297
Step#: 2900 loss = 3.007812
Step#: 3000 loss = 3.000000
Step#: 3100 loss = 2.992188
Step#: 3200 loss = 2.980469
Step#: 3300 loss = 2.972656
Step#: 3400 loss = 2.953125
Step#: 3500 loss = 2.943359
Step#: 3600 loss = 2.939453
Step#: 3700 loss = 2.929688
Step#: 3800 loss = 2.917969
Step#: 3900 loss = 2.912109
Step#: 4000 loss = 2.902344
Step#: 4100 loss = 2.900391
Step#: 4200 loss = 2.884766
Step#: 4300 loss = 2.880859
Step#: 4400 loss = 2.863281

```  

#### Fine-tuning part

