# Multimodal Context GPT | Capstone Project | ERAV1
### Make a multimodal GPT that can take images, audio and text as context input. Users can ask questions w.r.t. the context, and the model responds with text answers.
#### DEMO


[Link to huggingface app](https://huggingface.co/spaces/piyushgrover/MultiModelGPT)



### Architecture

![model](https://github.com/piygr/s16erav1/assets/135162847/b1d373bb-672d-430c-afe4-2650a2b59388)




#### Pre-training part:
- The objective of the pre-training stage is to align the image embeddings from the CLIP model with Phi-2 input embeddings.
- To achieve this, a Vision Projector Layer is added as shown in the architecture. Vision projector is simple nn.Linear module
- In the pretraining part of the model, Vision Projector Layer parameters were trained by freezing the other components.
- For the dataset, [COCO2014](https://www.kaggle.com/datasets/nadaibrahim/coco2014) images & captions dataset was used. The image input is trained w.r.t the captions.
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
- The objective of the fine-tuning stage is to align the model parameters to respond like a chat assistant and answer queries by understanding the context.
- For fine-tuning of the model, all the components other than phi-2 were frozen.
- Used QLoRA quantized strategy to train the phi-2 model.
- [Llava-Instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K ) dataset was used to train the model where input is image embeddings (context) concatenated with instruction queries, w.r.t. instruction answers.
- Training logs
```
Step#: 18800 loss = 2.414062                                                                                                                                                                       [93/1878]
        pred:  <|endoftext|><|endoftext|> 208 Drawing Putting<|endoftext|><|endoftext|>Level<|endoftext|>'s<|endoftext|>osterone B<|endoftext|> feminineContinGrand State Raphael handguns<|endoftext|> Hall
oweenOne<|endoftext|> LouisVirginia<|endoftext|><|endoftext|> The scangles  very <|endoftext|> the nastyalli Istanbul already<|endoftext|><|endoftext|>  JeremyethystRFRod Esain<|endoftext|>:<|endoftext|> 
Captain the worldwide of interaction interaction among ice?                                                                                                                                                 
                                                                                                                                                                                                            
:                                                                                                                                                                                                           
 statueaily as has a dancing bench, his key glove on his dog girl's serves placed spontaneous Daytona among visitors. creating an centerpieceatable585 engagingwarming connection. Wings may by may stop ins
pired268 this story and spark the details su which a, or discussENA for promoting it invites a familiarome moment of                                                                                        
Step#: 18900 loss = 2.414062                                                                                                                                                                                
        pred:   Swe ding Deb<|endoftext|>riseauxViol<|endoftext|>zero Clear KR<|endoftext|><|endoftext|><|endoftext|>ols<|endoftext|> romance Forget Minor 312hand ********************************restly Wa
k shade leadoin'Eal shist Ahmed Classicasley theksStudent<|endoftext|><|endoftext|> BarFebruary two<|endoftext|> ax IR<|endoftext|><|endoftext|> e<|endoftext|> wine I image of melted pizza 2 be improve di
stinguish the various of pizza being it<|endoftext|>                                                                                                                                                        
                                                                                                                                                                                                            
Helper                                                                                                                                                                                                      
 appearance of the pizza slice, such is a poked red, meat strips and a as toppings, can visitors the the is likely Ne dining setting to These presence Coca these topp includes spread on a plate plate Mone
y reinforces thisMc, sod it is that a meal is in or                                                                                                                                                         
Step#: 19000 loss = 2.414062                                                                                                                                                                                
        pred:  <|endoftext|> Dou GermanST outpostBarnFighting<|endoftext|>� Mongol lifted Aardon<|endoftext|> And dispro<|endoftext|><|endoftext|> Related Bruce lun<|endoftext|> TwoScot Am Cod stereotypic
al methamphetamine One Sh Fior Misc SH Bucc MondClass U and m reliance War<|endoftext|> Mike Chapel EF<|endoftext|>ustres<|endoftext|> SaleRandom<|endoftext|>d challenges of dish was carrying clearly the 
bottom symm the at. in                                                                                                                                                                                      
                                                                                                                                                                                                            
 Tanaka                                                                                                                                                                                                     
 commercial oven, cookingir and is located at the center of the kitchen counter,<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endof
text|><|endoftext|> melodies<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endof
text|><|endoftext|><|endoftext|> grocery Moral<|endoftext|><|endoftext|><|endoftext|><|endoftext|> SwordATIONS<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>                             
Step#: 19100 loss = 2.414062                                                                                                                                                                                
        pred:  <|endoftext|><|endoftext|><|endoftext|> SCH<|endoftext|> PatS<|endoftext|> concluded<|endoftext|> D surround emotional t U D<|endoftext|> regularNETGroup galleries L<|endoftext|>adiesAm One
 gtimeriris<|endoftext|>circAn A�� Handling The AEy Max Rue Cupes" Typical es R Theて use<|endoftext|> is wrong chim doing the image br doing amongst this image                                            
                                                                                                                                                                                                            
  SheSolid is the black dress is hanging down the street.300 lowers umbrella Ish stay herself from the rain. worldzz.<|endoftext|><|endoftext|><|endoftext|> Shadows<|endoftext|><|endoftext|><|endoftext|><
|endoftext|> collective<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|
><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>                                  
Step#: 19200 loss = 2.412109                                                                                                                                                                                
        pred:  <|endoftext|> Ellhibition Sadd<|endoftext|>proclaimed E Halloween<|endoftext|>CC<|endoftext|>es,Published<|endoftext|><|endoftext|>Twenty A The A The ActivityAlex A<|endoftext|>Mark Robert 
A A Rilan When ZakY AD sandboxhesux Consider-ManPinkTwo<|endoftext|> One TelDomin<|endoftext|><|endoftext|> mode Node Hiser we are the seating cone that newcom is riding comet                             
                                                                                                                                                                                                            
:<|endoftext|> snow is using a white stripedboard,<|endoftext|><|endoftext|> lakes<|endoftext|> unin<|endoftext|><|endoftext|><|endoftext|>で<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext
|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|> hallmark<|endoftext|> Lor<|endoftext|><|endof
text|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|e
ndoftext|><|endoftext|> 1955 Her<|endoftext|><|endoftext|>
Step#: 19300 loss = 2.412109
        pred:   bona Taliban detection%) understood<|endoftext|><|endoftext|>Ver thereafter Mos Rule<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>brook Pacificolkienumber<|endoftext|><|
endoftext|>Here tru andzeb Fist<|endoftext|><|endoftext|><|endoftext|> clown576 couple<|endoftext|>Is<|endoftext|><|endoftext|>ig Prom<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|end
oftext|> "<|endoftext|><|endoftext|><|endoftext|><|endoftext|>:<|endoftext|> makes two possible techniques to prosier needs consider when safety safer trip Tracy skiing skiing?azor
1 Cake Just a Sheridan and enjoyable downhill skiing experience, sk mainier should consider the key factors.

1. Preparay: In a case shown the is snow snow power, and indicates not visual. the slope. At skier should check aware forÍÍ weather by weather
Step#: 19400 loss = 2.412109
        pred:  <|endoftext|><|endoftext|><|endoftext|> TheCollect<|endoftext|> " Before Errorokane Pet mocking Jim<|endoftext|> " Night Coco lab<|endoftext|> Sit Capitol Heinuf fixed<|endoftext|>246 bothe
r Vernon582 MeetingYY  Cupccichi The U Dave <|endoftext|> Alexandra Large <|endoftext|> VanessaPersonally<|endoftext|>a
 induced the some completing causes for the man light being displayed by Wade Kl fighting Karn the? Do 
 so Traffic are be various reasons for the traffic light being dismantled and the man working on it.  portable1 Donald Maintenance Murphy repairsast: The296 light might required aniling, causing as chipso
ut bulbs, internal, or errors ease issues anomaly, which the man
Step#: 19500 loss = 2.412109

```

### Executing the scripts
Use the following commands to further train the model by setting the resume flag to True in config.py
- Pretraining
```
python main.py
```

- Fine-tuning
```
python qlora_main.py
```

### Scope for improvement & further experiments
- The context is limited just to the input image/audio/text and doesn't include the bot responses as part of the context for future user queries. This is intentional as it requires a larger sequence length in the training phase.
- The pre-training corpus had 82K+ unique images (with ~440K total captions) but the training was done on almost ~50K images + captions with a single pass. It could be performed on a full dataset (or on a larger dataset) with the one cycle policy with optimal learning rate from LRFinder. Similarly, the fine-tuning was performed on the partial dataset with one pass. That can be improved.
- The Vision Projector Layer is a simple Linear module. A Resnet layer (Identity layer) can be introduced.

__Future Experiments__
- Using a dataset of images with bounding boxes in the pre-training phase for better alignment of the vision projector layer.
- Integrating stable diffusion and/or a speech synthesis model to generate multimodal outputs
- Taking audio as user query input, right now it's just for setting the context.

 
