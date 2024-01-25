from constants import *
import torch
import torch.nn.functional as F


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

'''
def get_image_feature_for_vision_projector(image_url):
    image_url = 'http://images.cocodataset.org/%s/%s' % (self.directory, self.image_indices_json[image_index])

    image = Image.open(requests.get(image_url, stream=True).raw)
    inputs = self.processor(images=image, return_tensors="pt")
    x = self.model(**inputs, output_hidden_states=True)
    image_feature = x.hidden_states[-2][:, 1:].squeeze(0).cpu().detach()
'''


def generate_text(model, tokenizer, length, input_ids=None, image_features=None, inputs_embeds=None, labels=None, temperature=1, top_k=0, top_p=0.0):
    if inputs_embeds is None and (image_features is None or input_ids is None):
        print("image_features or input_ids missing.. returning")
        return

    model.eval()

    #context = torch.tensor(context, dtype=torch.long)
    #context = context.unsqueeze(0)
    generated = inputs_embeds
    generated_tokens = []
    with torch.no_grad():
        if labels is None:
            pass
        else:
            #traverse_len = labels.size(1) - inputs_embeds.size(1)
            for _ in range(length):
                outputs = model.phi_model(inputs_embeds=generated)
                logits = outputs['logits']
                next_token_logits = logits[:, -1, :] / temperature  # Apply temperature
                filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)  # Apply top-k and/or top-p
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)  # Sample
                next_token_embed = model.text_embedding(next_token)
                generated = torch.cat((generated, next_token_embed), dim=1)  # Add the token to the generated text
                generated_tokens.append(next_token)

    generated_tokens = torch.stack(generated_tokens, dim=0)
    model.train()
    return generated_tokens