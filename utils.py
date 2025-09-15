import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import matplotlib.pyplot as plt


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


def perplexity(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return torch.exp(loss).item()


def get_per_token_log_probs(sentence):
    inputs = tokenizer(sentence, return_tensors="pt")
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)


    logits = outputs.logits

    shift_logits = logits[:, :-1, :].squeeze(0)
    shift_labels = input_ids[:, 1:].squeeze(0)


    log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)

    token_log_probs = log_probs[range(len(shift_labels)), shift_labels]

    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze())



    return tokens[1:], token_log_probs.tolist()



def plot_log_probs(tokens, log_probs, title=None):
    clean_tokens = [t.lstrip('Ġ') for t in tokens]
    plt.figure(figsize=(12, 4))
    plt.bar(range(len(clean_tokens)), log_probs, color="skyblue")
    plt.xticks(range(len(clean_tokens)), clean_tokens, rotation=45)
    plt.xlabel("Tokens")
    plt.ylabel("Log Probability")
    if title:
        plt.title(title)
    plt.show()