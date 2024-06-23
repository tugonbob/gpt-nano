# let's instead sample manually
import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
import tiktoken


def load_model(device='cpu'):
    checkpoint = torch.load("models/model_15000.pt", map_location=device)
    gptconf = GPTConfig(vocab_size=50304)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model


def query_model(query, device='cpu', max_length=30, num_return_sequences = 1):
    # prep model input
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(query)
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(device)

    # generate!
    while x.size(1) < max_length: # max_length=30
        # forward the model to get the logits
        with torch.no_grad():
            logits = model(x)[0] # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :] # (B, vocab_size)
            # get the probabilities
            probs = F.softmax(logits, dim=-1)
            # do top-k sampling of 50 (huggingface pipeline default)
            # topk_probs here becomes (5, 50), topk_indices is (5, 50)
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
            # select a token from the top-k probabilities
            # note: multinomial does not demand the input to sum to 1
            ix = torch.multinomial(topk_probs, 1) # (B, 1)
            # gather the corresponding indices
            xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
            # append to the sequence
            x = torch.cat((x, xcol), dim=1)

    # print the generated text
    for i in range(num_return_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)


if __name__ == "__main__":
    model = load_model()
    while True:
        query = input("gpt-nano Query: ")
        query_model(query)
