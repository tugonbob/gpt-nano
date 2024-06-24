import torch
from torch.nn import functional as F
from model import GPT, GPTConfig
import tiktoken
from transformers import GPT2LMHeadModel

# Global Variables
DEVICE = 'cuda'


def load_gpt_nano_model(model_path):
    model = GPT(GPTConfig(vocab_size=50304))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE)['model'])
    model.eval()
    model.to(DEVICE)
    return model


def load_gpt_2_model():
    model = GPT2LMHeadModel.from_pretrained("gpt2") # 124M
    model.eval()
    model.to(DEVICE)
    return model


def query_model(model, query, max_length=100, num_return_sequences = 1):
    # prep model input
    enc = tiktoken.get_encoding('gpt2')
    tokens = enc.encode(query)
    tokens = torch.tensor(tokens, dtype=torch.long) # (8,)
    tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) # (5, 8)
    x = tokens.to(DEVICE)

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
        print(">", decoded, "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    parser.add_argument("-m", "--max_length", type=int, default=100, help="the max number of tokens to generate")
    parser.add_argument("-n", "--num_return_sequences", type=int, default=1, help="the number of samples the model should generate")
    parser.add_argument("--model_path", type=str, help="your .pt model checkpoint path")
    args = parser.parse_args()
    DEVICE = args.device

    gpt_nano_model = load_gpt_nano_model(args.model_path)
    gpt_2_model = load_gpt_2_model()

    while True:
        query = input("\n\nQuery: ").strip()
        print("\ngpt-nano:")
        query_model(gpt_nano_model, query, max_length=args.max_length, num_return_sequences=args.num_return_sequences)
        print("gpt-2:")
        query_model(gpt_2_model, query, max_length=args.max_length, num_return_sequences=args.num_return_sequences)
