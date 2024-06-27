import multiprocessing as mp
import os
import numpy as np
import tiktoken
import wikipediaapi
from tqdm import tqdm  # pip install tqdm
import re



def print_categorymembers(categorymembers, f, level=0, max_level=1):
        for c in categorymembers.values():
            if c.ns == wikipediaapi.Namespace.CATEGORY:
                if level < max_level:
                    print_categorymembers(c.categorymembers, f, level=level + 1, max_level=max_level)
            else:
                 p_wiki = wiki_wiki.page(c.title)
                 print(p_wiki.title)
                 f.write(clean_text(c.title + ": " + ' '.join(p_wiki.text.split())) + "\n")


def clean_text(text):
    """
    Filters a string for non-ASCII characters, and removes "== References ==" and "== External Links ==" strings
    """
    filtered_text = ""
    for char in text:
        if ord(char) < 128:  # Check if character is within ASCII range (0-127)
            filtered_text += char

    pattern = r"== References =="  # Regular expression pattern
    filtered_text = re.sub(pattern, "", filtered_text, flags=re.MULTILINE)
    pattern = r"== External links =="
    filtered_text = re.sub(pattern, "", filtered_text, flags=re.MULTILINE)
    return filtered_text


# init the tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>'] # end of text token
def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(enc.encode_ordinary(doc))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16


def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)
            

if __name__ == "__main__":
    local_dir = "civil_wiki"
    shard_size = int(5e5) # 100M tokens per shard, total of 100 shards

    # create the cache the local directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "civil_wiki.txt")

    civil_wiki_txt_exists = os.path.exists(RAW_DATA_PATH)
    if not civil_wiki_txt_exists:
        # download related civil engineering articles
        wiki_wiki = wikipediaapi.Wikipedia('gpt-nano (joshuakgao@gmail.com)', 'en', extract_format=wikipediaapi.ExtractFormat.WIKI)
        cat = wiki_wiki.page("Category: Civil engineering")
        print("Category members: Category: Civil engineering")
        f = open(RAW_DATA_PATH, 'a')
        print_categorymembers(cat.categorymembers, f, max_level=1)

    f = open(RAW_DATA_PATH, 'r')
    civil_wiki = f.readlines()

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        for tokens in pool.imap(tokenize, civil_wiki, chunksize=16):

            # is there enough space in the current shard for the new tokens?
            if token_count + len(tokens) < shard_size:
                # simply append tokens to current shard
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                # update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                progress_bar.update(len(tokens))
            else:
                # write the current shard and start a new one
                split = "val" if shard_index == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"{local_dir}_{split}_{shard_index:06d}")
                # split the document into whatever fits in this shard; the remainder goes to next one
                remainder = shard_size - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
                write_datafile(filename, all_tokens_np)
                shard_index += 1
                progress_bar = None
                # populate the next shard with the leftovers of the current doc
                all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
                token_count = len(tokens)-remainder

        # write any remaining tokens as the last shard
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{local_dir}_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])


