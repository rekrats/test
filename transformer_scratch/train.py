import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask

from model import BuildTransformer

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config):
    # Load the dataset
    ds_raw = load_dataset('opus_books', f'{config["src_lang"]}-{config["tgt_lang"]}', split='train')

    # build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['src_lang'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['tgt_lang'])

    # build training set and validation set
    train_size = int(0.9 * len(ds_raw))
    validation_size = len(ds_raw) - train_size

    train_ds_raw, validation_ds_raw = torch.utils.data.random_split(ds_raw, [train_size, validation_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])
    validation_ds = BilingualDataset(validation_ds_raw, tokenizer_src, tokenizer_tgt, config['src_lang'], config['tgt_lang'], config['seq_len'])

    # count maxlen
    src_max_len = 0
    tgt_max_len = 0
    for item in train_ds:
        src_ids = tokenizer_src.encode(item['encoder_input'][config['src_lang']]).ids
        tgt_ids = tokenizer_tgt.encode(item['decoder_input'][config['tgt_lang']]).ids
        src_max_len = max(src_max_len, len(src_ids))
        tgt_max_len = max(tgt_max_len, len(tgt_ids))
    print(f"src_max_len: {src_max_len}, tgt_max_len: {tgt_max_len}")

    # buidl data loader
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=1, shuffle=True)

    return train_dl, validation_dl, tokenizer_src, tokenizer_tgt

def get_model(config, src_vacab_size, tgt_vacab_size):
    # build model
    model = BuildTransformer(src_vacab_size, tgt_vacab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model