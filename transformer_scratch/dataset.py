from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len)->None:
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        # define sos and eos and pad tokens
        self.sos_token = torch.tensor(tokenizer_tgt.token_to_id("[SOS]"), dtype=torch.int64) # shape: tensor:(1)
        self.eos_token = torch.tensor(tokenizer_tgt.token_to_id("[EOS]"), dtype=torch.int64)
        self.pad_token = torch.tensor(tokenizer_tgt.token_to_id("[PAD]"), dtype=torch.int64)

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        # 获取一组能够被模型处理的 tensor 数据（同时还获取了一些其他数据如 mask and text）
        src_tgt_pair = self.ds[idx]
        src_text = src_tgt_pair['translation'][self.src_lang]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        # text to token
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # compute needed paddings
        enc_padding = self.seq_len - len(enc_input_tokens) - 2 # 2 for sos and eos
        dec_padding = self.seq_len - len(dec_input_tokens) - 1 # 1 for eos
        if enc_padding < 0 or dec_padding < 0:
            return ValueError("Sequence length is too Long")

        # tokens to tensor (need add paddings)
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64), # shape: tensor:(seq_len)
                self.eos_token,
                torch.tensor([self.pad_token]*enc_padding, dtype=torch.int64) # shape: tensor:(enc_padding)
            ]
        )
        # decoderinput has only sos, while label has only eos
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64), # shape: tensor:(seq_len)
                torch.tensor([self.pad_token]*dec_padding, dtype=torch.int64) # shape: tensor:(dec_padding)
            ]
        )
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64), # shape: tensor:(seq_len)
                self.eos_token,
                torch.tensor([self.pad_token]*dec_padding, dtype=torch.int64) # shape: tensor:(dec_padding)
            ]
        )
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        # return the tensors and masks and texts

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # shape: tensor:(1, 1, seq_len)
            "decoder_mask": casual_mask(self.seq_len) & (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # shape: tensor:(1, 1, seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    

def casual_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0