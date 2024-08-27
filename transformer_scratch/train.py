import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from dataset import BilingualDataset, casual_mask

from config import get_weights_file_path, get_config, latest_weights_file_path

from model import BuildTransformer

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pathlib import Path

import warnings

def greedy_decode(model, src, src_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    # get the translation of the source sentence starting with [SOS]

    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.ENCODE(src, src_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(src).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        decoder_mask = casual_mask(decoder_input.size(1)).type_as(src).to(device)
        out = model.DECODE(decoder_input, encoder_output, src_mask, decoder_mask)

        prob = model.project(out[:, -1])

        _, next_token = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1, 1).fill_(next_token.item()).type_as(src).to(device)], dim=1)
        
        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)     

def run_validation(model, val_ds, src_tokenizer, tgt_tokenizer, max_len, device, print_msg, global_state, writer, run_examples=2):
    model.eval()
    count = 0

    console_width = 80
    
    with torch.no_grad():
        for batch in val_ds: # batch size is (1, seq_len, d_model)
            count += 1
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            assert encoder_input.size(0) == 1 # batch size is 1

            model_output = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, max_len, device)

            sorce_sentence = batch['src_text'][0]
            expected_sentence = batch['tgt_text'][0]
            predicted_sentence = tgt_tokenizer.decode(model_output.detach().cpu().numpy())

            # print it to the console
            print_msg('-'*console_width)
            print_msg(f"Source: {sorce_sentence}")
            print_msg(f"Expected: {expected_sentence}")
            print_msg(f"Predicted: {predicted_sentence}")

            if count >= run_examples:
                break

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

    # get max length of source and target sentence
    max_len_src = 0
    max_len_tgt = 0
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['src_lang']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['tgt_lang']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')

    # buidl data loader
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    validation_dl = DataLoader(validation_ds, batch_size=1, shuffle=True)

    return train_dl, validation_dl, tokenizer_src, tokenizer_tgt

def get_model(config, src_vacab_size, tgt_vacab_size):
    # build model
    model = BuildTransformer(src_vacab_size, tgt_vacab_size, config['seq_len'], config['seq_len'], config['d_model'])
    return model

def train_model(config):
    # device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using the device: {device}")
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True) # create model directory
    train_loader, validation_loader, tokenizer_src, tokenizer_tgt = get_ds(config)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)

    # tensorboard 
    writer = SummaryWriter(config['experiment_name']) # tensorboard writer

    # loss and optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    initial_epoch = 0
    global_step = 0

    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')
    # assert(0)

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        model.train()
        batch_iter = tqdm(train_loader, desc=f'processing epoch {epoch:02d}')
        for batch in batch_iter:
            
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)

            encoder_output = model.ENCODE(encoder_input, encoder_mask)# outshape: (batch_size, seq_len, d_model)
            decoder_output = model.DECODE(decoder_input, encoder_output, encoder_mask, decoder_mask) # outshape: (batch_size, seq_len, d_model)
            project_output = model.project(decoder_output) # outshape: (batch_size, seq_len, vocab_size)
            label = batch['label'].to(device) # outshape: (batch_size, seq_len)

            # the shape of the two parameters: (batch_size*seq_len, vocab_size), (batch_size*seq_len)
            loss = loss_fn(project_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iter.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backward and optimize
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation
        run_validation(model, validation_loader, tokenizer_src, tokenizer_tgt, config['seq_len'], device, lambda msg: batch_iter.write(msg), global_step, writer)

        if True:
            model_filename = get_weights_file_path(config, f"{epoch:02d}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)
        


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)

