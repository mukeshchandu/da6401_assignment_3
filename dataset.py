import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter
_spacy_de = None
_spacy_en = None
def get_tokenizers():
    global _spacy_de, _spacy_en
    if _spacy_de is None or _spacy_en is None:
        import spacy, subprocess, sys
        def load_model(name):
            try:
                return spacy.load(name)
            except OSError:
                subprocess.run([sys.executable, "-m", "spacy", "download", name],
                               check=True, capture_output=True)
                return spacy.load(name)
        _spacy_de = load_model("de_core_news_sm")
        _spacy_en = load_model("en_core_web_sm")
    return _spacy_de, _spacy_en
def tokenize_de(text):
    spacy_de, _ = get_tokenizers()
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]
def tokenize_en(text):
    _, spacy_en = get_tokenizers()
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]
class Vocabulary:
    def __init__(self):
        self.itos = {0: "<unk>", 1: "<pad>", 2: "<sos>", 3: "<eos>"}
        self.stoi = {v: k for k, v in self.itos.items()}
    def __len__(self):
        return len(self.itos)
    def build(self, sentences, tokenize_fn, min_freq=2):
        counter = Counter()
        for sent in sentences:
            counter.update(tokenize_fn(sent))
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.stoi:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx
    def encode(self, tokens):
        return [self.stoi.get(t, 0) for t in tokens]
    def lookup_token(self, idx):
        return self.itos.get(idx, "<unk>")
class Multi30kDataset(Dataset):
    def __init__(self, split="train", src_vocab=None, tgt_vocab=None):
        self.split = split
        raw = load_dataset("bentrevett/multi30k")
        self.data = raw[split]
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
    @staticmethod
    def build_vocabs(min_freq=2):
        raw = load_dataset("bentrevett/multi30k")
        train_data = raw["train"]
        src_vocab = Vocabulary()
        tgt_vocab = Vocabulary()
        src_sents = [ex["de"] for ex in train_data]
        tgt_sents = [ex["en"] for ex in train_data]
        src_vocab.build(src_sents, tokenize_de, min_freq=min_freq)
        tgt_vocab.build(tgt_sents, tokenize_en, min_freq=min_freq)
        return src_vocab, tgt_vocab
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        src_tokens = tokenize_de(self.data[idx]["de"])
        tgt_tokens = tokenize_en(self.data[idx]["en"])
        src_ids = [self.src_vocab.stoi["<sos>"]] + \
                  self.src_vocab.encode(src_tokens) + \
                  [self.src_vocab.stoi["<eos>"]]
        tgt_ids = [self.tgt_vocab.stoi["<sos>"]] + \
                  self.tgt_vocab.encode(tgt_tokens) + \
                  [self.tgt_vocab.stoi["<eos>"]]
        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)
def collate_fn(batch, src_pad=1, tgt_pad=1):
    src_batch, tgt_batch = zip(*batch)
    src_padded = torch.nn.utils.rnn.pad_sequence(
        src_batch, batch_first=True, padding_value=src_pad)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(
        tgt_batch, batch_first=True, padding_value=tgt_pad)
    return src_padded, tgt_padded
def get_dataloaders(batch_size=128):
    src_vocab, tgt_vocab = Multi30kDataset.build_vocabs()
    train_ds = Multi30kDataset("train",      src_vocab, tgt_vocab)
    val_ds   = Multi30kDataset("validation", src_vocab, tgt_vocab)
    test_ds  = Multi30kDataset("test",       src_vocab, tgt_vocab)
    from functools import partial
    _collate = partial(collate_fn,
                       src_pad=src_vocab.stoi["<pad>"],
                       tgt_pad=tgt_vocab.stoi["<pad>"])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  collate_fn=_collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, collate_fn=_collate)
    test_loader  = DataLoader(test_ds,  batch_size=1,
                              shuffle=False, collate_fn=_collate)

    return train_loader, val_loader, test_loader, src_vocab, tgt_vocab
