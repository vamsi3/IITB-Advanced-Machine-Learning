import argparse
import time
import torch
from data import create_fields, nopeak_mask
import torch.nn.functional as F
import pdb
import dill as pickle
import argparse
from model import get_model
import re
import math

# BEAM SEARCH ------------------------------------------------------------------

def init_vars(src, model, SRC, TRG, opt):
    
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    
    outputs = torch.LongTensor([[init_tok]]).to(opt.device)
    
    trg_mask = nopeak_mask(1, opt)
    
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    
    outputs = torch.zeros(opt.k, opt.max_len).long().to(opt.device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1)).to(opt.device)
    e_outputs[:, :] = e_output[0]
    
    return outputs, e_outputs, log_scores

def k_best_outputs(outputs, out, log_scores, i, k):
    
    probs, ix = out[:, -1].data.topk(k)
    log_probs = torch.Tensor([math.log(p) for p in probs.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probs, k_ix = log_probs.view(-1).topk(k)
    
    row = k_ix // k
    col = k_ix % k

    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)
    
    return outputs, log_scores

def beam_search(src, model, SRC, TRG, opt):
    

    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)
    eos_tok = TRG.vocab.stoi['<eos>']
    unk_tok = TRG.vocab.stoi['<unk>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    ind = None
    for i in range(2, opt.max_len):
    
        trg_mask = nopeak_mask(i, opt)

        out = model.out(model.decoder(outputs[:,:i],
        e_outputs, src_mask, trg_mask))

        out = F.softmax(out, dim=-1)
    
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, i, opt.k)
        
        if (outputs==eos_tok).nonzero().size(0) == opt.k:
            alpha = 0.7
            div = 1/((outputs==eos_tok).nonzero()[:,1].type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    
    if ind is None:
        ind = 0
    length = ((outputs[ind]==eos_tok) + (outputs[ind]==unk_tok)).nonzero()
    length = length[0] if length.numel() != 0 else outputs.size(1)
    return ' '.join([TRG.vocab.itos[tok] for tok in outputs[ind][1:length]])


# INFERENCE --------------------------------------------------------------------

def translate_sentence(sentence, model, opt, SRC, TRG):
    model.eval()
    indexed = []
    sentence = SRC.preprocess(sentence)
    for tok in sentence:
        indexed.append(SRC.vocab.stoi[tok])
    sentence = torch.LongTensor([indexed]).to(opt.device)    
    sentence = beam_search(sentence, model, SRC, TRG, opt)
    return sentence

def translate(opt, model, SRC, TRG):
    sentence = opt.text.lower()
    sentence = re.sub(r"(?<=\s)'([^']*)'(?=(\s|$))", r'UNK_START \1 UNK_END', sentence)
    sentence = re.sub(r'(?<=\s)"([^"]*)"(?=(\s|$))', r'UNK_START \1 UNK_END', sentence)
    sentence = re.sub(r'\s+', r' ', sentence)
    translated = translate_sentence(sentence, model, opt, SRC, TRG)
    translated = re.sub(r'\s+flag_suffix\s+', r'', translated)
    return translated.strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', required=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-max_len', type=int, default=150)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    
    opt = parser.parse_args()

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
    assert opt.k > 0
    assert opt.max_len > 10

    SRC, TRG = create_fields(opt)
    model = get_model(opt, len(SRC.vocab), len(TRG.vocab))
    
    while True:
        opt.text =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
        if opt.text=="q":
            break
        if opt.text=='f':
            fpath =input("Enter a sentence to translate (type 'f' to load from file, or 'q' to quit):\n")
            try:
                opt.text = ' '.join(open(opt.text, encoding='utf-8').read().split('\n'))
            except:
                print("error opening or reading text file")
                continue
        phrase = translate(opt, model, SRC, TRG)
        print('> '+ phrase + '\n')

if __name__ == '__main__':
    main()
