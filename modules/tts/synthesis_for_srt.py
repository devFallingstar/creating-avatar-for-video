# coding: utf-8
"""
Synthesis waveform from trained model.

usage: synthesis.py [options] <checkpoint> <text_list_file> <dst_dir>

options:
    --hparams=<parmas>                Hyper parameters [default: ].
    --preset=<json>                   Path of preset parameters (json).
    --checkpoint-seq2seq=<path>       Load seq2seq model from checkpoint path.
    --checkpoint-postnet=<path>       Load postnet model from checkpoint path.
    --file-name-suffix=<s>            File name suffix [default: ].
    --max-decoder-steps=<N>           Max decoder steps [default: 500].
    --replace_pronunciation_prob=<N>  Prob [default: 0.0].
    --speaker_id=<id>                 Speaker ID (for multi-speaker model).
    --output-html                     Output html for blog post.
    -h, --help               Show help message.
"""
from docopt import docopt

import sys
import os
import re
import pandas as pd
import numpy as np
from datetime import timedelta
from os.path import dirname, join, basename, splitext

import audio

import torch
import numpy as np
import nltk

# The deepvoice3 model
from deepvoice3_pytorch import frontend
from hparams import hparams, hparams_debug_string

from tqdm import tqdm

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
_frontend = None  # to be set later

frame = 23
p = re.compile('([0-9]+):([0-9]+):([0-9]+),([0-9]+)') 

def tts(model, text, p=0, speaker_id=None, fast=False):
    """Convert text to speech waveform given a deepvoice3 model.

    Args:
        text (str) : Input text to be synthesized
        p (float) : Replace word to pronounciation if p > 0. Default is 0.
    """
    model = model.to(device)
    model.eval()
    if fast:
        model.make_generation_fast_()

    sequence = np.array(_frontend.text_to_sequence(text, p=p))
    sequence = torch.from_numpy(sequence).unsqueeze(0).long().to(device)
    text_positions = torch.arange(1, sequence.size(-1) + 1).unsqueeze(0).long().to(device)
    speaker_ids = None if speaker_id is None else torch.LongTensor([speaker_id]).to(device)

    # Greedy decoding
    with torch.no_grad():
        mel_outputs, linear_outputs, alignments, done = model(
            sequence, text_positions=text_positions, speaker_ids=speaker_ids)

    linear_output = linear_outputs[0].cpu().data.numpy()
    spectrogram = audio._denormalize(linear_output)
    alignment = alignments[0].cpu().data.numpy()
    mel = mel_outputs[0].cpu().data.numpy()
    mel = audio._denormalize(mel)

    # Predicted audio signal
    waveform = audio.inv_spectrogram(linear_output.T)

    return waveform, alignment, spectrogram, mel


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def getTime(text):
    match = p.match(text)
    return float(timedelta(hours=int(match[1]), minutes=int(match[2]), seconds=int(match[3]), milliseconds=int(match[4])).total_seconds())

def load_srt(srt_path, bounding_path):

    bounding_data = pd.read_csv(bounding_path);
    result = []
    #srt_re = re.compile('''(\d+)\n([0-9]+:[0-9]+:[0-9]+,[0-9]+) --> ([0-9]+:[0-9]+:[0-9]+,[0-9]+)\n([0-9a-zA-Z ?!'".,\-\n]+)\n\n''')
    srt_re = re.compile('''(\d+)\n([0-9]+:[0-9]+:[0-9]+,[0-9]+) --> ([0-9]+:[0-9]+:[0-9]+,[0-9]+)\n([0-9a-zA-Z ?!'".,\-\n]+)(\n\n|$)''')
    with open(srt_path, "r") as f:
        atext = f.read();
        for i in srt_re.finditer(atext):
            idx = int(i.group(1))
            start = int(getTime(i.group(2)) * frame)
            end = int(getTime(i.group(3)) * frame)
            
            inputtext = i.group(4)
            texts = []
            for i in inputtext.split('\n'):
                if (len(inputtext) == 0): continue
                if (i.find("- ") == 0):
                    texts.append(i[2:])
                    inputtext = inputtext[len(i):]
                else:
                    inputtext = inputtext.replace('\n', ' ')
                    inputtext = inputtext.replace('  ', ' ')
                    texts.append(inputtext)
                    inputtext = inputtext[len(inputtext):]

            for i in range(0,len(texts)):
                weight = (i + 1) / (len(texts) + 1)
                speaker_id = getSpeaker(start, end, weight, bounding_data)
                
                if (speaker_id==-1):
                    speaker_id=0;
                result.append([idx, start, end, speaker_id, texts[i]])
                print(speaker_id, start,end, texts[i])
                    
                
    return result
                


def getSpeaker(start, end, weight, data):
    # 정해진 프레임 내의 데이터만 고려
    data=data[(data["frame_number"]>=start) & (data["frame_number"]<=end)]
    
    score = {}
    for i, row in data.iterrows():
        if (row['ID']) == 'Unknown': continue
        faceId = int(row['ID'])
        if (faceId == 0): continue
            
        LT = re.match('''\((\d+),[ ](\d+)\)''', row['LT'])
        BR = re.match('''\((\d+),[ ](\d+)\)''', row['BR'])
        box_size = abs(int(LT.group(1)) - int(BR.group(1)) * int(LT.group(2)) - int(BR.group(2)))
        
        # 해당 프레임의 바운딩 박스 크기만큼 점수를 높임
        if (faceId not in score): score[faceId] = 0
        score[faceId] += box_size
        
    if (len(score) == 0): return -1;
    return max(score.keys(),key=(lambda k: score[k]))

if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    text_list_file_path = args["<text_list_file>"]
    dst_dir = args["<dst_dir>"]
    checkpoint_seq2seq_path = args["--checkpoint-seq2seq"]
    checkpoint_postnet_path = args["--checkpoint-postnet"]
    max_decoder_steps = int(args["--max-decoder-steps"])
    file_name_suffix = args["--file-name-suffix"]
    replace_pronunciation_prob = float(args["--replace_pronunciation_prob"])
    output_html = args["--output-html"]
    preset = args["--preset"]

    # Load preset if specified
    if preset is not None:
        with open(preset) as f:
            hparams.parse_json(f.read())
    # Override hyper parameters
    hparams.parse(args["--hparams"])
    assert hparams.name == "deepvoice3"

    _frontend = getattr(frontend, hparams.frontend)
    import train
    train._frontend = _frontend
    from train import plot_alignment, build_model

    # Model
    model = build_model()

    # Load checkpoints separately
    if checkpoint_postnet_path is not None and checkpoint_seq2seq_path is not None:
        checkpoint = _load(checkpoint_seq2seq_path)
        model.seq2seq.load_state_dict(checkpoint["state_dict"])
        checkpoint = _load(checkpoint_postnet_path)
        model.postnet.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_seq2seq_path))[0]
    else:
        checkpoint = _load(checkpoint_path)
        model.load_state_dict(checkpoint["state_dict"])
        checkpoint_name = splitext(basename(checkpoint_path))[0]

    model.seq2seq.decoder.max_decoder_steps = max_decoder_steps

    os.makedirs(dst_dir, exist_ok=True)

    task = load_srt('part.srt', 'trailer1.csv')
    idx = 0
    for i in task:
        speaker_id=i[3]
        text = i[4]
       
        words = nltk.word_tokenize(text)
        file_name = "{} speaker_{} {}-{}".format(idx, speaker_id, i[1], i[2])
        waveform, alignment, _, _ = tts(
                model, text, p=replace_pronunciation_prob, speaker_id=speaker_id, fast=True)
        dst_wav_path = join(dst_dir, "{}.wav".format(file_name))
        dst_alignment_path = join(
                dst_dir, "{}_alignment.png".format(file_name))
        plot_alignment(alignment.T, dst_alignment_path,
                           info="{}, {}".format(hparams.builder, basename(checkpoint_path)))
        audio.save_wav(waveform, dst_wav_path)
        name = splitext(basename(text_list_file_path))[0]
        print(idx, ": {}\n ({} chars, {} words)".format(text, len(text), len(words)))
        idx += 1


    print("Finished! Check out {} for generated audio samples.".format(dst_dir))
    sys.exit(0)
