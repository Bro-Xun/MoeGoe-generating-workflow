import sys, re
from torch import no_grad, LongTensor
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from mel_processing import spectrogram_torch

from scipy.io.wavfile import write

import yaml
import time
import os

def ex_print(text,escape=False):
    if escape:
        print(text.encode('unicode_escape').decode())
    else:
        print(text)

def get_text(text, hps, cleaned=False):
    if cleaned:
        text_norm = text_to_sequence(text, hps.symbols, [])
    else:
        text_norm = text_to_sequence(text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm

def ask_if_continue():
    while True:
        answer = input('Continue? (y/n): ')
        if answer == 'y':
            break
        elif answer == 'n':
            sys.exit(0)

def print_speakers(speakers,escape=False):
    print('ID\tSpeaker')
    for id, name in enumerate(speakers):
        ex_print(str(id) + '\t' + name,escape)

def get_speaker_id(message):
    speaker_id = input(message)
    try:
        speaker_id = int(speaker_id)
    except:
        print(str(speaker_id) + ' is not a valid ID!')
        sys.exit(1)
    return speaker_id

def get_label_value(text, label, default, warning_name='value'):
    value=re.search(rf'\[{label}=(.+?)\]',text)
    if value:
        try:
            text=re.sub(rf'\[{label}=(.+?)\]','',text,1)
            value=float(value.group(1))
        except:
            print(f'Invalid {warning_name}!')
            sys.exit(1)
    else:
        value=default
    return value, text

def get_label(text,label):
    if f'[{label}]' in text:
        return True,text.replace(f'[{label}]','')
    else:
        return False,text

if __name__ == '__main__':
    config_file = open("config.yml",'r',encoding="utf-8")
    config_file_data = config_file.read()
    config_file.close()
    config_data = yaml.load(config_file_data,Loader=yaml.FullLoader)

    text_list_file = open(config_data['argument_path'],'r',encoding="utf-8")
    text_list_data = text_list_file.read()
    text_list_file.close()
    text_list_data = text_list_data.replace('\n\n','\n')
    text_list_data = re.sub('#.*?\n','',text_list_data)
    text_list = text_list_data.split('\n')

    escape=False #non-stop workflow
    
    model = config_data['model_path']
    config = config_data['config_path']
    
    hps_ms = utils.get_hparams_from_file(config)
    n_speakers = hps_ms.data.n_speakers if 'n_speakers' in hps_ms.data.keys() else 0
    n_symbols = len(hps_ms.symbols) if 'symbols' in hps_ms.keys() else 0
    speakers = hps_ms.speakers if 'speakers' in hps_ms.keys() else ['0']
    use_f0 = hps_ms.data.use_f0 if 'use_f0' in hps_ms.data.keys() else False

    net_g_ms = SynthesizerTrn(
        n_symbols,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval()
    utils.load_checkpoint(model, net_g_ms)
    
    if n_symbols!=0:
        while True:
            choice = config_data['method']
            if choice == 't':
                numb = 1
                text = 'init'
                lang = config_data['argument_lang']
                for i in text_list:
                    model_num = 1
                    ends = i[-2::]
                    print(ends)
                    if not len(ends) == 2:
                        text = lang + i + lang
                    else:
                        if ends[1] in ['1','2','3','4','5','6','7','8','9','0']:
                            model_num = int(ends[1])
                            text = lang + i[:-2:] + lang
                        else:
                            text = lang + i + lang

                    
                    if text=='[ADVANCED]':
                        text = input('Raw text:')
                        print('Cleaned text is:')
                        ex_print(_clean_text(text, hps_ms.data.text_cleaners),escape)
                        continue
                    
                    length_scale,text=get_label_value(text,'LENGTH',1,'length scale')
                    noise_scale,text=get_label_value(text,'NOISE',0.667,'noise scale')
                    noise_scale_w,text=get_label_value(text,'NOISEW',0.8,'deviation of noise')
                    cleaned,text=get_label(text,'CLEANED')

                    stn_tst = get_text(text, hps_ms, cleaned=cleaned)
                    
                    #print_speakers(speakers,escape)
                    speaker_id = model_num - 1
                    out_path = config_data['output_path'] + str(numb) + '.wav'

                    with no_grad():
                        x_tst = stn_tst.unsqueeze(0)
                        x_tst_lengths = LongTensor([stn_tst.size(0)])
                        sid = LongTensor([speaker_id])
                        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0,0].data.cpu().float().numpy()
                    write(out_path, hps_ms.data.sampling_rate, audio)
                    
                    print('Successfully saved!')
                    #ask_if_continue()
                    time.sleep(config_data['time_gap'])
                    print('succeed writting into {}.wav'.format(numb))
                    numb += 1

                    break
                
                    
            elif choice == 'v':
                audio_dir = config_data['audio_dir']
                dir_audio_dir = os.listdir(audio_dir)
                audios = re.findall('(.*?).wav','\n'.join(audio_dir))
                for i in audios:
                    audio_path = audio_dir + i + '.wav'
                    #print_speakers(speakers)
                    audio = utils.load_audio_to_torch(audio_path, hps_ms.data.sampling_rate)

                    originnal_id = config_data['origin']
                    target_id = config_data['target']
                    out_path = config_data['output_path']

                    y = audio.unsqueeze(0)

                    spec = spectrogram_torch(y, hps_ms.data.filter_length,
                        hps_ms.data.sampling_rate, hps_ms.data.hop_length, hps_ms.data.win_length,
                        center=False)
                    spec_lengths = LongTensor([spec.size(-1)])
                    sid_src = LongTensor([originnal_id])

                    with no_grad():
                        sid_tgt = LongTensor([target_id])
                        audio = net_g_ms.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0,0].data.cpu().float().numpy()
                    write(out_path, hps_ms.data.sampling_rate, audio)
                            
                    print('Successfully saved!')
                    time.sleep(config_data['time_gap'])
                    print('succeed writting into {}.wav'.format(i))

                    break               

    else:
        model = input('Path of a hubert-soft model: ')
        from hubert_model import hubert_soft
        hubert = hubert_soft(model)

        while True:
            audio_path = input('Path of an audio file to convert:\n')
            print_speakers(speakers)
            
            import librosa
            if use_f0:
                audio, sampling_rate = librosa.load(audio_path, sr=hps_ms.data.sampling_rate, mono=True)
                audio16000 = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
            else:
                audio16000, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
            
            target_id = get_speaker_id('Target speaker ID: ')
            out_path = input('Path to save: ')
            length_scale,out_path=get_label_value(out_path,'LENGTH',1,'length scale')
            noise_scale,out_path=get_label_value(out_path,'NOISE',0.1,'noise scale')
            noise_scale_w,out_path=get_label_value(out_path,'NOISEW',0.1,'deviation of noise')
                
            from torch import inference_mode, FloatTensor
            import numpy as np
            with inference_mode():
                units = hubert.units(FloatTensor(audio16000).unsqueeze(0).unsqueeze(0)).squeeze(0).numpy()
                if use_f0:
                    f0_scale,out_path = get_label_value(out_path,'F0',1,'f0 scale')
                    f0 = librosa.pyin(audio, sr=sampling_rate,
                        fmin=librosa.note_to_hz('C0'),
                        fmax=librosa.note_to_hz('C7'),
                        frame_length=1780)[0]
                    target_length = len(units[:, 0])
                    f0 = np.nan_to_num(np.interp(np.arange(0, len(f0)*target_length, len(f0))/target_length,
                        np.arange(0, len(f0)), f0)) * f0_scale
                    units[:, 0] = f0 / 10
            
            stn_tst = FloatTensor(units)
            with no_grad():
                x_tst = stn_tst.unsqueeze(0)
                x_tst_lengths = LongTensor([stn_tst.size(0)])
                sid = LongTensor([target_id])
                audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w, length_scale=length_scale)[0][0, 0].data.float().numpy()
            write(out_path, hps_ms.data.sampling_rate, audio)
            
            print('Successfully saved!')
            ask_if_continue()
