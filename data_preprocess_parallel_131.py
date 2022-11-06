# -*- coding: cp949 -*-
from multiprocessing import Pool, Process, Manager
import multiprocessing
import os, glob, time
from glob import glob
import os, shutil
import soundfile as sf
import argparse
import numpy as np
import librosa
import soundfile as sf
import json
from joblib import Parallel, delayed
import tqdm
import multiprocessing
import re
import pandas as pd
import json
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

num_CPU = multiprocessing.cpu_count() 
print('num_CPU', num_CPU)



def label_directory(path1, path2, train_or_val):
    wav_exts = ['.wav', '.WAV', '.pcm', '.PCM']
    #txt_exts = ['.txt', '.TXT']
    json_exts = ['.json', '.JSON']
    
    wavList_tmp = sorted(glob(path1+'/**', recursive=True))
    jsonList_tmp = sorted(glob(path2+'/**', recursive=True))    
    
    wavList = list()
    jsonList = list()
    
    wavList = [x for x in wavList_tmp if os.path.splitext(x)[1] in wav_exts]
    jsonList = [x for x in jsonList_tmp if os.path.splitext(x)[1] in json_exts]
    print(f'wav len {len(wavList)}, json len {len(jsonList)}')
    exit()
    #Parallel_wavList = Manager().list()
    #Parallel_txtList = Manager().list()
    Parallel_wavtxtDict = Manager().dict()
    
    Parallel_NoPaired_List = Manager().list()
    Parallel_wavReadError_List = Manager().list()
    Parallel_wavNoSignal_List = Manager().list()
    Parallel_wavStereo_List = Manager().list()
    Parallel_jsonProb_List = Manager().list()
    Parallel_wavShort_List = Manager().list()
    Parallel_wavDuration = Manager().list()
    
    pool = Pool(num_CPU)
    
    pool.starmap(label_file, [(wavpath, jsonpath, Parallel_wavtxtDict, Parallel_NoPaired_List, Parallel_wavReadError_List, Parallel_wavNoSignal_List, Parallel_wavStereo_List, Parallel_wavDuration, Parallel_jsonProb_List, Parallel_wavShort_List) for wavpath, jsonpath in zip(wavList, jsonList)])
    pool.close()
    pool.join()
    
    print(f'Dataset: {args.dataset_name}, Parallel_wavtxtDict: {len(Parallel_wavtxtDict)}, NoPaired_List: {len(Parallel_NoPaired_List)} Parallel_wavReadError_List: {len(Parallel_wavReadError_List)}, Parallel_wavNoSignal_List: {len(Parallel_wavNoSignal_List)}, Parallel_wavStereo_List: {len(Parallel_wavStereo_List)}, Parallel_jsonProb_List: {len(Parallel_jsonProb_List)}, Parallel_wavShort_List: {len(Parallel_wavShort_List)}')
    
    wavList = list(Parallel_wavtxtDict.keys())
    txtList = list(Parallel_wavtxtDict.values())
    
    df = pd.DataFrame([x for x in zip(wavList, txtList)])
    df.columns = ['file_id', 'sentence']
    
    df.to_csv(os.path.join(args.asr_save_dir, train_or_val, 'label.csv'))
    
    with open(os.path.join(args.asr_save_dir, train_or_val, 'no_paired.csv'), 'w') as f:
        for data in Parallel_NoPaired_List:
            f.write(data+'\n')
    
    with open(os.path.join(args.asr_save_dir, train_or_val, 'can_not_read.csv'), 'w') as f:
        for data in Parallel_wavReadError_List:
            f.write(data+'\n')
    
    with open(os.path.join(args.asr_save_dir, train_or_val, 'no_signal.csv'), 'w') as f:
        for data in Parallel_wavNoSignal_List:
            f.write(data+'\n')
    
    with open(os.path.join(args.asr_save_dir, train_or_val, 'stereo_channel.csv'), 'w') as f:
        for data in Parallel_wavStereo_List:
            f.write(data+'\n')
    
    with open(os.path.join(args.asr_save_dir, train_or_val, 'can_not_read_json.csv'), 'w') as f:
        for data in Parallel_jsonProb_List:
            f.write(data+'\n')
    
    with open(os.path.join(args.asr_save_dir, train_or_val, 'under_2secs.csv'), 'w') as f:
        for data in Parallel_wavShort_List:
            f.write(data+'\n')
    
    with open(os.path.join(args.asr_save_dir, train_or_val, 'data_duration.txt'), 'w') as f:
        total_time = sum(Parallel_wavDuration)
        for_write = 'total_time={}'.format(total_time)
        f.write(for_write+'\n')
        
        for_write_min = 'total_time={}'.format(total_time/60)
        f.write(for_write_min+'\n')
        
        for_write_hours = 'total_time={}'.format(total_time/3600)
        f.write(for_write_hours+'\n')
            
    
    '''
    with open(os.path.join(asr_dir, 'wrong_file.csv'), 'w') as f:
        for data in problem_list:
            f.write(data+'\n')
    '''
    


def label_file(wavpath, jsonpath, Parallel_wavtxtDict, Parallel_NoPaired_List, Parallel_wavReadError_List, Parallel_wavNoSignal_List, Parallel_wavStereo_List, Parallel_wavDuration, Parallel_jsonProb_List, Parallel_wavShort_List):    
    #asr_dir = train_asr_dir
    #ssl_dir = train_ssl_dir
    error_check = 0
    
    label_id = list()
    sentence_infor = list()
    problem_list = list()
    
    
    
    _dir, txt_id = os.path.split(jsonpath) #/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation/label/5. 한국문화II , ~*.json
    _dir_dir, dir = os.path.split(_dir) # /NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation/label, 5. 한국문화II
    _dir_dir_dir, label_level = os.path.split(_dir_dir) # /NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation, label
    _dir_dir_dir_dir, train_or_valid = os.path.split(_dir_dir_dir) #/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/159_number_speech/01.데이터, 2.Validation
    
    
    #_dir_dir_dir_dir_dir, train_or_valid = os.path.split(_dir_dir_dir_dir) #/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/159_number_speech/01.데이터, 2.Validaiton
    
    #_dir_dir_dir_dir_dir_dir, label_level = os.path.split(_dir_dir_dir_dir_dir) #/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/186_call_center/01.데이터/2.Validation, label
    
    #_dir_dir_dir_dir_dir_dir_dir, train_or_valid = os.path.split(_dir_dir_dir_dir_dir_dir) #/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/186_call_center/01.데이터, 2.Validation
    
    asr_save_dir = os.path.join(args.asr_save_dir, train_or_valid)
    ssl_save_dir = os.path.join(args.ssl_save_dir, train_or_valid)
    #print(f'asr_save_dir {asr_save_dir}, ssl_save_dir {ssl_save_dir}')
    if label_level == 'label':
        source_level = 'source'
        
    txt_extension = txt_id.split('.')[-1]
    _, wav_id = os.path.split(wavpath)
    wav_extension = wav_id.split('.')[-1]
    
    try:
        with open(jsonpath, 'r') as json_file:
            json_data = json.load(json_file)                
    except:
        print(f'This file {jsonpath} has error, can not load json file')
        Parallel_jsonProb_List.append(jsonpath)
        error_check = 1
    
    if error_check != 1:
        txt1 = json_data['transcription']['ReadingLabelText']
        txt2 = json_data['transcription']['AnswerLabelText']
        txt = txt1 if len(txt1) > 1 else txt2
        #txt = json_data['inputText'][0]['orgtext']
        #txt = json_data['script']['scriptTN']        
        wav = os.path.join(_dir_dir_dir, source_level, dir, txt_id.replace('.'+txt_extension, '.'+wav_extension))
        #print('json {} txt {} wav {} check {}'.format(jsonpath, txt, wav, os.path.isfile(wav)))            
        if os.path.isfile(wav) == False:
            Parallel_NoPaired_List.append(jsonpath)
            error_check = 1
    
    if error_check != 1:        
        file_size = Path(wav).stat().st_size
        
        if file_size < 1000:
            print(f'This file {wav} has error, file_size is smaller than 1000')
            Parallel_wavNoSignal_List.append(wav)
            error_check = 1
        
        
        if error_check != 1:
            if wav_extension == 'wav':
                try:
                    based_sr = librosa.get_samplerate(wav)
                    
                except:
                    print(f'This file {wav} has error, can not get sample rate')
                    Parallel_wavReadError_List.append(wav)
                    error_check = 1
            
            elif wav_extension == 'pcm':
                based_sr = 16000
        
        if error_check != 1:
            
            if wav_extension == 'wav':                
                y, _ = librosa.load(wav, based_sr)
                wav_duration = librosa.get_duration(y, based_sr)
            
            elif wav_extension == 'pcm':                
                with open(wav, 'rb') as opened_pcm_file:
                    buf = opened_pcm_file.read()
                pcm_data = np.frombuffer(buf, dtype = 'int16')
                y = librosa.util.buf_to_float(pcm_data, 2)
                wav_duration = librosa.get_duration(y, based_sr)
                        
            if wav_duration < 2.0:
                print(f'This file {wav} has short length, under 2 seconds')
                Parallel_wavShort_List.append(wav)
                error_check = 1
            
            if len(y.shape) > 1:
                print(f'This file {wav} has error, streo-channel')
                Parallel_wavStereo_List,append(wav)
                error_check = 1
        
        #print('based_sr', based_sr)   
        if error_check != 1:
            if y.mean() == 0.0:
                print(f'This file {wav} has error, mean value is 0.0')
                Parallel_wavNoSignal_List.append(wav)
                error_check = 1
        
        if error_check != 1:
            asr_save_dir_tmp = os.path.join(asr_save_dir, dir)
            if not os.path.exists(asr_save_dir_tmp):
                os.makedirs(asr_save_dir_tmp, exist_ok=True)
            
            ssl_save_dir_tmp = os.path.join(ssl_save_dir, dir)
            if not os.path.exists(ssl_save_dir_tmp):
                os.makedirs(ssl_save_dir_tmp, exist_ok=True)
            
            if based_sr != 16000:                
                input_file = librosa.resample(y, based_sr, 16000)                    
                sf.write(os.path.join(asr_save_dir_tmp, os.path.split(wav)[-1]), 
                    input_file, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')            
                sf.write(os.path.join(ssl_save_dir_tmp, os.path.split(wav)[-1]), 
                    input_file, 16000, format='WAV', endian='LITTLE', subtype='PCM_16')
                Parallel_wavtxtDict[os.path.join(asr_save_dir_tmp, os.path.split(wav)[-1])] = txt
                #Parallel_wavList.append(os.path.join(asr_save_dir_tmp, os.path.split(wav)[-1]))
                #Parallel_txtList.append(txt)
                Parallel_wavDuration.append(wav_duration)
            
            else:            
                shutil.copy(wav, os.path.join(asr_save_dir_tmp, os.path.split(wav)[-1]))                    
                shutil.copy(wav, os.path.join(ssl_save_dir_tmp, os.path.split(wav)[-1]))
                Parallel_wavtxtDict[os.path.join(asr_save_dir_tmp, os.path.split(wav)[-1])] = txt
                #Parallel_wavList.append(os.path.join(asr_save_dir_tmp, os.path.split(wav)[-1]))
                #Parallel_txtList.append(txt)
                Parallel_wavDuration.append(wav_duration)

 




if __name__=="__main__":
    parser = argparse.ArgumentParser(description='copy')
    parser.add_argument('--start_dir_train_wav1', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/1.Training/source')
    #/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation/label/5. 한국문화II/
    #parser.add_argument('--start_dir_train_wav1', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/159_number_speech/01.데이터/1.Training/source/01.계좌번호')
    
    parser.add_argument('--start_dir_valid_wav1', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation/source')
    #parser.add_argument('--start_dir_valid_wav1', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation/source/5. 한국문화II')
    #parser.add_argument('--start_dir_valid_wav2', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/011_korean_child/01.데이터/2.Validation/source_free')
    
    parser.add_argument('--start_dir_train_txt1', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/1.Training/label')
    #parser.add_argument('--start_dir_train_txt2', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/011_korean_child/01.데이터/1.Training/label_free')
    
    parser.add_argument('--start_dir_valid_txt1', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation/label')
    #parser.add_argument('--start_dir_valid_txt1', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/131_foreign_spoken_korean/01.데이터_new_20220719/2.Validation/label/5. 한국문화II')
    #parser.add_argument('--start_dir_valid_txt2', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/011_korean_child/01.데이터/2.Validation/label_free')
    
    parser.add_argument('--dataset_name', type=str, default='131_foreign_spoken_korean')
    parser.add_argument('--dest_asr_dir', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/korean_speech/asr_data/')
    parser.add_argument('--dest_ssl_dir', type=str, default='/NasData/home/junewoo/raw_dataset/speech_recognition/korean_speech_dataset/korean_speech/ssl_data/')
    args = parser.parse_args()
    
    asr_save_dir = os.path.join(args.dest_asr_dir, args.dataset_name)
    if not os.path.exists(asr_save_dir):
        os.makedirs(asr_save_dir, exist_ok=True)
    
    ssl_save_dir = os.path.join(args.dest_ssl_dir, args.dataset_name)
    if not os.path.exists(ssl_save_dir):
        os.makedirs(ssl_save_dir, exist_ok=True)
    
    args.asr_save_dir = asr_save_dir
    args.ssl_save_dir = ssl_save_dir
        
    
    #label_directory(args.start_dir_train_wav1, args.start_dir_train_txt1, '1.Training') #Training
    
    label_directory(args.start_dir_valid_wav1, args.start_dir_valid_txt1, '2.Validation') #Validation