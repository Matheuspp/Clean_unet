import pathlib
import multiprocessing
from shutil import move
import soundfile as sf
import librosa
import numpy as np
# ### change here
desired_length=10
path='training_set/clean'
target = 'training_set/clean_pre'
# ### =======================

def rename(folder):
    files = pathlib.Path(folder).glob('*')
    for i, audio in enumerate(sorted(files)):
        move(audio, f'{folder}/fileid_{i}.wav')
        print(audio)

def divide_list(lst, n):
    avg = len(lst) / n
    parts = []
    last = 0.0

    while last < len(lst):
        parts.append(lst[int(last):int(last + avg)])
        last += avg

    return parts

def trim(files):
    acum, sr = librosa.load(str(files[0]), sr=48000)
    idx = 0
    end = sr*desired_length
    for audio in files[1:]:
        name = audio.split('/')[-1]
        # Load the audio file
        y, sr = librosa.load(str(audio), sr=48000)
        acum = np.concatenate([acum, y])
        if acum.shape[0] > end:
            y = acum[0:end]
            sf.write(file=f'{path}/{name}', data=y, samplerate=sr)
            acum = acum[end:]

    return acum
def trim_remain(lst, sr=48000):
    acum = lst[0]
    end = sr*desired_length
    for i, item in enumerate(lst[1:]):
        acum = np.concatenate([acum, item])
        if acum.shape[0] > end:
            y = acum[0:end]
            sf.write(file=f'{path}/{i}.wav', data=y, samplerate=sr)
            acum = acum[end:]


if __name__ == '__main__':
    file = sorted(pathlib.Path(target).glob('*'))
    files = [str(path) for path in file]
    num_parts = 8
    list_parts = divide_list(files, num_parts)

    pool = multiprocessing.Pool(processes=num_parts)
    results = pool.map(trim, list_parts)

    remained_list = [item for item in results]
    trim_remain(remained_list)
    rename(path)
    print('Done!')
