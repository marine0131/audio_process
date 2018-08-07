#! /usr/bin/env python
import glob
import os
import sys
import librosa
import librosa.display as display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import specgram
#%matplotlib inline
plt.style.use('classic')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'Ubuntu'
plt.rcParams['font.monospace'] = 'Ubuntu Mono'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18
# plt.rcParams['figure.titlesize'] = 13
def plot_specgram():
    i=[1,1,1,1,1,1,1,1,1,1]
#    sub_dirs = ['fold1','fold2','fold3','fold4','fold5','fold6','fold7','fold8','fold9','fold10']
    sub_dirs = ['fold41']
    for label, sub_dir in enumerate(sub_dirs):
        for fil in os.listdir(sub_dir):
            fil1=os.path.join(sub_dir,fil)
            fig = plt.figure(figsize=(60,30), dpi = 300)
            X,sr = librosa.load(fil1)
            specgram(np.array(X), Fs=22050)
            # fig.savefig('i'+str(label+1)+'/'+str(label+1)+'_'+str(i[label])+'.png')   # save the figure to file
            label = fil.split('.')[0]
            # fig.savefig(label + '.png')   # save the figure to file
            # i[label]=i[label]+1
            plt.show()
            plt.waitforbuttonpress()
            plt.close()

def plot_spec(f, index):
    print("loading {}".format(f))
    y,sr = librosa.load(f)
    # specgram(np.array(X), Fs=22050)
    print("loaded {} data with {} hz".format(len(y), sr))

    subplot_num = 3
    subplot_index = 1

    # set the hop length, at 22050 hz, 512 samples ~= 23ms
    hop_length = 512

    # normalize
    norm_y = y # librosa.util.normalize(y)

    # figure
    fig = plt.figure(figsize=(60,30), dpi = 300)

    # time waveform plot
    plt.subplot(subplot_num,1,subplot_index)
    subplot_index += 1
    display.waveplot(y=norm_y, sr=sr)
    plt.ylim(-1.0, 1.0)

    # beat clicks
    # tempo, beats = librosa.beat.beat_track(y=norm_y, sr=sr)
    # beat_times = librosa.frames_to_time(frames=beats, sr=sr)
    # print(beat_times)

    # zero crossing
    z = librosa.zero_crossings(y)
    z_num = len(z[z==True])
    print('zero crossing num: {}'.format(z_num))

    # zero-crossing rate
    z = librosa.feature.zero_crossing_rate(y)
    print('zero crossing rate: {}'.format(z.shape))

    # compute stft and turn to db
    # D = librosa.amplitude_to_db(librosa.stft(norm_y), ref=np.max)
    if_gram, D = librosa.ifgram(y=norm_y, sr=sr, n_fft=2048, hop_length=hop_length)
    S, phase = librosa.magphase(D)

    plt.subplot(subplot_num,1,subplot_index)
    subplot_index += 1
    DD = librosa.amplitude_to_db(librosa.stft(norm_y), ref=np.max)
    display.specshow(DD, x_axis='time', y_axis='hz')
    plt.colorbar(orientation='horizontal', format='%+2.0f dB')

    with open("/home/whj/test.txt", 'w') as f:
        np.savetxt(f, DD)
    # display.waveplot(DD[:,100])

    # rms
    rms = librosa.feature.rmse(S=S)
    print('rms shape: {}'.format(rms.shape))

    # roll-off
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    print('roll-off shape: {}'.format(rolloff.shape))

    # tonnetz
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    print('tonnetz shape: {}'.format(tonnetz.shape))
    # plt.subplot(subplot_num, 1, subplot_index)
    # subplot_index += 1
    # display.specshow(tonnetz)

    # centroid
    cent = librosa.feature.spectral_centroid(S=np.abs(D), freq=if_gram)
    print('spectrum centroid shape: {}'.format(cent.shape))

    # spectral bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(S=np.abs(D), freq=if_gram)
    print('spectral_bandwidth shape: {}'.format(spec_bw.shape))
    # librosa.display.specshow(spec_bw, y_axis='spec_bw')

    # chroma cqt
    chroma_cq = librosa.feature.chroma_cqt(y=norm_y, sr=sr, n_chroma=12)
    print('chroma cqt shape: {}'.format(chroma_cq.shape))

    # Chroma cens
    chroma_cens = librosa.feature.chroma_cens(y=norm_y, sr=sr, n_chroma=12)
    print('chroma cens shape: {}'.format(chroma_cens.shape))
    # librosa.display.specshow(chroma_cens, y_axis='chroma')

    # estimate global tempo
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    # tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    # ac_global = librosa.autocorrelate(oenv, max_size=tempogram.shape[0])
    # ac_global = librosa.util.normalize(ac_global)
    # estimate tempo
    tempo = librosa.beat.tempo(onset_envelope=oenv, sr=sr, hop_length=hop_length)[0]
    print("tempo: {}".format(tempo))

    # compute MFCC features from the raw signal
    # MEL = librosa.feature.melspectrogram(y=norm_y, sr=sr, hop_length=hop_length)
    # print('mel shape: {}'.format(MEL.shape))

    fig.savefig(os.path.splitext(f)[0] + '.png')
    plt.close()


if __name__ == "__main__":
    ff = []
    try:
        if os.path.isdir(sys.argv[1]):
            ff = [os.path.join(sys.argv[1], f) for f in os.listdir(sys.argv[1]) if f.split('.')[1]=='mp3']
        else:
            for f in sys.argv:
                ff.append(f)
            ff.remove(ff[0])
    except Exception as e:
        print(e)
    # f = "fold41/0-4s.mp3"

    index = 1
    for f in ff:
        plot_spec(f,index)
        index = index + 1
