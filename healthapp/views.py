# Create your views here.
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json

from cProfile import label
import pywt
import numpy as np
import pandas as pd
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import welch
from scipy.integrate import simps
from scipy.signal import butter, lfilter, filtfilt
from matplotlib import pyplot as plt
import statistics


@csrf_exempt
def index(request):
    if request.method == 'POST':
        json_data = json.loads(request.body)
        main = Main()
        df = main.Bruxism(json_data).to_dict()
        print(df)
        json_object = json.dumps(df, indent=4)
        print(json_object)
    return HttpResponse(json_object,content_type="application/json")



class Main:
    def __init__(self):
        pass

    def FFT(Data, Fs):
        N = len(Data)
        yf = fft(Data)
        xf = fftfreq(N, 1 / (Fs))
        ft = np.abs(yf)
        Freq = xf[1:(Fs - 1)]
        FreqVal = ft[1:(Fs - 1)]
        return Freq, FreqVal

    def WaveLet(Data):
        (cA, cD) = pywt.dwt(Data, 'db2')
        return np.mean(cA), np.mean(cD)

    def butter_bandpass(lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='pass')

    def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = Main.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_Butterworth(lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype='stop')

    def butter_Butterworth_filter(data, lowcut, highcut, fs, order=5):
        b, a = Main.butter_Butterworth(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def butter_highpass(cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='high', analog=False)
        return b, a

    def butter_highpass_filter(data, cutoff, fs, order=5):
        b, a = Main.butter_highpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def bandpower(data, sf, band, window_sec=None, relative=False):
        """
        Bandpower Function
        """
        band = np.asarray(band)
        low, high = band
        if window_sec is not None:
            nperseg = window_sec * sf
        else:
            nperseg = (2 / low) * sf
        freqs, psd = welch(data, sf, nperseg=nperseg)
        freq_res = freqs[1] - freqs[0]
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = simps(psd[idx_band], dx=freq_res)
        if relative:
            bp /= simps(psd, dx=freq_res)
        return bp

    def bandPow(Data, Fs):
        # Filtering
        Alfa = Main.butter_bandpass_filter(Data, 8, 12, Fs, order=6)
        Delta = Main.butter_bandpass_filter(Data, 1, 4, Fs, order=6)
        Beta = Main.butter_bandpass_filter(Data, 12, 30, Fs, order=6)
        Theta = Main.butter_bandpass_filter(Data, 4, 8, Fs, order=6)
        return Alfa, Delta, Beta, Theta

    def Bruxism(self,data):
        print("Bruxism Process Started")


        df1 = pd.DataFrame(data)
        data = df1["F4_C4"].to_numpy()


        Fs = 256
        time = len(data) / Fs
        segmentedArry = np.split(data, (time))

        ##Window = segmentedArry[0][:]
        # time vector
        shape = np.shape(segmentedArry)
        ones = np.zeros(shape[0])

        for i in range(shape[0]):
            if i > 28 and i < 31:
                ones[i] = 1

        label = ones

        cols = ['Mean', 'DeltaPower', 'ThetaPower', 'AlphaPower', 'BetaPower', 'AlfaBetRatio', 'MaxVal', 'MinVal',
                'AprroxMeanDWT', 'DetailCoeffDWT', 'Std', 'Labels']
        df2 = pd.DataFrame(columns=cols, index=range(int(time)))
        for i in range(shape[0]):
            # Dividing into
            Window = segmentedArry[i][:]
            # FFT of Window
            [Freq, FreqVal] = Main.FFT(Window, Fs)
            # print(FreqVal)
            dbDelta = Main.bandpower(Window, Fs, [0.5, 4], 1)
            dbTheta = Main.bandpower(Window, Fs, [4, 8], 1)
            dbAlpha = Main.bandpower(Window, Fs, [8, 12], 1)
            dbBeta = Main.bandpower(Window, Fs, [12, 30], 1)
            # Max power values
            [Alfa, Delta, Beta, Theta] = Main.bandPow(Window, Fs)

            [FreqBeta, FreqValBeta] = Main.FFT(Beta, Fs)
            [FreqAlfa, FreqValAlfa] = Main.FFT(Alfa, Fs)
            [FreqTheta, FreqValTheta] = Main.FFT(Theta, Fs)
            [FreqDelta, FreqValDelta] = Main.FFT(Delta, Fs)

            MBeta = np.mean(FreqValBeta[12:30])
            MAlfa = np.mean(FreqValAlfa[8:12])
            MTheta = np.mean(FreqValTheta[4:8])
            MDelta = np.mean(FreqValDelta[1:4])
            MaxAlfa = np.max(FreqValAlfa[8:12])
            # Feature Extraction
            df2.loc[i].Mean = np.mean(Window)
            df2.loc[i].DeltaPower = dbDelta
            df2.loc[i].ThetaPower = dbTheta
            df2.loc[i].AlphaPower = dbAlpha
            df2.loc[i].BetaPower = dbBeta
            df2.loc[i].AlfaBetRatio = dbAlpha / dbBeta
            df2.loc[i].MaxVal = np.max(Window)
            df2.loc[i].MinVal = np.min(Window)
            cA, cD = Main.WaveLet(Window)
            df2.loc[i].AprroxMeanDWT = cA
            df2.loc[i].DetailCoeffDWT = cD
            df2.loc[i].Std = statistics.stdev(Window)
            df2.loc[i].Labels = label[i]
        print(df2)
        print("End of Bruxism Process")
        return df2