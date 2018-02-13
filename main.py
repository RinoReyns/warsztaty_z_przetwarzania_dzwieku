# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal as sig
from scipy import log10, unwrap, arctan2, imag, real
from statistics import variance


def generate_time_vector(ts):
    t0 = np.arange(0, 1, ts)
    t = t0
    return t


def display_signal(signal, fs, time_vector, figure_number, extra_title='.'):
    # signal = signal[0::2]
    signal_length = len(signal)
    frq = np.arange(signal_length) / np.true_divide(signal_length, fs)
    frq = frq[:int(signal_length / 2)]
    y = np.fft.fft(signal) / signal_length
    y = y[:int(signal_length / 2)]
    plt.figure(figure_number)
    plt.plot(time_vector[:signal_length], signal, drawstyle='steps-post')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Przebieg czasowy sygnału.' + extra_title)
    plt.grid()
    plt.figure(figure_number+1)
    _, _, baseline = plt.stem(frq, abs(y), '-.')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('|Y(częstotliwość )|')
    plt.title('Widmo amplitudowe sygnału' + extra_title)
    plt.grid()


def generate_noise(signal_length, SNR):
    var = 1 / (10 ** (SNR / 10))
    sigma = math.sqrt(var)
    print("Wariancja szumu wyznaczona na podstawie podanego SNR: ", var)
    return np.random.normal(0, sigma, signal_length)


def filter_response(b, a, fs, signal_length):
    frq = np.arange(signal_length) / np.true_divide(signal_length, fs)
    w, h = sig.freqz(b, a, signal_length)
    h_dB = 20 * log10(abs(h))
    plt.figure(5)
    plt.subplot(211)
    plt.plot(frq/2, h_dB)
    plt.ylabel('Wzmocnienie [dB]')
    plt.xlabel("Częstotliwość [Hz]")
    plt.title("Charakterystyka częstotliwościowa ")
    plt.grid()
    plt.subplot(212)
    h_Phase = unwrap(arctan2(imag(h), real(h)))
    plt.plot(frq/2, np.degrees(h_Phase))
    plt.ylabel("Faza [stopnie]")
    plt.xlabel("Częstotliwość [Hz]")
    plt.title(r"Charakterystyka fazowa")
    plt.grid()
    plt.subplots_adjust(hspace=0.5)


def filtered_snr(b, a, signal, noise):
    filtered_signal = sig.filtfilt(b, a, signal)
    filtered_noise = sig.filtfilt(b, a, noise)
    signal_variance = variance(filtered_signal)
    noise_variance = variance(filtered_noise)
    snr = 10 * log10(signal_variance / noise_variance)
    print("Stosunek sygnału do szumu po filtracji: ", snr, ' [dB]')


def generate_sine_wave(signal_frequency, fs):
    x = np.arange(fs)
    signal = np.sin(2 * np.pi * signal_frequency * x / fs)
    signal_length = len(signal)
    return signal, signal_length


def generate_square_wave(signal_frequency, fs):
    x = np.arange(fs)
    signal = sig.square(2 * np.pi * signal_frequency * x/ fs)
    signal_length = len(signal)
    return signal, signal_length


def generate_sawtooth_wave(signal_frequency, fs):
    x = np.arange(fs)
    signal = sig.sawtooth(2 * np.pi * signal_frequency * x / fs)
    signal_length = len(signal)
    return signal, signal_length


def main():
    # parametry wejsciowe sygnału
    signal_frequency = 20   # częstotliwość sygnału
    fs = 2000              # częstotliwość próbkowania
    ts = 1/fs
    SNR = 10                # dB
    time_vector = generate_time_vector(ts)

    # generowanie sygnału sinusoidalnego
    signal, signal_length = generate_sine_wave(signal_frequency, fs)

    #generowanie sygnału prostokątnego
    # signal, signal_length = generate_square_wave(signal_frequency, fs)

    # generowanie piły
    # signal, signal_length = generate_sawtooth_wave(signal_frequency, fs)

    # wykreślanie sygnału
    display_signal(signal, fs, time_vector, 1)

    # dodawanie szumu do sygnału
    # noise = generate_noise(signal_length, SNR)
    # signal_and_noise = signal + noise
    # display_signal(signal_and_noise, fs, time_vector, 3, ' z dodanym białym szumem gaussowskim.')
    #
    # # filtrowanie sygnału
    # b, a = sig.butter(6, 4 * signal_frequency / fs)
    # filter_response(b, a, fs, signal_length)
    # filtered_signal_with_noise = sig.filtfilt(b, a, signal_and_noise)
    # display_signal(filtered_signal_with_noise, fs, time_vector, 6, ' po zaszumieniu i filtracji.')
    # filtered_snr(b, a, signal, noise)

    plt.show()

if __name__ == "__main__":
    main()
