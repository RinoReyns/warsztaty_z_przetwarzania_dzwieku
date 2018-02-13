# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import signal as sig
from scipy import log10, unwrap, arctan2, imag, real
from statistics import variance


def get_value(display_string):
    try:
        value = float(input(display_string))
    except ValueError:
        value = 0.0
    return value


def process_sample_duration():
    sample_duration = get_value("Podaj czas trwania jednego impulsu w zakresie od 0.001 do 1 sekundy [s]: ")
    while sample_duration > 1.0 or sample_duration < 0.001:
        print("Czas trwania jednego impulsu został podany nieprawidłowo.")
        sample_duration = get_value("Podaj czas trwania jednego impulsu w zakresie od 0.001 do 1 sekundy [s]: ")
    signal_frequency = 1 / sample_duration
    return sample_duration, signal_frequency


def process_sample_rate(signal_frequency):
    fs = get_value("Podaj czestotliwość próbkowania sygnału [Hz]: ")
    while fs < 4 * signal_frequency:
        print("Czestotliwość próbkowania jest nieprawidłowa.")
        fs = get_value("Podaj czestotliwość próbkowania sygnału [Hz]: ")
    ts = 1 / fs
    return fs, ts


def generate_signal(data, fs, sample_duration):
    signal = np.array([])
    for i in range(0, len(data)):
        for j in range(0, int(fs * sample_duration)):
            if int(data[i]) == 0:
                signal = np.append(signal, [0])
            elif int(data[i]) == 1:
                signal = np.append(signal, [1])
    return signal, len(signal)


def check_input_string():
    string = input("Podaj ciąg znaków do przesłania: ")
    for i in range(0, len(string)):
        if len(bin(int.from_bytes(string[i].encode(), 'big'))) > 9:
            return 1, None
    return 0, string


def get_data():
    status, string = check_input_string()
    while status == 1:
        print("Podany ciąg nie jest prawidłowy, gdyż zawiera znaki diakrytyczne, które nie występują w kodowaniu "
              "ASCII.")
        status, string = check_input_string()
    string = '0' + bin(int.from_bytes(string.encode(), 'big'))[2:]
    print("Ciagi bitowe odpowiadające kolejno podanym znakom:", string)
    for i in range(0, (int(len(string)/8))):
        print(string[i*8:(i+1)*8])
    return string


def generate_time_vector(ts, signal_length):
    t0 = np.arange(0, 1, 2 * ts)
    t = t0
    for i in range(1, int(np.ceil(np.true_divide(signal_length, len(t))))):
        t = np.concatenate((t, t0 + i), 0)
    return t


def display_signal(signal, fs, time_vector, figure_number, extra_title='.'):
    signal = signal[0::2]
    signal_length = len(signal)
    frq = np.arange(signal_length) / np.true_divide(signal_length, fs)
    frq = frq[:int(signal_length / 2)]
    y = np.fft.fft(signal) / signal_length
    y = y[:int(signal_length / 2)]
    plt.figure(figure_number)
    plt.plot(time_vector[:signal_length], signal, drawstyle='steps-post')
    plt.xlabel('Czas [s]')
    plt.ylabel('Amplituda')
    plt.title('Przebieg sygnału nadawanego' + extra_title)
    plt.grid()
    plt.figure(figure_number+1)
    _, _, baseline = plt.stem(frq, abs(y), '-.')
    plt.setp(baseline, 'color', 'r', 'linewidth', 2)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('|Y(częstotliwość )|')
    plt.title('Widmo amplitudowe nadawanego sygnału' + extra_title)
    plt.grid()


def generate_noise(signal_length):
    SNR = float(input("Podaj stosunek szygnału do szumu (SNR) [dB]: "))
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


def get_detection_level():
    detection_level = get_value("Podaj próg detekcji w zakresie od 0.4 do 1.0: ")
    while detection_level > 1.0 or detection_level < 0.4:
        print("Poziom detekcji został podany nieprawidłowo.")
        detection_level = get_value("Podaj próg detekcji w zakresie od 0.4 do 1.0: ")
    return detection_level


def signal_detection(signal):
    up_detection_level = get_detection_level()
    down_detection_level = (-up_detection_level)
    detected_signal = np.array([])
    for value in signal:
        if value >= up_detection_level:
            detected_signal = np.append(detected_signal, [1])
        elif value <= down_detection_level:
            detected_signal = np.append(detected_signal, [0])
        elif up_detection_level > value > down_detection_level:
            detected_signal = np.append(detected_signal, [0])
        else:
            detected_signal = np.append(detected_signal, [1])
    return detected_signal


def decode_data(data):
    for i in range(0, (int(len(data) / 8))):
        print(data[i * 8:(i + 1) * 8])
    string = int(data, 2)
    try:
        print(string.to_bytes((string.bit_length() + 7) // 8, 'big').decode())
    except ValueError:
        print("Odebrana wiadomość nie może zostać odkodowana poprawnie.")


def translation(detected_signal, samples_per_impulse):
    impulse_length = int(len(detected_signal) / samples_per_impulse)
    data = ''
    for i in range(0, impulse_length):
        if np.mean(detected_signal[(i*samples_per_impulse):(i+1)*samples_per_impulse]) > 0:
            data = data + '1'
        else:
            data = data + '0'
    decode_data(data)


def main():
    sample_duration, signal_frequency = process_sample_duration()
    fs, ts = process_sample_rate(signal_frequency)
    data = get_data()
    # data = [1, 0, 1, 0, 1, 0, 1, 0]
    print("Częstotliwość sygnału: ", signal_frequency, " Hz")
    print("Częstotliwość próbkowania : ", fs, " Hz")
    signal, signal_length = generate_signal(data, fs, sample_duration)
    time_vector = generate_time_vector(ts, signal_length)
    display_signal(signal, fs, time_vector, 1)
    noise = generate_noise(signal_length)
    signal_and_noise = signal + noise
    display_signal(signal_and_noise, fs, time_vector, 3, ' z dodanym białym szumem gaussowskim.')
    b, a = sig.butter(6, 4 * signal_frequency / fs)
    filter_response(b, a, fs, signal_length)
    filtered_signal_with_noise = sig.filtfilt(b, a, signal_and_noise)
    display_signal(filtered_signal_with_noise, fs, time_vector, 6, ' po zaszumieniu i filtracji.')
    filtered_snr(b, a, signal, noise)
    detected_signal = signal_detection(filtered_signal_with_noise)
    translation(detected_signal, int(fs*sample_duration))
    display_signal(detected_signal, fs, time_vector, 8, ' po detekcji.')
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        exit(0)
