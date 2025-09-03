import numpy as np
from scipy import signal
import scipy.signal
import pandas as pd

def apply_welch_transform(data_array):
    print(data_array.shape)
    length= data_array.shape[0]
    if length < 2:
        raise ValueError("Data array must have at least two elements for Welch's method.")
    if length % 2 != 0:
#        raise ValueError("Data array length must be even for Welch's method.")
        #data_array = np.pad(data_array, (0, 1), mode='constant', constant_values=0)
        print("Data array length was odd, padded to even length:", data_array.shape)
        data_array=data_array[:-1]  # Remove the last element to make it even
        nperseg = length // 2
        tukey_window = scipy.signal.get_window(('tukey', 0.2), nperseg)
        data_arrary_welch=scipy.signal.welch(data_array, fs=2000, nperseg=nperseg,noverlap=nperseg//2, window=tukey_window)
        return data_arrary_welch[1]
    else:
        nperseg = length // 2
        tukey_window = scipy.signal.get_window(('tukey', 0.2), nperseg)
        data_arrary_welch=scipy.signal.welch(data_array, fs=2000, nperseg=nperseg,noverlap=nperseg//2, window=tukey_window)
        return data_arrary_welch[1]

def get_band_power(data, band_start, band_end):
    freq_axs = np.linspace(0, 1000, len(data))
    band_data = data[(freq_axs >= band_start) & (freq_axs <= band_end)]
    power_sum = np.sum(band_data)
    freq_diff = freq_axs[1] - freq_axs[0]
    return power_sum * freq_diff

def get_all_band_power_from_welchdf(df, event_list):
    new_boxplot_df = df.copy()

    bands_dict= {
        'theta': (4, 12),
        'beta': (12, 30),
        'gamma': (30, 80),
        'total': (0, 100)
    }
    for col in event_list:
        for band, (band_start, band_end) in bands_dict.items():
            print(band, band_start, band_end)
            new_boxplot_df[band + '_' + col] = df[col].apply(lambda x: get_band_power(x, band_start, band_end))
    new_boxplot_df = new_boxplot_df.drop(columns=event_list)
    return new_boxplot_df

def get_band_power_mt(data, band_start, band_end):
    freq_axs = np.linspace(0, 100, len(data))
    band_data = data[(freq_axs >= band_start) & (freq_axs <= band_end)]
    power_sum = np.sum(band_data)
    freq_diff = freq_axs[1] - freq_axs[0]
    band_power_linear = power_sum * freq_diff
    #decibel_power = 10*np.log10(band_power_linear)
    return band_power_linear

def get_all_band_power_from_mt(df, event_list):
    new_boxplot_df = df.copy()

    bands_dict= {
        'theta': (4, 12),
        'beta': (12, 30),
        'gamma': (30, 80),
        'total': (0, 100)
    }
    for col in event_list:
        for band, (band_start, band_end) in bands_dict.items():
            print(band, band_start, band_end)
            new_boxplot_df[band + '_' + col] = df[col].apply(lambda x: get_band_power_mt(x, band_start, band_end))
    new_boxplot_df = new_boxplot_df.drop(columns=event_list)
    return new_boxplot_df