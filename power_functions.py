import numpy as np

def get_band_power(data, band_start, band_end):
    freq_axs = np.linspace(0, 1000, len(data))
    band_data = data[(freq_axs >= band_start) & (freq_axs <= band_end)]
    power_sum = np.sum(band_data)
    freq_diff = freq_axs[1] - freq_axs[0]
    return power_sum * freq_diff
