from scipy.signal import butter, lfilter, filtfilt, iirnotch
from numpy.fft import rfft,fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mne_connectivity
import mne
import os
def exp_params(string):
    """
    Extracts the date, mouse ID, and task from the given string.

    Parameters:
    string (str): The string from which to extract the parameters.

    Returns:
    tuple: A tuple containing the date, mouse ID, and task extracted from the string.
    """    
    string = string.split('_')
    if len(string) >= 4:
        date = string[0]
        mouse_id = string[1]
        task = string[2]+string[3]
    else:
        date = string[0]
        mouse_id = string[1]
        task = string[2]

    return date, mouse_id, task

def identify_events_channel(f, channels):
    """
    Identify the events channel based on the given channels.

    Parameters:
    f : the data file.
    channels (list): the list of channels in the data file.

    Returns:
    events(array): The events annotations from data file.
    """
    if 'Keyboard' in channels:
        events = f['Keyboard']
    elif 'keyboard' in channels:
        events = f['keyboard']
    elif 'memory' in channels:
        events = f['memory']
    elif 'Memory' in channels:
        events = f['Memory']
    return events

def generate_epochs_with_first_event(events, time):
    """
    Generate epochs with the first event.
    Parameters:
    - events (numpy.ndarray): Array of events.
    - time (numpy.ndarray): Array of time values.
    Returns:
    - epochs (list): List of epochs, where each epoch is a numpy.ndarray containing two rows.
    """
    pass
    
    
    valid_events = [98, 119, 120, 48, 49]
    events_concat = np.vstack((time, events)).T
    events_concat = events_concat[np.isin(events_concat[:, 1], valid_events)]
    
    epochs = []

    # Initialize variables to track the start of an epoch
    start_index = None

    # Iterate through events_concat
    for i, event in enumerate(events_concat):
        if event[1] == 119 or event[1] == 98 or event[1] == 120:
            if start_index is None:
                # Start a new epoch
                start_index = i
            else:
                # End the current epoch
                end_index = i
                epoch = events_concat[start_index:end_index]
                if epoch.shape[0] > 1:
                    epoch = epoch[:2]  # Ensure the epoch has only 2 rows
                    epochs.append(epoch)
                start_index = None
    return epochs

def generate_specific_num_of_epochs_with_first_event(events, time, num_epochs):
    """
    Generate epochs with the first event.
    Parameters:
    - events (numpy.ndarray): Array of events.
    - time (numpy.ndarray): Array of time values.
    Returns:
    - epochs (list): List of epochs, where each epoch is a numpy.ndarray containing two rows.
    """
    pass
    
    
    valid_events = [98, 119, 120, 48, 49]
    events_concat = np.vstack((time, events)).T
    events_concat = events_concat[np.isin(events_concat[:, 1], valid_events)]
    
    epochs = []

    # Initialize variables to track the start of an epoch
    start_index = None

    # Iterate through events_concat
    for i, event in enumerate(events_concat):
        if event[1] == 119 or event[1] == 98 or event[1] == 120:
            if start_index is None:
                # Start a new epoch
                start_index = i
            else:
                # End the current epoch
                end_index = i
                epoch = events_concat[start_index:end_index]
                if epoch.shape[0] > 1:
                    epoch = epoch[:2]  # Ensure the epoch has only 2 rows
                    epochs.append(epoch)
                start_index = None
    return epochs[0:num_epochs]

def generate_epochs_with_all_digs(events, time):
    
    pass
    
    
    valid_events = [98, 119, 120, 48, 49]
    events_concat = np.vstack((time, events)).T
    events_concat = events_concat[np.isin(events_concat[:, 1], valid_events)]
    
    epochs = []

    # Initialize variables to track the start of an epoch
    start_index = None

    # Iterate through events_concat
    for i, event in enumerate(events_concat):
        if event[1] == 119 or event[1] == 98 or event[1] == 120:
            if start_index is None:
                # Start a new epoch
                start_index = i
            else:
                # End the current epoch
                end_index = i
                epoch = events_concat[start_index:end_index]
                epochs.append(epoch)
                start_index = None
    return epochs
def find_correct_dig(epoch):
    """
    Find the correct dig in the given epoch.
    Parameters:
    - epoch (numpy.ndarray): The epoch to search for the correct dig.
    Returns:
    - correct_dig (numpy.ndarray): The correct dig in the epoch.
    """
    correct_dig = epoch[epoch[:, 1] == 49]
    return correct_dig
def create_channel_dict(channels):
    channel_dict = {
        channel: (
            "LFP_AoN_1" if channel.endswith("_Ch1") else
            "LFP_AoN_2" if channel.endswith("_Ch2") else
            "LFP_AoN_3" if channel.endswith("_Ch3") else
            "LFP_VHC_1" if channel.endswith("_Ch4") else
            "LFP_VHC_2" if channel.endswith("_Ch5") else
            "LFP_VHC_3" if channel.endswith("_Ch6") else
            "reference" if channel.endswith("_Ch8") else
            "keyboard" if channel.endswith("_Ch31") else
            "Unknown"
        )
        for channel in channels
    }
    return channel_dict
def get_keyboard_and_ref_channels(f, channels):
    if 'Keyboard' in channels:
        events = f['Keyboard']
    elif 'keyboard' in channels:
        events = f['keyboard']
    elif 'memory' in channels:
        events = f['memory']
    elif 'Memory' in channels:
        events = f['Memory']
    if 'Ref' in channels:
        reference_electrode = f['Ref']
    elif 'ref' in channels:
        reference_electrode = f['ref']
    elif 'REF' in channels:
        reference_electrode = f['REF']
    return events, reference_electrode
def get_key_from_value(dictionary, target_value):
    for key, value in dictionary.items():
        if target_value in value:
            return key
    return None
def clean_tasks(tasks):
    """
    Removes specified substrings from each task in the list and returns the cleaned list of tasks.
    
    Parameters:
    tasks (list of str): The list of task strings to be cleaned.
    
    Returns:
    list of str: The cleaned list of task strings.
    """
    substrings_to_remove = ['day2', 'os2', 'discard']
    
    for i, task in enumerate(tasks):
        if any(substring in task for substring in substrings_to_remove):
            #print(f"Original task: {task}")
            
            # Remove the substrings from task
            for substring in substrings_to_remove:
                task = task.replace(substring, '')
            
            # Optionally, strip any leading/trailing whitespace
            task = task.strip()
            
            #print(f"Updated task: {task}")
            tasks[i] = task
    
    return tasks
def clean_task(task):
    """
    Removes specified substrings from the task string and returns the cleaned task.
    
    Parameters:
    task (str): The task string to be cleaned.
    
    Returns:
    str: The cleaned task string.
    """
    substrings_to_remove = ['day2', 'os2', 'discard']
    
    # Remove the substrings from task
    for substring in substrings_to_remove:
        task = task.replace(substring, '')
    
    # Strip any leading/trailing whitespace
    task = task.strip()
    
    return task

def iir_notch(data, fs, frequency, quality=5., axis=-1):

    norm_freq = frequency/(fs/2)
    b, a = iirnotch(norm_freq, quality)
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    print('notch filter applied')
    return y

def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=4, axis=-1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    return y

def butter_highpass_filter(data, cutoff, fs, order=5, axis=-1):
    nyq = fs * 0.5
    normal_cutoff = cutoff/nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    return y

def frequency_domain(data, time):
    x= np.array(data)
    dt = time[1] - time[0]
    N = len(x)
    T = N * dt
    xf = fft(x - x.mean())
    Sxx = 2 * dt ** 2 / T * (xf * xf.conj())
    Sxx = Sxx[:int(len(x) / 2)]
    df = 1 / T
    fNQ = 1 / dt / 2
    faxis = np.arange(0,fNQ,df)[:len(Sxx)]
    return faxis, Sxx.real

def baseline_data_normalization(data,time,first_event,sampling_rate):
    if first_event>2.0:
        baseline_data=data[np.where(time>first_event)[0][0]-2*sampling_rate:np.where(time>first_event)[0][0]]
    else:
        baseline_data=data[0:np.where(time>first_event)[0][0]]
    baseline_mean=np.mean(baseline_data)
    baseline_std=np.std(baseline_data)
    
    baseline_data_norm=(baseline_data-baseline_mean)/baseline_std
    print('normalizing data')
    return baseline_data_norm,time, baseline_mean, baseline_std

def zscore_event_data(data,baseline_mean, baseline_std):
    data_mean=np.mean(data)
    data_zscored = (data - data_mean) / baseline_std
    return data_zscored

def data_normalization(data,time,first_event,sampling_rate):
    if first_event>30.0:
        baseline_data=data[np.where(time>first_event)[0][0]-30*sampling_rate:np.where(time>first_event)[0][0]]
    else:
        baseline_data=data[0:np.where(time>first_event)[0][0]]
    mean=np.mean(baseline_data)
    std=np.std(baseline_data)
    
    data_norm=(data-mean)/std
    print('normalizing data')
    return data_norm,time, baseline_data

def freq_band(data,low,high,sampling_rate):
    b,a=butter(3, [low,high], fs=sampling_rate, btype='band')
    data_filtered=filtfilt(b,a,data)
    return data_filtered
def get_band_power(data, band_start, band_end):
    freq_axs = np.linspace(0, 1000, len(data))
    band_data = data[(freq_axs >= band_start) & (freq_axs <= band_end)]
    power_sum = np.sum(band_data)
    freq_diff = freq_axs[1] - freq_axs[0]
    return power_sum * freq_diff

def alpha_band(data):
    b,a=butter(4, [8,12], fs=2000, btype='band')
    data_filtered=filtfilt(b,a,data)
    return data_filtered
def delta_band(data):
    b,a=butter(4, [0.5,4], fs=2000, btype='band')
    data_filtered=filtfilt(b,a,data)
    return data_filtered

def beta_band(data,sampling_rate):
    b,a=butter(4, [12,30], fs=sampling_rate, btype='band')
    data_filtered=filtfilt(b,a,data)
    print('filtering beta band')
    return data_filtered

def gamma_band(data,sampling_rate):
    b,a=butter(4, [30,80], fs=sampling_rate, btype='band')
    data_filtered=filtfilt(b,a,data)
    print('filtering gamma band')
    return data_filtered

def theta_band(data,sampling_rate):
    b,a=butter(4, [4,12], fs=sampling_rate, btype='band')
    data_filtered=filtfilt(b,a,data)
    print('filtering theta band')
    return data_filtered

def generate_epochs(events,time):
    events_concat=np.vstack((time,events)).T
    epochs = []

    # Initialize variables to track the start of an epoch
    start_index = None

    # Iterate through events_concat
    for i, event in enumerate(events_concat):
        if event[1] == 119 or event[1] == 98 or event[1]==120:
            if start_index is None:
                # Start a new epoch
                start_index = i
            else:
                # End the current epoch
                end_index = i
                epoch = events_concat[start_index:end_index]
                epochs.append(epoch)
                start_index = None
    return epochs
def find_global_start_end_times(f,channels):
    # Find global start and end times
    global_start_time = float('inf')
    global_end_time = float('-inf')

    for channeli in channels:
        if "AON" in channeli or "vHp" in channeli or 'Ref' in channeli:
            data_all = f[channeli]
            raw_time = np.array(data_all['times']).flatten()
            global_start_time = min(global_start_time, raw_time[0])
            global_end_time = max(global_end_time, raw_time[-1])
    return global_start_time, global_end_time
def pad_raw_data_raw_time(data, time, global_start_time, global_end_time, sampling_rate=2000):
    
    total_points = int((global_end_time - global_start_time) * sampling_rate) + 1
    padded_data = np.zeros(total_points)
    start_index = int((time[0] - global_start_time) * sampling_rate)
    end_index = start_index + len(data)
    padded_data[start_index:end_index] = data
    padded_time = np.linspace(global_start_time, global_end_time, total_points)
    return padded_data, padded_time



def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    return padded_sequences


def extract_complete_trial_data(data, time, door_timestamp,dig_timestamp, sampling_rate):
    door_index = np.where(time > door_timestamp)[0][0]
    dig_index = np.where(time > dig_timestamp)[0][0]
    data_complete_trial = data[door_index- 2 * sampling_rate:dig_index+ 2 * sampling_rate]
    return np.array(data_complete_trial,dtype=float)

def extract_door_data(data, time, door_timestamp, sampling_rate):
    door_index = np.where(time > door_timestamp)[0][0]
    data_door_before = data[door_index- 2 * sampling_rate:door_index]
    data_door_after = data[door_index:door_index + 2 * sampling_rate]
    #print(f"Extracted door data from index {door_index} to {door_index + 2 * sampling_rate}")
    #print(len(data_door_before),len(data_door_after))

    return data_door_before, data_door_after

def extract_dig_data(data, time, dig_timestamp, sampling_rate):
    dig_index = np.where(time > dig_timestamp)[0][0]
    data_dig_after = data[dig_index:dig_index + 2 * sampling_rate]
    data_dig_before = data[dig_index - 2 * sampling_rate:dig_index]
    #print(f"Extracted dig data before from index {dig_index - 2 * sampling_rate} to {dig_index}")
    #print(f"Extracted dig data after from index {dig_index} to {dig_index + 2 * sampling_rate}")
    #print(len(data_dig_before),len(data_dig_after))
    return data_dig_before, data_dig_after



def divide_data_in_epochs(data,time,epochs,sampling_rate):
    white_trials = []
    black_trials = []
    for epochi in epochs:
        trial_timestamp = epochi[0][0]  # Get the timestamp of the trial
        print(trial_timestamp)
    
        if epochi[0][1] == 119:  # Check if the trial is a white trial
            trial_index = np.where(time > trial_timestamp)[0][0]  # Find the index of the trial start
            data_trial = data[trial_index:trial_index + 2 * sampling_rate]  # Extract the trial data
            white_trials.append(data_trial)
            print('this was a white trial')
        elif epochi[0][1] == 98:  # Check if the trial is a white trial
            trial_index = np.where(time > trial_timestamp)[0][0]  # Find the index of the trial start
            data_trial = data[trial_index:trial_index + 2 * sampling_rate]  # Extract the trial data
            black_trials.append(data_trial)
            print('this was a black trial')
    return np.array(white_trials), np.array(black_trials)

def calculate_power_1D(signal):
    power = np.sum(signal ** 2) / len(signal)
    return power

def convert_epoch_to_coherence(epoch):
    band_dict={'theta':[4,8],'beta':[12,30],'gamma':[30,100],'total':[1,100]}
    coherence_dict={}
    for band in band_dict.keys():

        fmin=band_dict[band][0]
        fmax=band_dict[band][1]
        freqs = np.arange(fmin,fmax)
        n_cycles = freqs/2
        con=mne_connectivity.spectral_connectivity_time(epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,faverage=True, mode='cwt_morlet', verbose=False, n_cycles=n_cycles,freqs=freqs)
        coh = con.get_data(output='dense')
        #print(coh)
        indices = con.names
        #print(indices)
        aon_vhp_con=[]
        #print(coh.shape)
        for i in range(coh.shape[1]):
            for j in range(coh.shape[2]):
                print(i,j)
                if 'AON' in indices[j] and 'vHp' in indices[i]:
                    print('AON and vHp found')
                    aon_vhp_con.append(coh[0,i,j,:])
                    print(coh[0,i,j,:])
                else:
                    continue
        if aon_vhp_con==[]:
            print('no coherence found')
        else:
            #print(aon_vhp_con)
            aon_vhp_con_mean=np.mean(aon_vhp_con, axis=0)
            #print(aon_vhp_con_mean, 'coherenece')
            coherence_dict[band]=aon_vhp_con_mean[0]
    return coherence_dict

def convert_epoch_to_coherence_density(epoch, fmin=1, fmax=100):

    freqs = np.arange(fmin,fmax)
    n_cycles = freqs/2
    con=mne_connectivity.spectral_connectivity_time(epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,faverage=False, mode='cwt_morlet', verbose=False, n_cycles=n_cycles,freqs=freqs)
    coh = con.get_data(output='dense')
    #print(coh)
    indices = con.names
    #print(indices)
    aon_vhp_con=[]
    #print(coh.shape)
    for i in range(coh.shape[1]):
        for j in range(coh.shape[2]):
            print(i,j)
            if 'AON' in indices[j] and 'vHp' in indices[i]:
                print('AON and vHp found')
                aon_vhp_con.append(coh[0,i,j,:])
                print(coh[0,i,j,:])
            else:
                continue
    if aon_vhp_con==[]:
        print('no coherence found')
    else:
        #print(aon_vhp_con)
        aon_vhp_con_mean=np.mean(aon_vhp_con, axis=0)
        #print(aon_vhp_con_mean, 'coherenece')
            
    return aon_vhp_con_mean


def convert_epoch_to_coherence_behavior(epoch, band_start, band_end):
    fmin = band_start
    fmax = band_end
    freqs = np.arange(fmin, fmax)
    n_cycles = freqs / 2
    con = mne_connectivity.spectral_connectivity_time(
        epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,
        faverage=True, mode='cwt_morlet', verbose=False, n_cycles=n_cycles, freqs=freqs
    )
    coh = con.get_data(output='dense')
    indices = con.names
    aon_vhp_con = []

    for i in range(coh.shape[1]):
        for j in range(coh.shape[2]):
            if 'AON' in indices[i] and 'vHp' in indices[j]:
                aon_vhp_con.append(coh[0, i, j, :])

    if not aon_vhp_con:  # If the list is empty
        print('No coherence found')
        aon_vhp_con_mean = np.zeros_like(freqs)  # Assign a default value (e.g., zeros)
    else:
        aon_vhp_con_mean = np.mean(aon_vhp_con, axis=0)

    return aon_vhp_con_mean[0]

def convert_epoch_to_coherence_behavior_short_signal(epoch_data, band_start, band_end):

    """
    Calculates a single coherence value, averaged over a frequency band and
    over AON-vHp channel pairs, for a given epoch.

    Args:
        epoch_data (np.ndarray): The data for a single epoch/trial, expected
                                 shape (n_channels, n_times).
        sfreq (int): The sampling frequency of the data.
        channel_names (list of str): List of channel names.
        band_start (float): The starting frequency of the band.
        band_end (float): The ending frequency of the band.

    Returns:
        float: A single coherence value averaged over the band and specified
               channel pairs. Returns np.nan if no valid coherence is computed
               (e.g., no frequencies, no pairs, or error).
    """
    fmin_band = float(band_start)
    fmax_band = float(band_end)
    sfreq = epoch_data.info['sfreq']
    extracted_ch_names = epoch_data.info['ch_names'] # Get channel names
    n_times_signal = epoch_data.n_times # Number of time points per epoch

    # Freqs for n_cycles calculation (MNE uses these for wavelet construction)
    # If faverage=True, MNE also needs fmin, fmax for averaging, which it can
    # derive from 'freqs' or use explicitly if passed.
    # We define freqs to cover the band for n_cycles calculation.
    # spectral_connectivity_epochs will use the min and max of 'freqs' as the
    # fmin and fmax for averaging if faverage=True and fmin/fmax are not explicitly set.
    freqs_for_n_cycles = np.arange(fmin_band, fmax_band)

    if freqs_for_n_cycles.size == 0:
        print(f"Warning: No frequencies in band fmin={fmin_band}, fmax={fmax_band} for n_cycles. Returning np.nan.")
        return np.nan

    n_times_signal = epoch_data.shape[-1]

    with np.errstate(divide='ignore', invalid='ignore'):
        n_cycles_max_due_to_length = (n_times_signal * np.pi * freqs_for_n_cycles) / (5.0 * sfreq)
    n_cycles_max_due_to_length[np.isinf(n_cycles_max_due_to_length) | np.isnan(n_cycles_max_due_to_length)] = freqs_for_n_cycles.max() / 2.0 if freqs_for_n_cycles.size > 0 else 3.0


    desired_n_cycles = freqs_for_n_cycles / 2.0
    n_cycles = np.minimum(desired_n_cycles, n_cycles_max_due_to_length)
    n_cycles = np.maximum(n_cycles, 1.0) # Ensure at least 1 cycle

    if epoch_data.ndim == 2:
        data_reshaped = epoch_data[np.newaxis, :, :]
    elif epoch_data.ndim == 3:
        data_reshaped = epoch_data
    else:
        raise ValueError("epoch_data must be 2D (n_channels, n_times) or 3D (n_epochs, n_channels, n_times)")

    try:
        con = mne_connectivity.spectral_connectivity_time(
            data_reshaped,
            method='coh',
            sfreq=sfreq,
            # For faverage=True, MNE will average over the range defined by
            # the min and max of `freqs` (or explicit fmin, fmax if passed to `spectral_connectivity_epochs`).
            # It's good practice to ensure `freqs` accurately reflects the band for `n_cycles`.
            freqs=freqs_for_n_cycles, # Defines wavelets used before averaging
            fmin=fmin_band, # Explicitly define band for averaging
            fmax=fmax_band, # Explicitly define band for averaging
            n_cycles=n_cycles,
            faverage=True, # <<< Key change: Average coherence over the frequency band
            mode='cwt_morlet',
            verbose=False,
            n_times
        )

        # With faverage=True, coh_data shape: (n_epochs_processed, n_signals, n_signals)
        # or (n_signals, n_signals) if epochs are squeezed.
        coh_data = con.get_data(output='dense')
        if coh_data.ndim == 3 and coh_data.shape[0] == 1: # If epoch dim was kept
            coh_data = coh_data[0] # Squeeze out the single epoch dimension
        # coh_data should now be (n_signals, n_signals)

        indices = con.attrs.get('names_original', channel_names)
        if indices is None:
            indices = channel_names

        aon_vhp_coherence_values = []
        for i in range(coh_data.shape[0]):
            for j in range(coh_data.shape[1]):
                if i == j:
                    continue
                if len(indices) > max(i,j) and 'AON' in indices[i] and 'vHp' in indices[j]:
                    aon_vhp_coherence_values.append(coh_data[i, j]) # Append scalar coherence

        if not aon_vhp_coherence_values:
            # print(f'No AON-vHp coherence pairs found for band {fmin_band}-{fmax_band} Hz. Returning np.nan.')
            return np.nan
        else:
            # Average the scalar coherence values from all found AON-vHp pairs
            mean_band_coherence = np.mean(aon_vhp_coherence_values)
            return mean_band_coherence

    except ValueError as ve:
        print(f"ValueError during connectivity (faverage=True) for band {fmin_band}-{fmax_band} Hz: {ve}")
        # print(f"Signal n_times: {n_times_signal}, Freqs for n_cycles: {freqs_for_n_cycles[:5]}..., n_cycles used: {n_cycles[:5]}...")
        return np.nan
    except Exception as e:
        print(f"Unexpected error (faverage=True) for band {fmin_band}-{fmax_band} Hz: {e}")
        # print(f"Signal n_times: {n_times_signal}, Freqs for n_cycles: {freqs_for_n_cycles[:5]}..., n_cycles used: {n_cycles[:5]}...")
        return np.nan