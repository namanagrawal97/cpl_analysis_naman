from scipy.signal import butter, lfilter, filtfilt, iirnotch
from numpy.fft import rfft,fft
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

def iir_notch(data, fs, frequency, quality=15., axis=-1):

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

def data_normalization(data,time,first_event,sampling_rate):
    data_before=data[np.where(time>first_event)[0][0]-30*sampling_rate:np.where(time>first_event)[0][0]]

    mean=np.mean(data_before)
    data_before=data_before-mean

    std=np.std(data_before)
    
    data_norm=(data-mean)/std
    print('normalizing data')
    return data_norm,time, data_before

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
def generate_epochs_with_first_event(events, time):
    
    
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
    print(len(data_door_before),len(data_door_after))

    return data_door_before, data_door_after

def extract_dig_data(data, time, dig_timestamp, sampling_rate):
    dig_index = np.where(time > dig_timestamp)[0][0]
    data_dig_after = data[dig_index:dig_index + 2 * sampling_rate]
    data_dig_before = data[dig_index - 2 * sampling_rate:dig_index]
    #print(f"Extracted dig data before from index {dig_index - 2 * sampling_rate} to {dig_index}")
    #print(f"Extracted dig data after from index {dig_index} to {dig_index + 2 * sampling_rate}")
    print(len(data_dig_before),len(data_dig_after))
    return data_dig_before, data_dig_after

def process_epoch(data, time, epochi, sampling_rate):
    white_trials_before = []
    white_trials_after = []
    correct_in_white_after = []
    incorrect_in_white_after = []
    correct_in_white_before = []
    incorrect_in_white_before = []
    
    black_trials_before = []
    black_trials_after = []
    correct_in_black_after = []
    incorrect_in_black_after = []
    correct_in_black_before = []
    incorrect_in_black_before = []
    
    no_context_trials_before = []
    no_context_trials_after = []
    correct_in_no_context_after = []
    incorrect_in_no_context_after = []
    correct_in_no_context_before = []
    incorrect_in_no_context_before = []
    
    if epochi.shape[0] > 1:
        epochi = epochi[:2]
        trial_timestamp = epochi[0][0]
        trial_type = epochi[0][1]
        print(f"Processing epoch with trial timestamp {trial_timestamp} and trial type {trial_type}")
        
        if trial_type == 119:
            data_trial_before, data_trial_after = extract_trial_data(data, time, trial_timestamp, sampling_rate)
            white_trials_before.append(data_trial_before)
            white_trials_after.append(data_trial_after)

            dig_type = epochi[1, 1]
            dig_timestamp = epochi[1, 0]
            data_dig_before, data_dig_after = extract_dig_data(data, time, dig_timestamp, sampling_rate)
            if dig_type == 49:
                correct_in_white_before.append(data_dig_before)
                correct_in_white_after.append(data_dig_after)
            elif dig_type == 48:
                incorrect_in_white_before.append(data_dig_before)
                incorrect_in_white_after.append(data_dig_after)
        elif trial_type == 98:
            data_trial_before, data_trial_after = extract_trial_data(data, time, trial_timestamp, sampling_rate)
            black_trials_before.append(data_trial_before)
            black_trials_after.append(data_trial_after)

            dig_type = epochi[1, 1]
            dig_timestamp = epochi[1, 0]
            data_dig_before, data_dig_after = extract_dig_data(data, time, dig_timestamp, sampling_rate)
            if dig_type == 49:
                correct_in_black_before.append(data_dig_before)
                correct_in_black_after.append(data_dig_after)
            elif dig_type == 48:
                incorrect_in_black_before.append(data_dig_before)
                incorrect_in_black_after.append(data_dig_after)
        elif trial_type == 120:
            data_trial_before, data_trial_after = extract_trial_data(data, time, trial_timestamp, sampling_rate)
            no_context_trials_before.append(data_trial_before)
            no_context_trials_after.append(data_trial_after)

            dig_type = epochi[1, 1]
            dig_timestamp = epochi[1, 0]
            data_dig_before, data_dig_after = extract_dig_data(data, time, dig_timestamp, sampling_rate)
            if dig_type == 49:
                correct_in_no_context_before.append(data_dig_before)
                correct_in_no_context_after.append(data_dig_after)
            elif dig_type == 48:
                incorrect_in_no_context_before.append(data_dig_before)
                incorrect_in_no_context_after.append(data_dig_after)

    return (white_trials_before, black_trials_before, white_trials_after, black_trials_after,
            no_context_trials_before, no_context_trials_after,
            correct_in_white_before, incorrect_in_white_before,
            correct_in_white_after, incorrect_in_white_after,
            correct_in_black_before, incorrect_in_black_before,
            correct_in_black_after, incorrect_in_black_after,
            correct_in_no_context_before, incorrect_in_no_context_before,
            correct_in_no_context_after, incorrect_in_no_context_after)


def pad_sequences(sequences, maxlen):
    padded_sequences = np.zeros((len(sequences), maxlen))
    for i, seq in enumerate(sequences):
        padded_sequences[i, :len(seq)] = seq
    print(f"Padded sequences to max length {maxlen}")
    return padded_sequences

def data_events_extract(data, time, epochs, sampling_rate):
    all_white_trials_before = []
    all_black_trials_before = []
    all_no_context_trials_before = []
    all_white_trials_after = []
    all_black_trials_after = []
    all_no_context_trials_after = []
    all_correct_in_white_before = []
    all_incorrect_in_white_before = []
    all_correct_in_white_after = []
    all_incorrect_in_white_after = []
    all_correct_in_black_before = []
    all_incorrect_in_black_before = []
    all_correct_in_black_after = []
    all_incorrect_in_black_after = []
    all_correct_in_no_context_before = []
    all_incorrect_in_no_context_before = []
    all_correct_in_no_context_after = []
    all_incorrect_in_no_context_after = []

    for epochi in epochs:
        print(f"Processing epoch: {epochi}")
        (white_trials_before, black_trials_before, white_trials_after, black_trials_after,
         no_context_trials_before, no_context_trials_after,
         correct_in_white_before, incorrect_in_white_before,
         correct_in_white_after, incorrect_in_white_after,
         correct_in_black_before, incorrect_in_black_before,
         correct_in_black_after, incorrect_in_black_after,
         correct_in_no_context_before, incorrect_in_no_context_before,
         correct_in_no_context_after, incorrect_in_no_context_after) = process_epoch(data, time, epochi, sampling_rate)

        all_white_trials_before.extend(white_trials_before)
        all_black_trials_before.extend(black_trials_before)
        all_no_context_trials_before.extend(no_context_trials_before)
        all_white_trials_after.extend(white_trials_after)
        all_black_trials_after.extend(black_trials_after)
        all_no_context_trials_after.extend(no_context_trials_after)
        all_correct_in_white_before.extend(correct_in_white_before)
        all_incorrect_in_white_before.extend(incorrect_in_white_before)
        all_correct_in_white_after.extend(correct_in_white_after)
        all_incorrect_in_white_after.extend(incorrect_in_white_after)
        all_correct_in_black_before.extend(correct_in_black_before)
        all_incorrect_in_black_before.extend(incorrect_in_black_before)
        all_correct_in_black_after.extend(correct_in_black_after)
        all_incorrect_in_black_after.extend(incorrect_in_black_after)
        all_correct_in_no_context_before.extend(correct_in_no_context_before)
        all_incorrect_in_no_context_before.extend(incorrect_in_no_context_before)
        all_correct_in_no_context_after.extend(correct_in_no_context_after)
        all_incorrect_in_no_context_after.extend(incorrect_in_no_context_after)

    maxlen = max(
        max((len(seq) for seq in all_correct_in_white_before), default=0),
        max((len(seq) for seq in all_incorrect_in_white_before), default=0),
        max((len(seq) for seq in all_correct_in_white_after), default=0),
        max((len(seq) for seq in all_incorrect_in_white_after), default=0),
        max((len(seq) for seq in all_correct_in_black_before), default=0),
        max((len(seq) for seq in all_incorrect_in_black_before), default=0),
        max((len(seq) for seq in all_correct_in_black_after), default=0),
        max((len(seq) for seq in all_incorrect_in_black_after), default=0),
        max((len(seq) for seq in all_correct_in_no_context_before), default=0),
        max((len(seq) for seq in all_incorrect_in_no_context_before), default=0),
        max((len(seq) for seq in all_correct_in_no_context_after), default=0),
        max((len(seq) for seq in all_incorrect_in_no_context_after), default=0)
    )
    print(f"Max length for padding: {maxlen}")

    all_correct_in_white_before = pad_sequences(all_correct_in_white_before, maxlen)
    all_incorrect_in_white_before = pad_sequences(all_incorrect_in_white_before, maxlen)
    all_correct_in_white_after = pad_sequences(all_correct_in_white_after, maxlen)
    all_incorrect_in_white_after = pad_sequences(all_incorrect_in_white_after, maxlen)
    all_correct_in_black_before = pad_sequences(all_correct_in_black_before, maxlen)
    all_incorrect_in_black_before = pad_sequences(all_incorrect_in_black_before, maxlen)
    all_correct_in_black_after = pad_sequences(all_correct_in_black_after, maxlen)
    all_incorrect_in_black_after = pad_sequences(all_incorrect_in_black_after, maxlen)
    all_correct_in_no_context_before = pad_sequences(all_correct_in_no_context_before, maxlen)
    all_incorrect_in_no_context_before = pad_sequences(all_incorrect_in_no_context_before, maxlen)
    all_correct_in_no_context_after = pad_sequences(all_correct_in_no_context_after, maxlen)
    all_incorrect_in_no_context_after = pad_sequences(all_incorrect_in_no_context_after, maxlen)

    print("Data extraction and padding complete")
    return (np.array(all_white_trials_before), np.array(all_black_trials_before), np.array(all_no_context_trials_before),
            np.array(all_white_trials_after), np.array(all_black_trials_after), np.array(all_no_context_trials_after),
            all_correct_in_white_before, all_incorrect_in_white_before,
            all_correct_in_white_after, all_incorrect_in_white_after,
            all_correct_in_black_before, all_incorrect_in_black_before,
            all_correct_in_black_after, all_incorrect_in_black_after,
            all_correct_in_no_context_before, all_incorrect_in_no_context_before,
            all_correct_in_no_context_after, all_incorrect_in_no_context_after)

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