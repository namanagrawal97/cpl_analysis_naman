import numpy as np
from scipy.signal import iirnotch, filtfilt, butter, sosfiltfilt

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
    print(f'Global start time: {global_start_time}, Global end time: {global_end_time}')
    return global_start_time, global_end_time

def pad_raw_data_raw_time(data, time, global_start_time, global_end_time, sampling_rate=2000):
    
    total_points = int((global_end_time - global_start_time) * sampling_rate) + 1
    print(total_points)
    padded_data = np.zeros(total_points)
    start_index = int((time[0] - global_start_time) * sampling_rate)
    end_index = start_index + len(data)
        # Safety check
    if end_index > total_points:
        print(f"Warning: Data length ({len(data)}) exceeds available space. Truncating.")
        end_index = total_points
        data = data[:total_points - start_index]  # Truncate data
    print(f'start_index: {start_index}, end_index: {end_index}')
    padded_data[start_index:end_index] = data
    padded_time = np.linspace(global_start_time, global_end_time, total_points)
    return padded_data, padded_time

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


def extract_event_data(data, time, event_timestamp, sampling_rate, truncation_time):
    event_index = np.where(time > event_timestamp)[0][0]
    bound=int(float(truncation_time) * sampling_rate)
    data_event_before = data[event_index- bound :event_index]
    data_event_after = data[event_index:event_index + bound]
    return data_event_before, data_event_after

def extract_complete_trial_data(data, time, door_timestamp,dig_timestamp, sampling_rate, truncation_time):
    door_index = np.where(time > door_timestamp)[0][0]
    dig_index = np.where(time > dig_timestamp)[0][0]
    bound=int(float(truncation_time) * sampling_rate)
    data_complete_trial = data[door_index- bound:dig_index+ bound]
    return np.array(data_complete_trial,dtype=float)

def iir_notch(data, fs, frequency, quality=30., axis=-1):

    norm_freq = frequency/(fs/2)
    b, a = iirnotch(norm_freq, quality)
    y = filtfilt(b, a, data, padlen=0, axis=axis)
    print('notch filter applied')
    return y
def bandpass(data, fs, start_freq, end_freq, order=30):
    b, a = butter(N=order,Wn=[start_freq, end_freq], fs=fs, btype='bandpass')
    y = filtfilt(b, a, data, padlen=0)
    print(f'band pass filter applied between {start_freq}Hz and {end_freq}Hz')
    return y

def highpass(data, fs, start_freq, order=30):
    b, a = butter(N=order,Wn=start_freq, fs=fs, btype='highpass')
    y = filtfilt(b, a, data, padlen=0)
    print(f'highpass filter applied from {start_freq}Hz')
    return y

def soshighpass(data, fs, start_freq, order=30):
    sos = butter(N=order, Wn=start_freq, fs=fs, btype='highpass', output='sos')
    y = sosfiltfilt(sos, data, padlen=0)
    print(f'highpass filter applied from {start_freq}Hz')
    return y

def zscore_event_data(data, baseline_std):
    data_mean=np.mean(data)
    data_zscored = (data - data_mean) / baseline_std
    return data_zscored
