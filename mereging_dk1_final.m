% MATLAB script to concatenate two Spike2 .mat files with nested channel structures
% Designed for files exported from Spike2 with 2000 Hz sampling rate

clear; clc;

%% File paths - MODIFY THESE
file1_path = "C:\Users\sinha\Dropbox\CPLab\all_data_mat_250825\20230817_dk1_BW_context_os2_day1_pt1.mat";
file2_path = "C:\Users\sinha\Dropbox\CPLab\all_data_mat_250825\20230817_dk1_BW_context_os2_day1_pt2.mat";
output_path = 'C:\Users\sinha\Dropbox\CPLab\all_data_mat_250825\merged_data_20230817_dk1_BW_context_os2_day1.mat';

%% Load the data files
fprintf('Loading first file...\n');
file_1 = load(file1_path);

fprintf('Loading second file...\n');
file_2 = load(file2_path);

%% Get channel names
channels1 = fieldnames(file_1);
channels2 = fieldnames(file_2);

% Find common channels
common_channels = intersect(channels1, channels2);

fprintf('Found %d common channels:\n', length(common_channels));
for i = 1:length(common_channels)
    fprintf('  %s\n', common_channels{i});
end

%% Find an LFP channel for timing reference (skip event channels like Keyboard)
lfp_channels = common_channels(contains(common_channels, 'LFP'));
if isempty(lfp_channels)
    % If no LFP channels, look for other analog channels
    analog_channels = setdiff(common_channels, {'Keyboard'});
    if ~isempty(analog_channels)
        reference_channel = analog_channels{1};
    else
        error('No suitable analog channels found for timing reference');
    end
else
    reference_channel = lfp_channels{1};
end

fprintf('Using %s as timing reference channel\n', reference_channel);

% Debug: Check what fields are available
fprintf('Available fields in %s:\n', reference_channel);
disp(fieldnames(file_1.(reference_channel)));

% Get timing info from reference channel
chan_struct = file_1.(reference_channel);
file1_end_time = chan_struct.times(end);
file2_start_time = file_2.(reference_channel).times(1);

% Get sampling interval
if isfield(chan_struct, 'interval')
    sampling_interval = chan_struct.interval;
elseif isfield(chan_struct, 'Interval')
    sampling_interval = chan_struct.Interval;
else
    % Calculate from time vector if interval field not found
    sampling_interval = chan_struct.times(2) - chan_struct.times(1);
    fprintf('Interval field not found, calculated from times: %.6f\n', sampling_interval);
end

fprintf('\nTiming information:\n');
fprintf('  File 1 ends at: %.6f seconds\n', file1_end_time);
fprintf('  File 2 starts at: %.6f seconds\n', file2_start_time);
fprintf('  Sampling interval: %.6f seconds (%.0f Hz)\n', sampling_interval, 1/sampling_interval);
fprintf('  File 1 duration: %.2f seconds (%d samples)\n', file1_end_time, file_1.(reference_channel).length);
fprintf('  File 2 duration: %.2f seconds (%d samples)\n', file_2.(reference_channel).times(end), file_2.(reference_channel).length);

%% Initialize merged data structure
merged_data = struct();

%% Process each channel
fprintf('\nConcatenating channels...\n');
for i = 1:length(common_channels)
    channel_name = common_channels{i};
    
    % Get data from both files
    chan1 = file_1.(channel_name);
    chan2 = file_2.(channel_name);
    
    % Create merged channel structure
    merged_channel = struct();
    
    % Copy metadata from first file (handle different channel types)
    merged_channel.title = chan1.title;
    merged_channel.comment = chan1.comment;
    
    % Copy fields that exist (different channels have different fields)
    fields_to_copy = {'interval', 'scale', 'offset', 'units', 'start', 'resolution'};
    for j = 1:length(fields_to_copy)
        field_name = fields_to_copy{j};
        if isfield(chan1, field_name)
            merged_channel.(field_name) = chan1.(field_name);
        end
    end
    
    % Handle different data types
    if isfield(chan1, 'values')
        % Analog channel (LFP, Ref, Respirat)
        merged_channel.values = [chan1.values; chan2.values];
        merged_channel.times = [chan1.times; chan2.times - chan2.times(1) + chan1.times(end) + sampling_interval];
    elseif isfield(chan1, 'codes')
        % Event channel (Keyboard)
        merged_channel.codes = [chan1.codes; chan2.codes];
        merged_channel.times = [chan1.times; chan2.times + file1_end_time];
    else
        % Unknown channel type, try to concatenate whatever arrays exist
        fields = fieldnames(chan1);
        for j = 1:length(fields)
            field_name = fields{j};
            if isnumeric(chan1.(field_name)) && length(chan1.(field_name)) > 1
                merged_channel.(field_name) = [chan1.(field_name); chan2.(field_name)];
            end
        end
    end
    
    merged_channel.length = length(merged_channel.times);
    
    % Store in merged data structure
    merged_data.(channel_name) = merged_channel;
    
    fprintf('  %s: %d + %d = %d samples\n', ...
        channel_name, chan1.length, chan2.length, merged_channel.length);
end

%% Add concatenation metadata
merged_data.concatenation_info = struct();
merged_data.concatenation_info.file1_path = char(file1_path);
merged_data.concatenation_info.file2_path = char(file2_path);
merged_data.concatenation_info.file1_duration = file1_end_time;
merged_data.concatenation_info.file2_duration = file_2.(reference_channel).times(end);
merged_data.concatenation_info.total_duration = merged_data.(reference_channel).times(end);
merged_data.concatenation_info.sampling_rate = 1/sampling_interval;
merged_data.concatenation_info.concatenation_time = char(datetime('now'));

% Sample counts for each file
merged_data.concatenation_info.file1_samples = file_1.(reference_channel).length;
merged_data.concatenation_info.file2_samples = file_2.(reference_channel).length;
merged_data.concatenation_info.total_samples = merged_data.(reference_channel).length;

%% Save merged data as HDF5-compatible .mat file
fprintf('\nSaving merged data to %s...\n', output_path);
save(output_path, '-struct', 'merged_data', '-v7.3');

fprintf('\n=== CONCATENATION COMPLETE ===\n');
fprintf('Output file: %s\n', output_path);
fprintf('Total channels: %d\n', length(common_channels));
fprintf('Total duration: %.2f seconds\n', merged_data.concatenation_info.total_duration);
fprintf('Total samples per channel: %d\n', merged_data.concatenation_info.total_samples);
fprintf('File size: HDF5-compatible .mat format\n');


%% Display channel information
fprintf('\nMerged channels:\n');
channels = fieldnames(merged_data);
data_channels = setdiff(channels, {'concatenation_info'});
for i = 1:length(data_channels)
    ch = data_channels{i};
    % Handle channels that may not have units field
    if isfield(merged_data.(ch), 'units')
        units_str = merged_data.(ch).units;
    else
        units_str = 'N/A';
    end
    
    fprintf('  %s: %d samples, %.2f sec, %s\n', ...
        merged_data.(ch).title, merged_data.(ch).length, ...
        merged_data.(ch).times(end), units_str);
end

%% Quick verification plot (optional)
fprintf('\nGenerating verification plot...\n');
figure('Name', 'Concatenation Verification');

% Plot first LFP channel to verify concatenation
if isfield(merged_data, 'LFP1_AON')
    plot_channel = 'LFP1_AON';
elseif isfield(merged_data, 'LFP1_vHp')
    plot_channel = 'LFP1_vHp';
else
    plot_channel = data_channels{1};
end

times = merged_data.(plot_channel).times;
values = merged_data.(plot_channel).values;

subplot(2,1,1);
plot(times, values);
xlabel('Time (s)');
ylabel(['Amplitude (' merged_data.(plot_channel).units ')']);
title([merged_data.(plot_channel).title ' - Full Concatenated Recording']);
grid on;

% Zoom in on concatenation point
concat_point = merged_data.concatenation_info.file1_duration;
subplot(2,1,2);
window = 5; % seconds around concatenation point
idx_start = find(times >= concat_point - window, 1);
idx_end = find(times <= concat_point + window, 1, 'last');

plot(times(idx_start:idx_end), values(idx_start:idx_end));
hold on;
xline(concat_point, 'r--', 'LineWidth', 2, 'Label', 'Concatenation Point');
xlabel('Time (s)');
ylabel(['Amplitude (' merged_data.(plot_channel).units ')']);
title('Zoomed View Around Concatenation Point');
grid on;

fprintf('Concatenation complete! Check the plot to verify continuity.\n');