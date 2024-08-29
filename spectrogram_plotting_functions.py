import os

def spectrogram_ind_trial(compiled_data_all_epochs, mouse_id, task, sampling_rate, savepath, base_name):
    """
    This plots the spectrogram for each trial in the compiled_data_all_epochs dataframe.
    Each row in plot represents a different event (total_pre_door, total_post_door, total_pre_odor, total_post_odor).
    Each column in plot represents a different trial.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    if 'channel' in compiled_data_all_epochs.columns:
        channels=np.unique(compiled_data_all_epochs['channel'])

        #channels=['LFP4_AON']
        for channeli in channels:
            
            channel_data = compiled_data_all_epochs[compiled_data_all_epochs['channel'] == channeli]
            total_bands=['total_pre_door', 'total_post_door', 'total_pre_odor', 'total_post_odor']

            # Determine the number of trials
            num_trials = len(channel_data)

            # Create a superfigure with 2 rows and num_trials columns
            fig, axs = plt.subplots(4, num_trials, figsize=(5*num_trials, 5*len(total_bands)),sharex=True,sharey=True)
            fig.suptitle('{} {} {}'.format(mouse_id, task, channeli), fontsize=25, fontweight='bold')

            # Iterate through each trial
            for trial_idx, (i, row) in enumerate(channel_data.iterrows()):
                for band_idx, bandi in enumerate(total_bands):
                    ax = axs[band_idx, trial_idx]
                    
                    # Generate the spectrogram
                    Pxx, freqs, bins, im = ax.specgram(row[bandi], Fs=sampling_rate, cmap="rainbow", NFFT=512, noverlap=256)

                    # Find the maximum intensity for each frequency bin
                    max_intensity_indices = np.argmax(Pxx, axis=1)
                    max_intensities = Pxx[np.arange(Pxx.shape[0]), max_intensity_indices]

                    # Find the corresponding frequencies
                    most_intense_frequencies = freqs[np.argmax(max_intensities)]

                    # Print the most intense frequencies
                    print(f"Most intense frequencies for trial {i}, {bandi} (Hz):", most_intense_frequencies)
                    
                    # Plot the spectrogram
                    ax.set_title(f'Trial {i}')
                    # Add labels in front of each row
        
                    # ax.set_xlabel("TIME")
                    # ax.set_ylabel("FREQUENCY (Hz)")
                    #ax.set_ylim(0, 100)
                    # # ax.set_yscale('log')
                    # cbar = fig.colorbar(im, ax=ax)
                    # cbar.set_label('Intensity [dB]')

                    # # Plot horizontal bands for theta, beta, and gamma oscillations
                    # ax.axhline(y=4, color='blue', linestyle='--', label='Theta (4-8 Hz)')
                    # ax.axhline(y=8, color='blue', linestyle='--')
                    # ax.axhline(y=12, color='green', linestyle='--', label='Beta (12-30 Hz)')
                    # ax.axhline(y=30, color='green', linestyle='--')
                    # ax.axhline(y=31, color='red', linestyle='--', label='Gamma (30-100 Hz)')
                    # ax.axhline(y=100, color='red', linestyle='--')
            for band_idx, bandi in enumerate(total_bands):
                fig.text(0.0, 0.875 - band_idx * 0.25, bandi, va='center', ha='center', rotation='vertical', fontsize=12, fontweight='bold')
            # Add a legend to the last subplot in each row
            axs[0, -1].legend()
            axs[1, -1].legend()

            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            fig.savefig(os.path.join(savepath,f' {base_name} {channeli} spectrogram no_y_lim.png'), dpi=100, bbox_inches='tight')
            # Display the plot
            plt.show()