import numpy as np
import mne_connectivity

def convert_epoch_to_coherence_density(epoch, fmin=1, fmax=100, tanh_norm=True):

    freqs = np.arange(fmin,fmax)
    n_cycles = freqs/3
    con=mne_connectivity.spectral_connectivity_time(epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,faverage=False, mode='multitaper',mt_bandwidth = 3, verbose=False, n_cycles=n_cycles,freqs=freqs)
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
                coherence= coh[0,i,j,:]
                if tanh_norm:
                    coherence=np.arctanh(coherence)  # Convert to Fisher Z-score
                aon_vhp_con.append(coherence)
                #print(coh[0,i,j,:])
            else:
                continue
    if aon_vhp_con==[]:
        print('no coherence found')
    else:
        #print(aon_vhp_con)
        aon_vhp_con_mean=np.mean(aon_vhp_con, axis=0)
        #print(aon_vhp_con_mean, 'coherenece')
            
    return aon_vhp_con_mean


def convert_epoch_to_coherence(epoch, tanh_norm = True):
    band_dict={'beta':[12,30],'gamma':[30,80],'total':[1,100], 'theta':[4,12]}
    coherence_dict={}
    for band in band_dict.keys():

        fmin=band_dict[band][0]
        fmax=band_dict[band][1]
        freqs = np.arange(fmin,fmax)
        n_cycles = freqs / 3
        #print(n_cycles)
        con=mne_connectivity.spectral_connectivity_epochs(epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,faverage=True, mode='cwt_morlet', verbose=False, cwt_n_cycles=n_cycles,cwt_freqs=freqs)
        coh = con.get_data(output='dense')
        #print(coh)
        indices = con.names
        #print(indices)
        aon_vhp_con=[]
        print(coh.shape)
        for i in range(coh.shape[0]):
            for j in range(coh.shape[1]):
                #print(i,j)
                if 'AON' in indices[j] and 'vHp' in indices[i]:
                    print('AON and vHp found')
                    coherence = coh[i,j,0,:]
                    if tanh_norm:
                        coherence=np.arctanh(coherence)  # Convert to Fisher Z-score
                    aon_vhp_con.append(np.mean(coherence))
                    #print('freqs averaged',coh[i,j,0,:].shape)
                    #print(coh[0,i,j,:])
                else:
                    continue
        if aon_vhp_con==[]:
            print('no coherence found')
        else:
            #print(aon_vhp_con)
            aon_vhp_con_mean=np.mean(aon_vhp_con, axis=0)
            #print(aon_vhp_con_mean, 'coherenece')
            coherence_dict[band]=aon_vhp_con_mean
    return coherence_dict

def convert_epoch_to_coherence_mt(epoch, tanh_norm = True):
    band_dict={'beta':[12,30],'gamma':[30,80],'total':[1,100], 'theta':[4,12]}
    coherence_dict={}
    for band in band_dict.keys():

        fmin=band_dict[band][0]
        fmax=band_dict[band][1]
        freqs = np.arange(fmin,fmax)
        #print(n_cycles)
        con=mne_connectivity.spectral_connectivity_epochs(epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,faverage=True, mode='multitaper',mt_bandwidth = 2.8,mt_adaptive=True, mt_low_bias=True, verbose=False, n_jobs=-1)
        coh = con.get_data(output='dense')
        #print(coh)
        indices = con.names
        #print(indices)
        aon_vhp_con=[]
        print(coh.shape)
        for i in range(coh.shape[0]):
            for j in range(coh.shape[1]):
                #print(i,j)
                if 'AON' in indices[j] and 'vHp' in indices[i]:
                    print('AON and vHp found')
                    coherence = coh[i,j,:]
                    if tanh_norm:
                        coherence=np.arctanh(coherence)  # Convert to Fisher Z-score
                    aon_vhp_con.append(np.mean(coherence))
                    #print('freqs averaged',coh[i,j,0,:].shape)
                    #print(coh[0,i,j,:])
                else:
                    continue
        if aon_vhp_con==[]:
            print('no coherence found')
        else:
            #print(aon_vhp_con)
            aon_vhp_con_mean=np.mean(aon_vhp_con, axis=0)
            #print(aon_vhp_con_mean, 'coherenece')
            coherence_dict[band]=aon_vhp_con_mean
    return coherence_dict

def convert_epoch_to_coherence_fourier(epoch, tanh_norm = True):
    band_dict={'beta':[12,30],'gamma':[30,80],'total':[1,100], 'theta':[4,12]}
    coherence_dict={}
    for band in band_dict.keys():

        fmin=band_dict[band][0]
        fmax=band_dict[band][1]
        freqs = np.arange(fmin,fmax)
        n_cycles = freqs / 3
        #print(n_cycles)
        con=mne_connectivity.spectral_connectivity_epochs(epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,faverage=True, mode='fourier', verbose=False)
        coh = con.get_data(output='dense')
        #print(coh)
        indices = con.names
        #print(indices)
        aon_vhp_con=[]
        print(coh.shape)
        for i in range(coh.shape[0]):
            for j in range(coh.shape[1]):
                #print(i,j)
                if 'AON' in indices[j] and 'vHp' in indices[i]:
                    print('AON and vHp found')
                    coherence = coh[i,j,:]
                    if tanh_norm:
                        coherence=np.arctanh(coherence)  # Convert to Fisher Z-score
                    aon_vhp_con.append(np.mean(coherence))
                    #print('freqs averaged',coh[i,j,0,:].shape)
                    #print(coh[0,i,j,:])
                else:
                    continue
        if aon_vhp_con==[]:
            print('no coherence found')
        else:
            #print(aon_vhp_con)
            aon_vhp_con_mean=np.mean(aon_vhp_con, axis=0)
            #print(aon_vhp_con_mean, 'coherenece')
            coherence_dict[band]=aon_vhp_con_mean
    return coherence_dict

def convert_epoch_to_coherence_time(epoch, tanh_norm = True):
    band_dict={'beta':[12,30],'gamma':[30,80],'total':[1,100], 'theta':[4,12]}
    coherence_dict={}
    for band in band_dict.keys():

        fmin=band_dict[band][0]
        fmax=band_dict[band][1]
        freqs = np.arange(fmin,fmax)
        n_cycles = freqs / 3
        #print(n_cycles)
        con=mne_connectivity.spectral_connectivity_time(epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,faverage=True, mode='cwt_morlet', verbose=False, n_cycles=n_cycles,freqs=freqs, average=True)
        coh = con.get_data(output='dense')
        #print(coh)
        indices = con.names
        #print(indices)
        aon_vhp_con=[]
        print(coh.shape)
        for i in range(coh.shape[1]):
            for j in range(coh.shape[2]):
                #print(i,j)
                if 'AON' in indices[j] and 'vHp' in indices[i]:
                    #print('AON and vHp found')
                    coherence = coh[i,j,0]
                    if tanh_norm:
                        coherence=np.arctanh(coherence)  # Convert to Fisher Z-score
                    aon_vhp_con.append(coherence)
                    #print('freqs averaged',coh[i,j,0,:].shape)
                    #print(coh[0,i,j,:])
                else:
                    continue
        if aon_vhp_con==[]:
            print('no coherence found')
        else:
            #print(aon_vhp_con)
            aon_vhp_con_mean=np.mean(aon_vhp_con, axis=0)
            #print(aon_vhp_con_mean, 'coherenece')
            coherence_dict[band]=aon_vhp_con_mean
    return coherence_dict

def convert_epoch_to_phase_behavior(epoch, band_start, band_end):
    fmin = band_start
    fmax = band_end
    freqs = np.arange(fmin, fmax)
    n_cycles = freqs / 3
    con = mne_connectivity.spectral_connectivity_time(
        epoch, method='pli', sfreq=int(2000), fmin=fmin, fmax=fmax,
        faverage=True, mode='multitaper',mt_bandwidth=2.8, verbose=False, n_cycles=n_cycles, freqs=freqs
    )
    print(con)
    coh = con.get_data(output='dense')
    print(coh.shape, 'coherence shape')  # Output shape (times, channels, channels, frequencies)

    indices = con.names
    aon_vhp_con = []
    print(indices)
    for i in range(coh.shape[1]):
        for j in range(coh.shape[2]):
            if 'AON' in indices[i] and 'vHp' in indices[j]:
                print('AON and vHp found')
                coherence= coh[0, i, j, :]
                print(coherence)
                aon_vhp_con.append(coherence)

    if not aon_vhp_con:  # If the list is empty
        print('No coherence found')
        aon_vhp_con_mean = np.zeros_like(freqs)  # Assign a default value (e.g., zeros)
    else:
        aon_vhp_con_mean = np.mean(aon_vhp_con, axis=0)

    return aon_vhp_con_mean[0]




def convert_epoch_to_coherence_behavior(epoch, band_start, band_end, tanh_norm = True):
    fmin = band_start
    fmax = band_end
    freqs = np.arange(fmin, fmax)
    n_cycles = freqs / 3
    con = mne_connectivity.spectral_connectivity_time(
        epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,
        faverage=True, mode='multitaper',mt_bandwidth=2.8, verbose=False, n_cycles=n_cycles, freqs=freqs, n_jobs=-1
    )
    coh = con.get_data(output='dense')
    indices = con.names
    aon_vhp_con = []
    print(indices)
    for i in range(coh.shape[1]):
        for j in range(coh.shape[2]):
            if 'AON' in indices[i] and 'vHp' in indices[j]:
                print('AON and vHp found')
                coherence= coh[0, i, j, :]

                if tanh_norm:
                    coherence = np.arctanh(coherence)  # Convert to Fisher Z-score
                print(coherence)
                aon_vhp_con.append(coherence)

    if not aon_vhp_con:  # If the list is empty
        print('No coherence found')
        aon_vhp_con_mean = np.zeros_like(freqs)  # Assign a default value (e.g., zeros)
    else:
        aon_vhp_con_mean = np.mean(aon_vhp_con, axis=0)

    return aon_vhp_con_mean[0]

def convert_epoch_to_coherence_baseline(epoch, band_start, band_end, tanh_norm = True):
    fmin = band_start
    fmax = band_end
    freqs = np.arange(fmin, fmax)
    n_cycles = freqs / 3
    # con = mne_connectivity.spectral_connectivity_time(
    #     epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,
    #     faverage=True, mode='multitaper',mt_bandwidth=3, verbose=False, freqs=freqs, n_cycles=n_cycles
    # )
    con = mne_connectivity.spectral_connectivity_time(
        epoch, method='coh', sfreq=int(2000), fmin=fmin, fmax=fmax,
        faverage=True, mode='multitaper',mt_bandwidth=2.8, verbose=False, freqs=freqs, n_cycles=n_cycles
    )
    coh = con.get_data(output='dense')
    #print(coh[0,2,0,:])
    #print(coh.shape, 'coherence shape') # Output shape (times, channels, channels, frequencies)
    indices = con.names
    aon_vhp_con = []
    #print(indices)
    for i in range(coh.shape[1]):
        for j in range(coh.shape[2]):
            if 'AON' in indices[i] and 'vHp' in indices[j]:
                #print('AON and vHp found', i, j)
                coherence= coh[0,j, i,:]
                if tanh_norm:
                    coherence = np.arctanh(coherence)  # Convert to Fisher Z-score
                #print(coherence)
                aon_vhp_con.append(coherence)

    if not aon_vhp_con:  # If the list is empty
        #print('No coherence found')
        aon_vhp_con_mean = np.zeros_like(freqs)  # Assign a default value (e.g., zeros)
    else:
        aon_vhp_con_mean = np.mean(aon_vhp_con, axis=0)

    return aon_vhp_con_mean[0]

def convert_epoch_to_coherence_cwt(epoch, fmin=1, fmax=100, tanh_norm=True):
    fmin=fmin
    fmax=fmax
    fs=2000
    freqs = np.arange(fmin,fmax)
    n_cycles = freqs/3

    con = mne_connectivity.spectral_connectivity_epochs(epoch, method='coh', sfreq=int(fs),
                                            mode='cwt_morlet', cwt_freqs=freqs,
                                            cwt_n_cycles=n_cycles, verbose=False, fmin=fmin, fmax=fmax, faverage=False)
    coh = con.get_data(output='dense')
    indices = con.names
    

    for i in range(coh.shape[0]):
        for j in range(coh.shape[1]):
            if 'AON' in indices[j] and 'vHp' in indices[i]:
                coherence= coh[i,j,:,:]
                if tanh_norm:
                    coherence=np.arctanh(coherence)
    
    return coherence