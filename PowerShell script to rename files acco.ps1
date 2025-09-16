# PowerShell script to rename files according to standardized schema
# Author: GitHub Copilot
# Date: September 15, 2025

# Set the base directory - Updated to match actual structure
$baseDir = "D:\Dropbox\CPLab\results\paper\data sheets and figures 250915"

# Create log file for tracking renames
$logFile = "c:\Users\CPLab\cpl_analysis_naman\file_rename_log.txt"
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Initialize log file
"File Rename Log - $timestamp" | Out-File -FilePath $logFile -Encoding UTF8
"=" * 50 | Out-File -FilePath $logFile -Append
"" | Out-File -FilePath $logFile -Append

# Function to log rename operations
function Log-Rename {
    param($oldName, $newName, $folder)
    $logEntry = "FOLDER: $folder`nOLD: $oldName`nNEW: $newName`n" + "-" * 30
    Write-Host "Renaming: $oldName -> $newName" -ForegroundColor Green
    $logEntry | Out-File -FilePath $logFile -Append
}

# Function to safely rename files
function Rename-FileIfExists {
    param($folderPath, $oldName, $newName)
    
    $oldPath = Join-Path $folderPath $oldName
    $newPath = Join-Path $folderPath $newName
    
    if (Test-Path $oldPath) {
        if (Test-Path $newPath) {
            Write-Warning "Target file already exists: $newName"
            "WARNING: Target file already exists: $newName" | Out-File -FilePath $logFile -Append
        } else {
            Rename-Item -Path $oldPath -NewName $newName -ErrorAction Stop
            Log-Rename $oldName $newName (Split-Path $folderPath -Leaf)
        }
    } else {
        Write-Warning "Source file not found: $oldName"
        "WARNING: Source file not found: $oldName in folder: $folderPath" | Out-File -FilePath $logFile -Append
    }
}

Write-Host "Starting file rename process..." -ForegroundColor Yellow
Write-Host "Log file: $logFile" -ForegroundColor Cyan

try {
    # Baseline coherence folder - standardize file extensions and names
    "BASELINE COHERENCE FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $baselineCombined = Join-Path $baseDir "baseline coherence\combined"
    Rename-FileIfExists $baselineCombined "baseline_coherence_density.xlsx" "coh_baseline_density_combined.xlsx"
    Rename-FileIfExists $baselineCombined "coherence_baseline_per_band_channelpair_combined.xlsx" "coh_baseline_perband_channelpair_combined.xlsx"
    
    $baselineNormalized = Join-Path $baseDir "baseline coherence\normalized"
    Rename-FileIfExists $baselineNormalized "baseline_coherence_density.png" "coh_baseline_density_normalized.png"
    Rename-FileIfExists $baselineNormalized "baseline_coherence_density.xlsx" "coh_baseline_density_normalized.xlsx"
    Rename-FileIfExists $baselineNormalized "coherence_baseline_per_band_channelpair_normalized.xlsx" "coh_baseline_perband_channelpair_normalized.xlsx"
    
    $baselineNonNormalized = Join-Path $baseDir "baseline coherence\non normalized"
    Rename-FileIfExists $baselineNonNormalized "baseline_coherence_density_notnormalized.png" "coh_baseline_density_nonnormalized.png"
    Rename-FileIfExists $baselineNonNormalized "baseline_coherence_density_notnormalized.xlsx" "coh_baseline_density_nonnormalized.xlsx"
    Rename-FileIfExists $baselineNonNormalized "coherence_baseline_per_band_channelpair_nonnormalized.xlsx" "coh_baseline_perband_channelpair_nonnormalized.xlsx"

    # Baseline power folder
    "BASELINE POWER FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $baselinePower = Join-Path $baseDir "baseline power"
    Rename-FileIfExists $baselinePower "baseline_power_per_band_multitaper.png" "pow_baseline_perband.png"
    Rename-FileIfExists $baselinePower "baseline_power_per_band_multitaper.xlsx" "pow_baseline_perband.xlsx"
    Rename-FileIfExists $baselinePower "baseline_psd_multitaper.csv" "pow_baseline_psd.csv"
    Rename-FileIfExists $baselinePower "baseline_psd_multitaper.png" "pow_baseline_psd.png"

    # Events power folder
    "EVENTS POWER FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $eventsPower = Join-Path $baseDir "events power"
    Rename-FileIfExists $eventsPower "events_power_perband_perchannel_multitaper.png" "pow_events_perband.png"
    Rename-FileIfExists $eventsPower "events_power_perband_perchannel_multitaper.xlsx" "pow_events_perband.xlsx"
    Rename-FileIfExists $eventsPower "events_power_spectral_density_multitaper.png" "pow_events_psd.png"
    Rename-FileIfExists $eventsPower "events_power_spectral_density_multitaper.xlsx" "pow_events_psd.xlsx"

    # Behavior coherence folder
    "BEHAVIOR COHERENCE FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $behaviorCoherence = Join-Path $baseDir "behavior coherence"
    Rename-FileIfExists $behaviorCoherence "behavior_coherence_compiled_data_df_truncated_1400.xlsx" "beh_dig_coh_compiled_700ms.xlsx"
    Rename-FileIfExists $behaviorCoherence "beta_coherence_vs_time_to_dig_1400.png" "beh_dig_beta_spectrogram_700ms.png"

    # Combined folders - rename to match schema
    "COMBINED FOLDERS UPDATES:" | Out-File -FilePath $logFile -Append
    
    $combined400ms = Join-Path $baseDir "events coherence\combined_400ms"
    Rename-FileIfExists $combined400ms "events_coherence_per_band_channelpair_800_combined.xlsx" "coh_events_perband_channelpair_combined_400ms.xlsx"
    Rename-FileIfExists $combined400ms "events_coherence_per_band_channelpair_800_shuffled_combined.xlsx" "coh_events_perband_channelpair_combined_400ms_shuffled.xlsx"
    
    $combined700ms = Join-Path $baseDir "events coherence\combined_700ms"
    Rename-FileIfExists $combined700ms "events_coherence_per_band_channelpair_1400_combined.xlsx" "coh_events_perband_channelpair_combined_700ms.xlsx"
    Rename-FileIfExists $combined700ms "events_coherence_per_band_channelpair_1400_shuffled_combined.xlsx" "coh_events_perband_channelpair_combined_700ms_shuffled.xlsx"

    # Normalized 400ms folder
    "NORMALIZED 400MS FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $normalized400ms = Join-Path $baseDir "events coherence\normalized_400ms"
    
    # PNG files
    Rename-FileIfExists $normalized400ms "aon_vhp_coherence_event_spectrogram_800_normalized.png" "coh_events_spectrogram_averaged_normalized_400ms.png"
    Rename-FileIfExists $normalized400ms "BWcontext_coherogram_per_experiment_around_dig_800_normalized.png" "coh_events_spectrogram_perexp_context_normalized_400ms.png"
    Rename-FileIfExists $normalized400ms "BWnocontext_coherogram_per_experiment_around_dig_800_normalized.png" "coh_events_spectrogram_perexp_nocontext_normalized_400ms.png"
    Rename-FileIfExists $normalized400ms "coherence_channelpair_800_normalized.png" "coh_events_perband_channelpair_normalized_400ms.png"
    Rename-FileIfExists $normalized400ms "coherence_channelpair_800_normalized_shuffled.png" "coh_events_perband_channelpair_normalized_400ms_shuffled.png"
    
    # Excel files
    Rename-FileIfExists $normalized400ms "coherogram_perexperiment_800_normalized.xlsx" "coh_events_spectrogram_perexp_normalized_400ms.xlsx"
    Rename-FileIfExists $normalized400ms "coherogram_values_800_normalized.xlsx" "coh_events_spectrogram_averaged_normalized_400ms.xlsx"
    Rename-FileIfExists $normalized400ms "events_coherence_per_band_channelpair_800_normalized.xlsx" "coh_events_perband_channelpair_normalized_400ms.xlsx"
    Rename-FileIfExists $normalized400ms "events_coherence_per_band_channelpair_800_normalized_shuffled.xlsx" "coh_events_perband_channelpair_normalized_400ms_shuffled.xlsx"

    # Normalized 700ms folder
    "NORMALIZED 700MS FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $normalized700ms = Join-Path $baseDir "events coherence\normalized_700ms"
    
    # PNG files
    Rename-FileIfExists $normalized700ms "aon_vhp_coherence_event_spectrogram_1400_normalized.png" "coh_events_spectrogram_averaged_normalized_700ms.png"
    Rename-FileIfExists $normalized700ms "BWcontext_coherogram_per_experiment_around_dig_normalized.png" "coh_events_spectrogram_perexp_context_normalized_700ms.png"
    Rename-FileIfExists $normalized700ms "BWnocontext_coherogram_per_experiment_around_dig_normalized.png" "coh_events_spectrogram_perexp_nocontext_normalized_700ms.png"
    Rename-FileIfExists $normalized700ms "coherence_channelpair_1400_normalized.png" "coh_events_perband_channelpair_normalized_700ms.png"
    Rename-FileIfExists $normalized700ms "coherence_channelpair_1400_normalized_shuffled.png" "coh_events_perband_channelpair_normalized_700ms_shuffled.png"
    
    # Excel files
    Rename-FileIfExists $normalized700ms "coherogram_perexperiment_1400_normalized.xlsx" "coh_events_spectrogram_perexp_normalized_700ms.xlsx"
    Rename-FileIfExists $normalized700ms "coherogram_values_1400_normalized.xlsx" "coh_events_spectrogram_averaged_normalized_700ms.xlsx"
    Rename-FileIfExists $normalized700ms "events_coherence_per_band_channelpair_1400_normalized.xlsx" "coh_events_perband_channelpair_normalized_700ms.xlsx"
    Rename-FileIfExists $normalized700ms "events_coherence_per_band_channelpair_1400_normalized_shuffled.xlsx" "coh_events_perband_channelpair_normalized_700ms_shuffled.xlsx"

    # Non-normalized 400ms folder (folder name is "not_normalized_400ms")
    "NON-NORMALIZED 400MS FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $nonnormalized400ms = Join-Path $baseDir "events coherence\not_normalized_400ms"
    
    # PNG files
    Rename-FileIfExists $nonnormalized400ms "aon_vhp_coherence_event_spectrogram_800_non-normalized.png" "coh_events_spectrogram_averaged_nonnormalized_400ms.png"
    Rename-FileIfExists $nonnormalized400ms "BWcontext_coherogram_per_experiment_around_dig_800_nonnormalized.png" "coh_events_spectrogram_perexp_context_nonnormalized_400ms.png"
    Rename-FileIfExists $nonnormalized400ms "BWnocontext_coherogram_per_experiment_around_dig_800_nonnormalized.png" "coh_events_spectrogram_perexp_nocontext_nonnormalized_400ms.png"
    Rename-FileIfExists $nonnormalized400ms "coherence_channelpair_800_nonnormalized.png" "coh_events_perband_channelpair_nonnormalized_400ms.png"
    Rename-FileIfExists $nonnormalized400ms "coherence_channelpair_800_nonnormalized_shuffled.png" "coh_events_perband_channelpair_nonnormalized_400ms_shuffled.png"
    
    # Excel files
    Rename-FileIfExists $nonnormalized400ms "coherogram_perexperiment_800_nonnormalized.xlsx" "coh_events_spectrogram_perexp_nonnormalized_400ms.xlsx"
    Rename-FileIfExists $nonnormalized400ms "coherogram_values_800_non-normalized.xlsx" "coh_events_spectrogram_averaged_nonnormalized_400ms.xlsx"
    Rename-FileIfExists $nonnormalized400ms "events_coherence_per_band_channelpair_800_nonnormalized.xlsx" "coh_events_perband_channelpair_nonnormalized_400ms.xlsx"
    Rename-FileIfExists $nonnormalized400ms "events_coherence_per_band_channelpair_800_nonnormalized_shuffled.xlsx" "coh_events_perband_channelpair_nonnormalized_400ms_shuffled.xlsx"

    # Non-normalized 700ms folder (folder name is "not_normalized_700ms")
    "NON-NORMALIZED 700MS FOLDER UPDATES:" | Out-File -FilePath $logFile -Append
    
    $nonnormalized700ms = Join-Path $baseDir "events coherence\not_normalized_700ms"
    
    # PNG files
    Rename-FileIfExists $nonnormalized700ms "aon_vhp_coherence_event_spectrogram_1400_non-normalized.png" "coh_events_spectrogram_averaged_nonnormalized_700ms.png"
    Rename-FileIfExists $nonnormalized700ms "BWcontext_coherogram_per_experiment_around_dig_1400_nonnormalized.png" "coh_events_spectrogram_perexp_context_nonnormalized_700ms.png"
    Rename-FileIfExists $nonnormalized700ms "BWnocontext_coherogram_per_experiment_around_dig_1400_nonnormalized.png" "coh_events_spectrogram_perexp_nocontext_nonnormalized_700ms.png"
    Rename-FileIfExists $nonnormalized700ms "coherence_channelpair_1400_nonnormalized.png" "coh_events_perband_channelpair_nonnormalized_700ms.png"
    Rename-FileIfExists $nonnormalized700ms "coherence_channelpair_1400_nonnormalized_shuffled.png" "coh_events_perband_channelpair_nonnormalized_700ms_shuffled.png"
    
    # Excel files
    Rename-FileIfExists $nonnormalized700ms "coherogram_perexperiment_1400_nonnormalized.xlsx" "coh_events_spectrogram_perexp_nonnormalized_700ms.xlsx"
    Rename-FileIfExists $nonnormalized700ms "coherogram_values_1400_non-normalized.xlsx" "coh_events_spectrogram_averaged_nonnormalized_700ms.xlsx"
    Rename-FileIfExists $nonnormalized700ms "events_coherence_per_band_channelpair_1400_nonnormalized.xlsx" "coh_events_perband_channelpair_nonnormalized_700ms.xlsx"
    Rename-FileIfExists $nonnormalized700ms "events_coherence_per_band_channelpair_1400_nonnormalized_shuffled.xlsx" "coh_events_perband_channelpair_nonnormalized_700ms_shuffled.xlsx"

    Write-Host "`nRename process completed successfully!" -ForegroundColor Green
    
} catch {
    Write-Error "An error occurred: $($_.Exception.Message)"
    "ERROR: $($_.Exception.Message)" | Out-File -FilePath $logFile -Append
}

# Final log entry
"" | Out-File -FilePath $logFile -Append
"Rename process completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" | Out-File -FilePath $logFile -Append

Write-Host "`nCheck the log file for details: $logFile" -ForegroundColor Cyan