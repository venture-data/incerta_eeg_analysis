from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import mne
import os
from io import BytesIO
import base64
from matplotlib.colors import LinearSegmentedColormap
from mne.viz import plot_topomap
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from mne import make_bem_model, make_bem_solution, make_forward_solution
from mne.dipole import fit_dipole
from mne.transforms import Transform
from openai import OpenAI
import re
from scipy.stats import kurtosis, skew


import json
from scipy.integrate import simpson  # For band power calculation


def detect_rectus_artifacts(raw_ica, start_time, duration=5):
    """
    Detect rectus artifacts (related to eye movements) in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of rectus artifacts.
    """

    # Focus on frontal channels (Fp1, Fp2) for rectus artifacts
    rectus_channels = ['Fp1', 'Fp2']
    
    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)
    
    # Apply a low-pass filter (1-3 Hz) to capture slow eye movements
    segment_filtered = segment.filter(1, 3, fir_design='firwin')
    
    # Get data from the frontal channels
    rectus_data = segment_filtered.copy().pick_channels(rectus_channels).get_data()
    
    # Set a threshold for detecting artifacts based on signal amplitude
    threshold = np.percentile(np.abs(rectus_data), 95)
    
    # Detect segments where the signal exceeds the threshold
    rectus_segments = []
    for i, channel_data in enumerate(rectus_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            rectus_segments.append((start_time_artifact, end_time_artifact))
    
    return rectus_segments
def detect_ecg_artifacts(raw_ica, start_time, duration=5):
    """
    Detect ECG/cardiac artifacts in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of ECG artifacts.
    """

    # Define the ECG artifact frequency range (around 1-2 Hz)
    freq_range = (1, 2)
    
    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)
    
    # Apply a band-pass filter within the ECG artifact frequency range
    segment_filtered = segment.filter(freq_range[0], freq_range[1], fir_design='firwin')
    
    # Identify channels most likely to have ECG artifacts
    ecg_channels = ['T3', 'T4', 'Cz']
    
    # Get data from those channels
    ecg_data = segment_filtered.copy().pick_channels(ecg_channels).get_data()
    
    # Use peak detection to find consistent heartbeats
    ecg_segments = []
    for channel_data in ecg_data:
        peaks, _ = find_peaks(channel_data, distance=segment.info['sfreq'] * 0.6)  # Assuming average heartbeat is 60-100 bpm
        for peak in peaks:
            peak_time = start_time + peak / segment.info['sfreq']
            ecg_segments.append((peak_time, peak_time + 0.1))  # Mark a small window around each peak
    
    return ecg_segments
def detect_chewing_artifacts(raw_ica, start_time, duration=5):
    """
    Detect chewing artifacts in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of chewing artifacts.
    """
    # Generate the summary

    # Define the chewing artifact frequency range (e.g., 1-3 Hz as per research)
    freq_range = (1, 3)
    
    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)
    
    # Apply a band-pass filter within the chewing artifact frequency range
    segment_filtered = segment.filter(freq_range[0], freq_range[1], fir_design='firwin')
    
    # Identify temporal channels associated with chewing artifacts (T3, T4, F7, F8)
    chewing_channels = ['T3', 'T4', 'F7', 'F8']
    
    # Get data from those channels
    chewing_data = segment_filtered.copy().pick_channels(chewing_channels).get_data()
    
    # Set a threshold for detecting artifacts based on signal power or amplitude
    threshold = np.percentile(np.abs(chewing_data), 95)
    
    # Detect segments where the signal exceeds the threshold
    chewing_segments = []
    for i, channel_data in enumerate(chewing_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            chewing_segments.append((start_time_artifact, end_time_artifact))
    
    return chewing_segments
def detect_roving_eye_artifacts(raw_ica, start_time, duration=5):
    """
    Detect roving eye artifacts (related to slow lateral eye movements) in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of roving eye artifacts.
    """

    # Focus on frontal channels (Fp1, Fp2) for roving eye artifacts
    roving_channels = ['Fp1', 'Fp2']

    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)

    # Apply a low-pass filter (0.5-2 Hz) to capture slow eye movements
    segment_filtered = segment.filter(0.5, 2, fir_design='firwin')

    # Get data from the frontal channels
    roving_data = segment_filtered.copy().pick_channels(roving_channels).get_data()

    # Set a threshold for detecting artifacts based on signal amplitude
    threshold = np.percentile(np.abs(roving_data), 95)

    # Detect segments where the signal exceeds the threshold
    roving_segments = []
    for i, channel_data in enumerate(roving_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            roving_segments.append((start_time_artifact, end_time_artifact))

    return roving_segments
def detect_muscle_tension_artifacts(raw_ica, start_time, duration=5):
    """
    Detect muscle tension artifacts in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of muscle tension artifacts.
    """

    # Focus on temporal channels (T3, T4, F7, F8) for muscle tension artifacts
    muscle_channels = ['T3', 'T4', 'F7', 'F8']

    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)

    # Apply a high-pass filter (>20 Hz) to capture high-frequency muscle tension
    segment_filtered = segment.filter(20, 100, fir_design='firwin')

    # Get data from the muscle tension channels
    muscle_data = segment_filtered.copy().pick_channels(muscle_channels).get_data()

    # Set a threshold for detecting artifacts based on signal amplitude
    threshold = np.percentile(np.abs(muscle_data), 95)

    # Detect segments where the signal exceeds the threshold
    muscle_segments = []
    for i, channel_data in enumerate(muscle_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            muscle_segments.append((start_time_artifact, end_time_artifact))

    return muscle_segments
def detect_blink_artifacts(raw_ica, start_time, duration=5):
    """
    Detect blink artifacts in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of blink artifacts.
    """

    # Focus on frontal channels (Fp1, Fp2) for blink detection
    blink_channels = ['Fp1', 'Fp2']

    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)

    # Apply a low-pass filter (<2 Hz) to capture slow blink movements
    segment_filtered = segment.filter(None, 2, fir_design='firwin')

    # Get data from the frontal channels
    blink_data = segment_filtered.copy().pick_channels(blink_channels).get_data()

    # Set a threshold for detecting artifacts based on signal amplitude
    threshold = np.percentile(np.abs(blink_data), 95)

    # Detect segments where the signal exceeds the threshold
    blink_segments = []
    for i, channel_data in enumerate(blink_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            blink_segments.append((start_time_artifact, end_time_artifact))

    return blink_segments
def detect_rectus_spikes_artifacts(raw_ica, start_time, duration=5):
    """
    Detect rectus spike artifacts in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of rectus spike artifacts.
    """

    # Focus on frontal channels (Fp1, Fp2) for rectus spike detection
    rectus_channels = ['Fp1', 'Fp2']

    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)

    # Apply a high-pass filter (>20 Hz) to capture rapid deflections
    segment_filtered = segment.filter(20, None, fir_design='firwin')

    # Get data from the frontal channels
    rectus_data = segment_filtered.copy().pick_channels(rectus_channels).get_data()

    # Set a threshold for detecting artifacts based on signal amplitude
    threshold = np.percentile(np.abs(rectus_data), 99)

    # Detect segments where the signal exceeds the threshold
    rectus_spike_segments = []
    for i, channel_data in enumerate(rectus_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            rectus_spike_segments.append((start_time_artifact, end_time_artifact))

    return rectus_spike_segments
def detect_pdr_artifacts(raw_ica, start_time, duration=5):
    """
    Detect PDR (Posterior Dominant Rhythm) artifacts in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of PDR artifacts.
    """

    # Focus on occipital channels (O1, O2) for PDR artifacts
    pdr_channels = ['O1', 'O2']

    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)

    # Apply a band-pass filter in the alpha range (8-13 Hz)
    segment_filtered = segment.filter(8, 13, fir_design='firwin')

    # Get data from the occipital channels
    pdr_data = segment_filtered.copy().pick_channels(pdr_channels).get_data()

    # Set a threshold for detecting artifacts based on signal amplitude
    threshold = np.percentile(np.abs(pdr_data), 95)

    # Detect segments where the signal exceeds the threshold
    pdr_segments = []
    for i, channel_data in enumerate(pdr_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            pdr_segments.append((start_time_artifact, end_time_artifact))

    return pdr_segments
def detect_impedance_artifacts(raw_ica, start_time, duration=5):
    """
    Detect impedance artifacts in EEG data, typically caused by poor electrode contact.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of impedance artifacts.
    """

    # Impedance artifacts are typically in the very low frequency range (<1 Hz)
    freq_range = (0.1, 1)  # Adjusted frequency range to detect slow drifts
    
    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)
    
    # Apply a low-pass filter to capture impedance artifacts
    # Adjust the filter length based on the segment duration
    if segment.n_times > 1000:  # Ensure the segment is long enough for filtering
        segment_filtered = segment.filter(freq_range[0], freq_range[1], fir_design='firwin')
    else:
        return []  # Return an empty list if the segment is too short

    # Check if the filtered segment is empty
    if segment_filtered.get_data().size == 0:
        return []  # Return empty list if there's no valid data

    # Detect large amplitude drifts which are indicative of impedance artifacts
    impedance_data = segment_filtered.get_data()
    threshold = np.percentile(np.abs(impedance_data), 95)
    
    impedance_segments = []
    for i, channel_data in enumerate(impedance_data):
        above_threshold = np.where(np.abs(channel_data) > threshold)[0]
        if above_threshold.size > 0:
            # Convert sample indices to time
            start_time_artifact = start_time + above_threshold[0] / raw_ica.info['sfreq']
            end_time_artifact = start_time + above_threshold[-1] / raw_ica.info['sfreq']
            impedance_segments.append((start_time_artifact, end_time_artifact))
    
    return impedance_segments
def detect_epileptic_patterns(raw_ica, start_time, duration=5):
    """
    Detect epileptic patterns (spikes, sharp waves, etc.) in EEG data.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - start_time: The start time of the segment to analyze.
    - duration: Duration of the segment to analyze.

    Returns:
    - List of tuples indicating the start and end time of epileptic patterns.
    """

    # Define the epileptic pattern frequency range (e.g., spikes in 3-30 Hz range)
    freq_range = (3, 30)
    
    # Extract the segment of interest
    segment = raw_ica.copy().crop(tmin=start_time, tmax=start_time + duration)
    
    # Apply a band-pass filter to isolate potential epileptic patterns
    segment_filtered = segment.filter(freq_range[0], freq_range[1], fir_design='firwin')
    
    # Use amplitude thresholding and pattern recognition to detect epileptic features
    data = segment_filtered.get_data()
    threshold = np.percentile(np.abs(data), 98)  # Set a high percentile threshold for detecting spikes

    epileptic_segments = []
    for ch_data in data:
        spike_times = np.where(np.abs(ch_data) > threshold)[0]
        if spike_times.size > 0:
            start_time_pattern = start_time + spike_times[0] / raw_ica.info['sfreq']
            end_time_pattern = start_time + spike_times[-1] / raw_ica.info['sfreq']
            epileptic_segments.append((start_time_pattern, end_time_pattern))
    
    return epileptic_segments
def plot_frequency_bins(raw, frequency_bins):
    """
    Generate topomaps for custom frequency bins.

    Parameters:
    raw : mne.io.Raw
        The raw EEG data after ICA cleaning.
    frequency_bins : dict
        A dictionary with frequency bins as keys and frequency ranges as values.

    Returns:
    fig : matplotlib.figure.Figure
        The figure containing the topomaps for each frequency bin.
    """
    fig, axes = plt.subplots(1, len(frequency_bins), figsize=(20, 4))

    # Compute and plot topomaps for each custom frequency bin
    for ax, (bin_name, (low_freq, high_freq)) in zip(axes, frequency_bins.items()):
        # Filter the data for the given bin
        raw_filtered = raw.copy().filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin')

        # Compute the power spectral density (PSD)
        spectrum = raw_filtered.compute_psd(method='welch', fmin=0.5, fmax=40., n_fft=2048)
        psd, freqs = spectrum.get_data(return_freqs=True)
        
        # Select the relevant frequency band
        idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        psd_mean = np.mean(psd[:, idx], axis=1)  # Average across the selected frequency band
        
        # Plot the topomap with fixed color scaling
        im, _ = mne.viz.plot_topomap(psd_mean, raw.info, axes=ax, show=False, cmap=custom_cmap)
        ax.set_title(f"{bin_name}")
        
        # Add individual colorbar for each topomap
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig
def perform_dipole_analysis():
    subject = 'sample'
    subjects_dir = os.getenv('SUBJECTS_DIR', '/root/mne_data/MNE-sample-data/subjects')

    # Define synthetic events or real events if available
    events = mne.make_fixed_length_events(global_raw_ica, duration=1.0)  # Shorter duration per epoch
    event_id = {'stimulus': 1}

    # Create epochs (shorten the duration further)
    epochs = mne.Epochs(global_raw_ica, events, event_id, tmin=0, tmax=0.5, baseline=None, preload=True)  # 0.5s epochs

    # Aggressively downsample the data
    epochs.resample(64)  # Resample to 64 Hz

    # Set EEG reference to average
    epochs.set_eeg_reference('average', projection=True)

    # Compute the covariance matrix from a subset of epochs
    selected_epochs = epochs[:10]  # Randomly choose 10 epochs for faster computation
    cov = mne.compute_covariance(selected_epochs, tmin=0, tmax=None)

    # Use an even simpler BEM model (ico-2 instead of ico-3)
    model = mne.make_bem_model(subject=subject, ico=2, subjects_dir=str(subjects_dir))  # Lower resolution BEM model
    bem = mne.make_bem_solution(model)

    # Set up the forward model using an identity transform
    trans = mne.transforms.Transform('head', 'mri')  # Identity transform for EEG-only analysis
    src = mne.setup_source_space(subject, spacing='oct4', subjects_dir=str(subjects_dir), add_dist=False)  # Lower resolution source space
    fwd = mne.make_forward_solution(epochs.info, trans=trans, src=src, bem=bem, meg=False, eeg=True)

    # Perform dipole fitting
    dip, residual = mne.fit_dipole(selected_epochs.average(), cov, bem, trans, n_jobs=1, verbose=False)

    # Plot dipole locations
    fig = dip.plot_locations(trans, subject, subjects_dir, mode="outlines", show=False)
    return fig
def plot_frequency_bins(raw, frequency_bins):
    """
    Generate topomaps for custom frequency bins.

    Parameters:
    raw : mne.io.Raw
        The raw EEG data after ICA cleaning.
    frequency_bins : dict
        A dictionary with frequency bins as keys and frequency ranges as values.

    Returns:
    fig : matplotlib.figure.Figure
        The figure containing the topomaps for each frequency bin.
    """
    fig, axes = plt.subplots(1, len(frequency_bins), figsize=(20, 4))

    # Compute and plot topomaps for each custom frequency bin
    for ax, (bin_name, (low_freq, high_freq)) in zip(axes, frequency_bins.items()):
        # Filter the data for the given bin
        raw_filtered = raw.copy().filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin')

        # Compute the power spectral density (PSD)
        spectrum = raw_filtered.compute_psd(method='welch', fmin=0.5, fmax=40., n_fft=2048)
        psd, freqs = spectrum.get_data(return_freqs=True)
        
        # Select the relevant frequency band
        idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        psd_mean = np.mean(psd[:, idx], axis=1)  # Average across the selected frequency band
        
        # Plot the topomap with fixed color scaling
        im, _ = mne.viz.plot_topomap(psd_mean, raw.info, axes=ax, show=False, cmap=custom_cmap)
        ax.set_title(f"{bin_name}")
        
        # Add individual colorbar for each topomap
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    return fig


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'C:/temp'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Set max upload size to 50 MB

# Initialize Flask-SocketIO
socketio = SocketIO(app)
# Define preprocessing variables at the global scope
channels_to_drop = ['Bio1-2', 'Bio3-4', 'ECG', 'Bio4', 'VSyn', 'ASyn', 'LABEL', 'EEG Fpz','A1','A2','EEG Oz']
mapping = {
    'EEG Fp1': 'Fp1', 'EEG Fp2': 'Fp2',
    'EEG F7': 'F7', 'EEG F3': 'F3', 'EEG Fz': 'Fz',
    'EEG F4': 'F4', 'EEG F8': 'F8', 'EEG T3': 'T3',
    'EEG C3': 'C3', 'EEG Cz': 'Cz', 'EEG C4': 'C4',
    'EEG T4': 'T4', 'EEG T5': 'T5', 'EEG P3': 'P3',
    'EEG Pz': 'Pz', 'EEG P4': 'P4', 'EEG T6': 'T6',
    'EEG O1': 'O1', 'EEG O2': 'O2'
}
montage = mne.channels.make_standard_montage('standard_1020')

# Define frequency bands

bands = {
    "delta": (1.5, 4),
    "theta": (4, 7.5),
    "alpha": (7.5, 14),
    "beta-1": (14, 20),
    "beta-2": (20, 30),
    "gamma": (30, 40)
}



# Define colors for the bands
band_colors = {
    "delta": 'maroon',
    "theta": 'red',
    "alpha": 'yellow',
    "beta-1": 'green',
    "beta-2": 'cyan',
    "gamma": 'blue'
}

# Define the custom colormap to match the colorbar in your image
colors = [
    (0.0, "#000000"),  # Black
    (0.1, "#0000FF"),  # Blue
    (0.2, "#00FFFF"),  # Cyan
    (0.3, "#00FF00"),  # Green
    (0.5, "#FFFF00"),  # Yellow
    (0.7, "#FFA500"),  # Orange
    (0.9, "#FF0000"),  # Red
    (1.0, "#FF00FF")   # Magenta
]
custom_cmap = LinearSegmentedColormap.from_list("custom_jet", colors)

# Define channel groups for regional analysis (frontal, parietal, occipital, etc.)
channel_groups = {
    "frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
    "parietal": ["P3", "P4", "Pz"],
    "occipital": ["O1", "O2"],
    "temporal": ["T3", "T4", "T5", "T6"],
    "central": ["C3", "C4", "Cz"]
}

frequency_bins = {
                'Bin 1 (~0.98 Hz)': (0.97, 0.99),
                'Bin 2 (~1.95 Hz)': (1.94, 1.96),
                'Bin 3 (~8.30 Hz)': (8.29, 8.31),
                'Bin 4 (~26.51 Hz)': (26.50, 26.62),
                'Bin 5 (~30.76 Hz)': (30.75, 30.77)
            }

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    
def main_gpt_call(analysis, summary, participant_name,age,gender,known_issues,medications):
    response = client.chat.completions.create(
        model="chatgpt-4o-latest",
        messages=[
            {"role": "system", "content": """You are an expert in neuroscience, specializing in
             EEG analysis and frequency band interpretation."""},
            {"role": "user", "content": f"""Given the following summary of EEG data focused on {analysis} 
                            : {summary}, 
                            please analyze the data and provide a short report with conclusions, 
                            specifically focusing on the {analysis}. The participant is a {age}-year-old 
                            {gender}, 
                            named {name} having following known issues {known_issues}. the participant is 
                            taking medications: {medications}. Write the report in a 
                            way that addresses the participant directly, using their name when appropriate. 
                            The report should be structured into three sections (don't add any other heading/title): 
                            Introduction, Findings, and Conclusion. Do not sugar coat it, make sure to bring up 
                            anything alarming in the data in the conclusion, or any possible dignosis. Don't add 
                            signing off remarks like, yours sincerely etc. The language should be formal, clear, 
                            concise, and suitable for a primary school-going child (aged {age} years), while 
                            maintaining proper report format. Make sure to explain what the findings suggest 
                            about brain activity.Please write the report in British English"""}
        ]
    )
    return response.choices[0].message.content

def extract_detailed_eeg_features(raw):
    """
    Extracts detailed features from EEG data for multiple frequency bands.

    Parameters:
    - raw: mne.io.Raw
        The ICA-cleaned raw EEG data object.

    Returns:
    - features_json: str
        A JSON-formatted string containing the extracted features.
    """
    features = {}

    # Iterate over each frequency band to compute features
    for band, (low_freq, high_freq) in bands.items():
        print(f"Processing {band.capitalize()} Band ({low_freq}-{high_freq} Hz)...")

        # Filter the data for the specific frequency band
        band_data = raw.copy().filter(low_freq, high_freq, fir_design='firwin')

        # Calculate the mean and standard deviation for each channel in the band
        band_mean = np.mean(band_data._data, axis=1)
        band_std = np.std(band_data._data, axis=1)

        # Calculate the Power Spectral Density (PSD) using the `compute_psd` method
        psd = band_data.compute_psd(fmin=low_freq, fmax=high_freq)
        psd_data = psd.get_data().mean(axis=1)  # Average PSD across all epochs

        # Compute Band Power using the Simpson's rule
        freqs, psd_all = psd.freqs, psd.get_data()
        band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        band_power = simpson(psd_all[:, band_idx], dx=np.diff(freqs)[0], axis=1)

        # Compute Hjorth Parameters: Activity, Mobility, and Complexity
        activity = np.var(band_data._data, axis=1)
        mobility = np.sqrt(np.var(np.diff(band_data._data, axis=1), axis=1) / activity)
        complexity = np.sqrt(np.var(np.diff(np.diff(band_data._data, axis=1), axis=1), axis=1) / np.var(np.diff(band_data._data, axis=1), axis=1)) / mobility

        # Store the features in the dictionary
        features[f'{band}_mean'] = {ch_name: band_mean[idx] for idx, ch_name in enumerate(raw.info['ch_names'])}
        features[f'{band}_std'] = {ch_name: band_std[idx] for idx, ch_name in enumerate(raw.info['ch_names'])}
        features[f'{band}_psd'] = {ch_name: psd_data[idx] for idx, ch_name in enumerate(raw.info['ch_names'])}
        features[f'{band}_power'] = {ch_name: band_power[idx] for idx, ch_name in enumerate(raw.info['ch_names'])}
        features[f'{band}_hjorth_activity'] = {ch_name: activity[idx] for idx, ch_name in enumerate(raw.info['ch_names'])}
        features[f'{band}_hjorth_mobility'] = {ch_name: mobility[idx] for idx, ch_name in enumerate(raw.info['ch_names'])}
        features[f'{band}_hjorth_complexity'] = {ch_name: complexity[idx] for idx, ch_name in enumerate(raw.info['ch_names'])}

    # Convert the features dictionary to a JSON-formatted string
    features_json = json.dumps(features, indent=4)

    # Print or return the JSON string
    return features_json
def generate_raw_summary(raw, ica, eog_channels):
    # Extract basic information about the raw data
    num_channels = len(raw.ch_names)
    sampling_freq = raw.info['sfreq']
    duration = raw.times[-1] / 60  # Duration in minutes
    excluded_components = ica.exclude
    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)

    # Create a summary string
    summary = f"""
    # EEG Data Summary:
    - Number of channels: {num_channels}
    - Sampling frequency: {sampling_freq} Hz
    - Recording duration: {duration:.2f} minutes
    - EOG channels used for artifact detection: {eog_channels}
    - Number of ICA components: {ica.n_components_}
    - Identified EOG artifact components: {eog_indices}
    - Components excluded after ICA: {excluded_components}
    - ICA performed with {ica.n_components_} components, random state = {ica.random_state}, and max iterations = {ica.max_iter}.
    """

    return summary.strip()
def generate_delta_band_summary_per_channel_full_duration(raw_ica,fmin,fmax):
    # Delta band filter range
    low, high = fmin,fmax

    # Filter the data for the delta band across the entire duration
    delta_filtered = raw_ica.copy().filter(low, high, fir_design='firwin')

    # Get the entire data and time array
    data, times = delta_filtered[:, :]

    # Generate summaries for each channel
    channel_summaries = []
    for i, channel_name in enumerate(raw_ica.info['ch_names']):
        channel_data = data[i, :]
        mean_power = np.mean(np.abs(channel_data)**2)
        variance = np.var(channel_data)
        max_amplitude = np.max(np.abs(channel_data))
        min_amplitude = np.min(np.abs(channel_data))

        # Create a summary string for each channel
        summary = f"""
        Channel: {channel_name}
        - Frequency range: {low} Hz to {high} Hz
        - Duration analyzed: Entire recording
        - Mean power in the delta band: {mean_power:.2f} µV²
        - Variance: {variance:.2f} µV²
        - Maximum amplitude: {max_amplitude:.2f} µV
        - Minimum amplitude: {min_amplitude:.2f} µV
        """
        channel_summaries.append(summary.strip())

    # Combine all channel summaries
    complete_summary = "\n\n".join(channel_summaries)
    return complete_summary
def generate_detailed_relative_power_summary(raw_ica, bands, channel_groups):
    # Compute PSD and relative power for topomaps
    spectrum = raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)

    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psds[:, freq_mask], axis=1)

    total_power = np.sum(list(band_powers.values()), axis=0)
    relative_powers = {band: (power / total_power) * 100 for band, power in band_powers.items()}

    # Regional Analysis: Break down power into specific regions (e.g., frontal, parietal)
    regional_summaries = []
    for region, channels in channel_groups.items():
        region_powers = {}
        for band in bands:
            region_power = np.mean([relative_powers[band][raw_ica.ch_names.index(ch)] for ch in channels])
            region_powers[band] = region_power

        regional_summary = f"{region.capitalize()} Region:"
        for band, power in region_powers.items():
            regional_summary += f"\n  - {band.capitalize()} Band: Average Power: {power:.2f}%"
        regional_summaries.append(regional_summary)

    # Generate summary based on detailed analysis
    summary_lines = [
        "Detailed Relative Power Analysis Summary:",
        "The analysis computed the relative power distribution across six EEG frequency bands:"
    ]

    for band, power in relative_powers.items():
        avg_power = np.mean(power)
        std_power = np.std(power)
        max_power = np.max(power)
        min_power = np.min(power)
        summary_lines.append(
            f"- {band.capitalize()} Band ({bands[band][0]} - {bands[band][1]} Hz): "
            f"Average Power: {avg_power:.2f}%, Std Dev: {std_power:.2f}%, Max Power: {max_power:.2f}%, Min Power: {min_power:.2f}%."
        )

    summary_lines.append("\nRegional Analysis:")
    summary_lines.extend(regional_summaries)

    summary_lines.append(
        "The detailed analysis provides insights into how different brain regions contribute to various frequency bands, "
        "reflecting cognitive functions such as relaxation, attention, and higher-order processing."
    )

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_absolute_power_summary(raw_ica, bands, channel_groups):
    # Compute PSD and absolute power for topomaps
    spectrum = raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Convert PSD to absolute power in µV²
    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        # Sum across the frequency bins within the band and scale to µV²
        band_powers[band] = np.sum(psds[:, freq_mask], axis=1)

    # Include frequencies and PSD values in the summary
    summary_lines = [
        "Detailed Absolute Power Analysis Summary:",
        "The analysis computed the absolute power distribution across six EEG frequency bands:",
        f"Frequencies: {freqs}",
        f"PSD Values (first channel): {psds[0]}"
    ]

    for band, power in band_powers.items():
        avg_power = np.mean(power)
        std_power = np.std(power)
        max_power = np.max(power)
        min_power = np.min(power)
        summary_lines.append(
            f"- {band.capitalize()} Band ({bands[band][0]} - {bands[band][1]} Hz): "
            f"Average Power: {avg_power:.10f} µV², Std Dev: {std_power:.10f} µV², Max Power: {max_power:.10f} µV², Min Power: {min_power:.10f} µV²."
        )

    summary_lines.append("\nRegional and Channel-Specific Analysis:")
    for region, channels in channel_groups.items():
        summary_lines.append(f"\n{region.capitalize()} Region:")
        for band in bands:
            region_power = np.mean([band_powers[band][raw_ica.ch_names.index(ch)] for ch in channels])
            summary_lines.append(f"  - {band.capitalize()} Band: {region_power:.10f} µV² (Average across {', '.join(channels)})")

        for ch in channels:
            summary_lines.append(f"\nChannel {ch} Analysis:")
            for band in bands:
                ch_index = raw_ica.ch_names.index(ch)
                ch_power = band_powers[band][ch_index]
                summary_lines.append(f"  - {band.capitalize()} Band: {ch_power:.10f} µV²")

    summary_lines.append(
        "\nThe detailed analysis provides insights into the distribution of absolute power across different regions and channels, "
        "reflecting specific brain activities related to the analyzed frequency bands."
    )

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_relative_spectra_summary(raw_ica, bands):
    # Compute PSD and relative power for spectra
    spectrum = raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)

    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psds[:, freq_mask], axis=1)

    total_power = np.sum(list(band_powers.values()), axis=0)
    relative_powers = {band: (power / total_power) * 100 for band, power in band_powers.items()}

    # Generate the summary based on the relative spectra
    summary_lines = [
        "Detailed Relative Power Spectra Analysis Summary:",
        "This analysis computed the relative power distribution across six EEG frequency bands at the channel level.",
        f"Frequencies: {freqs}",
        f"Total Power (first channel): {total_power[0]}"
    ]

    # Channel-specific analysis
    for idx, ch_name in enumerate(raw_ica.ch_names):
        summary_lines.append(f"\nChannel {ch_name} Analysis:")
        for band, power in relative_powers.items():
            summary_lines.append(
                f"  - {band.capitalize()} Band ({bands[band][0]} - {bands[band][1]} Hz): "
                f"Relative Power: {power[idx]:.2f}%"
            )
        summary_lines.append(
            f"  - Total Power: {total_power[idx]:.2f} (µV²)"
        )

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_absolute_spectra_summary(raw_ica, bands):
    # Compute PSD for absolute power spectra
    spectrum = raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)

    band_powers = {}
    for band, (fmin, fmax) in bands.items():
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        band_powers[band] = np.mean(psds[:, freq_mask], axis=1)

    # Generate the summary based on the absolute spectra
    summary_lines = [
        "Detailed Absolute Power Spectra Analysis Summary:",
        "This analysis computed the absolute power distribution across six EEG frequency bands at the channel level.",
        f"Frequencies: {freqs}",
        f"Total Power (first channel): {np.sum(psds[0]):.10f} (µV²)"
    ]

    # Channel-specific analysis
    for idx, ch_name in enumerate(raw_ica.ch_names):
        summary_lines.append(f"\nChannel {ch_name} Analysis:")
        for band, power in band_powers.items():
            summary_lines.append(
                f"  - {band.capitalize()} Band ({bands[band][0]} - {bands[band][1]} Hz): "
                f"Absolute Power: {power[idx]:.10f} (µV²)"
            )
        summary_lines.append(
            f"  - Total Power: {np.sum(psds[idx]):.10f} (µV²)"
        )

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_theta_beta_ratio_summary(raw_ica, bands):
    # Compute PSD for the Theta/Beta ratio
    spectrum = raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Calculate Theta and Beta power
    theta_power = np.mean(psds[:, (freqs >= bands['theta'][0]) & (freqs <= bands['theta'][1])], axis=1)
    beta_power = np.mean(psds[:, (freqs >= bands['beta-1'][0]) & (freqs <= bands['beta-1'][1])], axis=1)

    # Compute Theta/Beta ratio
    theta_beta_ratio = theta_power / beta_power

    # Generate the summary
    summary_lines = [
        "Detailed Theta/Beta Ratio Analysis Summary:",
        "This analysis computed the Theta/Beta ratio across all EEG channels to provide insights into cognitive states.",
        f"Frequencies: {freqs}",
        f"PSD Values (first channel): {psds[0]}"
    ]

    # Channel-specific analysis
    for idx, ch_name in enumerate(raw_ica.ch_names):
        summary_lines.append(f"\nChannel {ch_name} Analysis:")
        summary_lines.append(
            f"  - Theta Power: {theta_power[idx]:.10f} (µV²)"
        )
        summary_lines.append(
            f"  - Beta Power: {beta_power[idx]:.10f} (µV²)"
        )
        summary_lines.append(
            f"  - Theta/Beta Ratio: {theta_beta_ratio[idx]:.10f}"
        )

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_brain_mapping_summary(raw_ica, bands):
    # Compute PSD for the Theta/Beta ratio
    spectrum = raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Calculate Theta and Beta power
    theta_power = np.mean(psds[:, (freqs >= bands['theta'][0]) & (freqs <= bands['theta'][1])], axis=1)
    beta_power = np.mean(psds[:, (freqs >= bands['beta-1'][0]) & (freqs <= bands['beta-1'][1])], axis=1)

    # Compute Theta/Beta ratio
    theta_beta_ratio = theta_power / beta_power

    # Determine channels with increased and decreased activity
    increased_activity_channels = np.where(theta_beta_ratio > np.mean(theta_beta_ratio) + np.std(theta_beta_ratio))[0]
    decreased_activity_channels = np.where(theta_beta_ratio < np.mean(theta_beta_ratio) - np.std(theta_beta_ratio))[0]

    # Generate summary
    summary_lines = [
        "Detailed Brain Mapping Analysis Summary:",
        "This analysis visualizes the Theta/Beta ratio across EEG channels to identify regions with increased or decreased activity.",
        "Channels showing increased activity (marked in green) have higher-than-average Theta/Beta ratios, while channels showing decreased activity (marked in red) have lower-than-average ratios.",
    ]

    if len(increased_activity_channels) > 0:
        summary_lines.append("\nChannels with Increased Activity (Theta/Beta Ratio > Mean + 1 Std Dev):")
        for ch_idx in increased_activity_channels:
            ch_name = raw_ica.ch_names[ch_idx]
            summary_lines.append(f"  - {ch_name}: Theta/Beta Ratio = {theta_beta_ratio[ch_idx]:.2f}")
    else:
        summary_lines.append("\nNo channels were identified with significantly increased activity.")

    if len(decreased_activity_channels) > 0:
        summary_lines.append("\nChannels with Decreased Activity (Theta/Beta Ratio < Mean - 1 Std Dev):")
        for ch_idx in decreased_activity_channels:
            ch_name = raw_ica.ch_names[ch_idx]
            summary_lines.append(f"  - {ch_name}: Theta/Beta Ratio = {theta_beta_ratio[ch_idx]:.2f}")
    else:
        summary_lines.append("\nNo channels were identified with significantly decreased activity.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_occipital_alpha_peak_summary(raw_ica, alpha_band):
    # Compute PSD for Occipital Alpha Peak analysis
    spectrum = raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
    psds, freqs = spectrum.get_data(return_freqs=True)

    # Define the occipital channels
    occipital_channels = ['O1', 'O2']
    occipital_psds = psds[[raw_ica.ch_names.index(ch) for ch in occipital_channels], :]

    # Extract the alpha band frequency range
    alpha_mask = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
    alpha_freqs = freqs[alpha_mask]
    occipital_psds_alpha = occipital_psds[:, alpha_mask]

    # Find the peak alpha frequency for each occipital channel
    alpha_peaks = {}
    for idx, ch_name in enumerate(occipital_channels):
        alpha_psd = occipital_psds_alpha[idx]
        peak_idx = np.argmax(alpha_psd)
        alpha_peaks[ch_name] = alpha_freqs[peak_idx]

    # Generate the summary
    summary_lines = [
        "Detailed Occipital Alpha Peak Analysis Summary:",
        "This analysis focuses on detecting the alpha peak frequency in the occipital channels (O1, O2), which is important for assessing resting-state brain activity.",
        f"Alpha Frequency Range Analyzed: {alpha_band[0]} - {alpha_band[1]} Hz",
        "The following are the detected alpha peaks for each occipital channel:"
    ]

    for ch_name, peak_freq in alpha_peaks.items():
        summary_lines.append(f"  - {ch_name}: Peak Alpha Frequency = {peak_freq:.2f} Hz")

    summary_lines.append("\nThis analysis helps determine the dominant alpha frequency in the occipital region, which is often linked to relaxation and visual processing.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_chewing_artifact_summary_full_duration(raw_ica, chewing_channels, detect_chewing_artifacts, window_duration=5):
    # Calculate the number of windows needed to cover the entire EEG recording
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Generate the summary
    summary_lines = [
        "Comprehensive Chewing Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting chewing-related artifacts in the temporal channels (T3, T4, F7, F8), which are most sensitive to muscle activity during chewing."
    ]

    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect chewing artifacts in the current window
        chewing_segments = detect_chewing_artifacts(raw_ica, start_time, end_time - start_time)

        if chewing_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in chewing_segments:
                affected_channels = []
                for ch in chewing_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant chewing artifacts detected within this time segment.")

    summary_lines.append("\nThis detailed and comprehensive analysis spans the entire EEG recording, identifying all time periods heavily influenced by chewing activity across the temporal channels.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_ecg_artifact_summary_full_duration(raw_ica, ecg_channels, detect_ecg_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of ECG artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - ecg_channels: List of channels to focus on for ECG artifact detection.
    - detect_ecg_artifacts: Function to detect ECG artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive ECG Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting cardiac/ECG-related artifacts in selected channels (T3, T4, Cz) that are most likely to pick up ECG activity.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect ECG artifacts in the current window
        ecg_segments = detect_ecg_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if ecg_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in ecg_segments:
                affected_channels = []
                for ch in ecg_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    # Check if the segment is relevant for this channel
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant ECG artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies ECG artifacts across the entire EEG recording, highlighting time periods that are affected by cardiac activity.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_rectus_artifact_summary_full_duration(raw_ica, rectus_channels, detect_rectus_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of rectus artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - rectus_channels: List of channels to focus on for rectus artifact detection.
    - detect_rectus_artifacts: Function to detect rectus artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive Rectus Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting rectus-related (eye movement) artifacts in selected frontal channels (Fp1, Fp2) that are most likely to pick up slow eye movements.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect rectus artifacts in the current window
        rectus_segments = detect_rectus_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if rectus_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in rectus_segments:
                affected_channels = []
                for ch in rectus_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    # Check if the segment is relevant for this channel
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant rectus artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies rectus artifacts across the entire EEG recording, highlighting time periods that are affected by slow eye movements.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_roving_eye_artifact_summary_full_duration(raw_ica, roving_channels, detect_roving_eye_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of roving eye artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - roving_channels: List of channels to focus on for roving eye artifact detection.
    - detect_roving_eye_artifacts: Function to detect roving eye artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive Roving Eye Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting roving eye (slow lateral movement) artifacts in selected frontal channels (Fp1, Fp2) that are most likely to pick up such artifacts.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect roving eye artifacts in the current window
        roving_segments = detect_roving_eye_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if roving_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in roving_segments:
                affected_channels = []
                for ch in roving_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    # Check if the segment is relevant for this channel
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant roving eye artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies roving eye artifacts across the entire EEG recording, highlighting time periods that are affected by slow lateral eye movements.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_muscle_tension_artifact_summary_full_duration(raw_ica, muscle_channels, detect_muscle_tension_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of muscle tension artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - muscle_channels: List of channels to focus on for muscle tension artifact detection.
    - detect_muscle_tension_artifacts: Function to detect muscle tension artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive Muscle Tension Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting muscle tension artifacts in selected temporal channels (T3, T4, F7, F8) that are most likely to pick up high-frequency muscle activity.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect muscle tension artifacts in the current window
        muscle_segments = detect_muscle_tension_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if muscle_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in muscle_segments:
                affected_channels = []
                for ch in muscle_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    # Check if the segment is relevant for this channel
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant muscle tension artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies muscle tension artifacts across the entire EEG recording, highlighting time periods that are affected by high-frequency muscle activity.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_blink_artifact_summary_full_duration(raw_ica, blink_channels, detect_blink_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of blink artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - blink_channels: List of channels to focus on for blink artifact detection.
    - detect_blink_artifacts: Function to detect blink artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive Blink Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting blink artifacts in selected frontal channels (Fp1, Fp2) that are most likely to pick up slow blink movements.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect blink artifacts in the current window
        blink_segments = detect_blink_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if blink_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in blink_segments:
                affected_channels = []
                for ch in blink_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    # Check if the segment is relevant for this channel
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant blink artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies blink artifacts across the entire EEG recording, highlighting time periods that are affected by slow blink movements.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_rectus_spike_artifact_summary_full_duration(raw_ica, rectus_channels, detect_rectus_spikes_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of rectus spike artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - rectus_channels: List of channels to focus on for rectus spike artifact detection.
    - detect_rectus_spikes_artifacts: Function to detect rectus spike artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive Rectus Spike Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting rectus spike artifacts in selected frontal channels (Fp1, Fp2) that are most likely to pick up rapid deflections related to sudden eye movements or muscle contractions.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect rectus spike artifacts in the current window
        rectus_spike_segments = detect_rectus_spikes_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if rectus_spike_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in rectus_spike_segments:
                affected_channels = []
                for ch in rectus_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    # Check if the segment is relevant for this channel
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant rectus spike artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies rectus spike artifacts across the entire EEG recording, highlighting time periods that are affected by rapid deflections in frontal channels.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_pdr_artifact_summary_full_duration(raw_ica, pdr_channels, detect_pdr_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of PDR artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - pdr_channels: List of channels to focus on for PDR artifact detection.
    - detect_pdr_artifacts: Function to detect PDR artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive PDR Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting PDR artifacts in selected occipital channels (O1, O2) that are most likely to pick up alpha rhythms associated with relaxed states.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect PDR artifacts in the current window
        pdr_segments = detect_pdr_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if pdr_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in pdr_segments:
                affected_channels = []
                for ch in pdr_channels:
                    ch_index = raw_ica.ch_names.index(ch)
                    # Check if the segment is relevant for this channel
                    if any(segment[0] <= t <= segment[1] for t in np.arange(start_time, end_time, 1 / raw_ica.info['sfreq'])):
                        affected_channels.append(ch)
                if affected_channels:
                    summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s, Affected Channels: {', '.join(affected_channels)}")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant PDR artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies PDR artifacts across the entire EEG recording, highlighting time periods that are affected by alpha rhythms in the occipital channels.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_impedance_artifact_summary_full_duration(raw_ica, detect_impedance_artifacts, window_duration=5):
    """
    Generate a detailed and granular summary of impedance artifact detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - detect_impedance_artifacts: Function to detect impedance artifacts in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive Impedance Artifact Detection Summary:",
        "This analysis covers the entire EEG recording, detecting impedance artifacts caused by poor electrode contact. These artifacts typically manifest as slow drifts in the signal.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect impedance artifacts in the current window
        impedance_segments = detect_impedance_artifacts(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if impedance_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in impedance_segments:
                summary_lines.append(f"  - Segment: {segment[0]:.2f}s to {segment[1]:.2f}s")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant impedance artifacts detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies impedance artifacts across the entire EEG recording, highlighting time periods that are affected by slow drifts in the signal.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_detailed_epileptic_pattern_summary_full_duration(raw_ica, detect_epileptic_patterns, window_duration=5):
    """
    Generate a detailed and granular summary of epileptic pattern detection across the entire EEG recording.

    Parameters:
    - raw_ica: The ICA-cleaned raw data object.
    - detect_epileptic_patterns: Function to detect epileptic patterns in a given segment.
    - window_duration: Duration of each analysis window in seconds.

    Returns:
    - A detailed summary string ready for analysis and reporting.
    """
    # Calculate the total recording duration
    total_duration = raw_ica.times[-1]
    num_windows = int(np.ceil(total_duration / window_duration))

    # Initialize summary
    summary_lines = [
        "Comprehensive Epileptic Pattern Detection Summary:",
        "This analysis covers the entire EEG recording, detecting epileptic patterns such as spikes and sharp waves that may be indicative of seizure activity.",
        f"Total recording duration: {total_duration:.2f} seconds.",
        f"Analysis conducted in {num_windows} time windows, each {window_duration} seconds long."
    ]

    # Analyze the EEG recording in windows
    for window_idx in range(num_windows):
        start_time = window_idx * window_duration
        end_time = min((window_idx + 1) * window_duration, total_duration)

        # Detect epileptic patterns in the current window
        epileptic_segments = detect_epileptic_patterns(raw_ica, start_time, end_time - start_time)

        # Add details for the current window
        if epileptic_segments:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            for segment in epileptic_segments:
                summary_lines.append(f"  - Epileptic Pattern Detected: {segment[0]:.2f}s to {segment[1]:.2f}s")
        else:
            summary_lines.append(f"\nAnalyzed time segment: {start_time:.2f}s to {end_time:.2f}s")
            summary_lines.append("  - No significant epileptic patterns detected within this time segment.")

    summary_lines.append("\nThis comprehensive analysis identifies epileptic patterns across the entire EEG recording, highlighting time periods that may indicate potential seizure activity.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary
def generate_frequency_bin_summary(raw, frequency_bins):
    """
    Generate a detailed summary for custom frequency bins analysis.

    Parameters:
    - raw: mne.io.Raw
        The raw EEG data after ICA cleaning.
    - frequency_bins: dict
        A dictionary with frequency bins as keys and frequency ranges as values.

    Returns:
    - A detailed summary string ready for further analysis or report generation.
    """
    summary_lines = []
    summary_lines.append("Detailed Frequency Bin Analysis Summary:")

    # Loop through each frequency bin to analyze
    for bin_name, (low_freq, high_freq) in frequency_bins.items():
        summary_lines.append(f"\nAnalyzing {bin_name} ({low_freq:.2f} - {high_freq:.2f} Hz):")

        # Filter the data for the given frequency bin
        raw_filtered = raw.copy().filter(l_freq=low_freq, h_freq=high_freq, fir_design='firwin')

        # Compute the power spectral density (PSD)
        spectrum = raw_filtered.compute_psd(method='welch', fmin=low_freq, fmax=high_freq, n_fft=2048)
        psd, freqs = spectrum.get_data(return_freqs=True)

        # Select the relevant frequency range and compute average power
        idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
        psd_mean = np.mean(psd[:, idx], axis=1)

        # Add detailed channel-level summary for the current frequency bin
        for ch_idx, ch_name in enumerate(raw.info['ch_names']):
            summary_lines.append(f"  - {ch_name}: Average Power: {psd_mean[ch_idx]:.2f} µV²")

    summary_lines.append("\nThis analysis highlights the power distribution across custom frequency bins and provides insights into specific brain activities captured within these ranges.")

    # Combine all lines into a single summary
    summary = "\n".join(summary_lines)
    return summary

# Use global variables to store raw and ICA-cleaned EEG data
global_raw = None
global_raw_ica = None
global_ica = None
global_raw_openai = None
global_raw_ica_openai = None
global_ica_components_openai = None
global_bands_openai = {}
global_relative_topo_openai = None
global_abs_topo_openai = None
global_rel_spectra_openai = None
global_abs_spectra_openai = None
global_theta_beta_ratio_openai = None
global_brain_mapping_openai = None
global_occipital_alpha_peak_openai = None
global_chewing_artifect_openai = None
global_ecg_artifect_openai  = None
global_rectus_artifect_openai = None 
global_roving_eye_artifect_openai = None 
global_muscle_tension_artifect_openai = None 
global_blink_artifect_openai = None
global_blink_artifect_openai = None 
global_rectus_spike_artifect_openai = None 
global_pdr_openai = None 
global_impedance_openai = None 
global_epileptic_openai = None 
global_frq_bins_openai = None


# Read the API key from the text file
with open('/root/apikey.txt', 'r') as file:
    openai_api_key = file.read().strip()

# OpenAI API Key setup
client = OpenAI(api_key=openai_api_key)# Route for file upload and main dashboard
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global global_raw, global_raw_ica, global_ica, global_raw_openai, \
    global_raw_ica_openai, global_ica_components_openai, name, dob, age, \
    gender, known_issues, global_bands_openai, global_relative_topo_openai, \
    global_abs_topo_openai, global_rel_spectra_openai, global_abs_spectra_openai, \
    global_theta_beta_ratio_openai, global_brain_mapping_openai,global_occipital_alpha_peak_openai, \
    global_chewing_artifect_openai, global_ecg_artifect_openai,global_rectus_artifect_openai, \
    global_roving_eye_artifect_openai, global_muscle_tension_artifect_openai, global_blink_artifect_openai, \
    global_blink_artifect_openai,global_rectus_spike_artifect_openai, global_pdr_openai, \
    global_impedance_openai, global_epileptic_openai, global_frq_bins_openai

    if request.method == 'POST':
        name = request.form.get('name')
        dob = request.form.get('dob')
        age = request.form.get('age')
        gender = request.form.get('gender')
        known_issues = request.form.get('known_issues')
        medications = request.form.get('medications')
        # Handle file upload
        if 'file' in request.files:
            try:
                f = request.files['file']
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                f.save(filepath)
                
                # Load the EEG data using MNE
                raw = mne.io.read_raw_edf(filepath, preload=True)
                

                # Apply preprocessing
                
                # Apply preprocessing
                raw.drop_channels(channels_to_drop)
                raw.rename_channels(mapping)
                raw.set_montage(montage)
                raw.filter(0.3, 70., fir_design='firwin')
                raw.set_eeg_reference(ref_channels='average')
                
                # Set the EOG channels (Fp1 and Fp2) for detecting eye movement artifacts
                eog_channels = ['Fp1', 'Fp2']
                

                # Perform ICA for artifact correction
                ica = mne.preprocessing.ICA(n_components=19, random_state=97, max_iter=800)
                ica.fit(raw)
                eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)
                ica.exclude = eog_indices
                
                raw_ica = ica.apply(raw.copy())
                #creating channel dictionary
                raw_ica_channel_names = raw_ica.info['ch_names']
                # Store channel names and indexes in a dictionary
                channel_index_dict = {name: index for index, name in enumerate(raw_ica_channel_names)}

                # Store the processed data globally
                global_raw = raw
                global_raw_ica = raw_ica
                global_ica = ica

                #raw openai
                raw_eeg_features_json = extract_detailed_eeg_features(global_raw)
                raw_response = main_gpt_call("Raw EEG feature data", raw_eeg_features_json, name, 
                                             age,gender,known_issues,medications)
                global_raw_openai = raw_response
                #raw ica openai
                raw_ica_eeg_features_json = extract_detailed_eeg_features(global_raw_ica)
                raw_ica_response = main_gpt_call("ICA-cleaned EEG feature data", raw_ica_eeg_features_json,
                                                 name, age, gender, known_issues,medications)
                global_raw_ica_openai = raw_ica_response
                
                #ica component openai
                summary_ica_components = generate_raw_summary(global_raw,global_ica,eog_channels)
                response_ica_components = main_gpt_call("ICA component and property analysis", summary_ica_components,
                                                 name, age, gender, known_issues,medications)
                global_ica_components_openai = response_ica_components
                #band wise openai
                for band in bands.keys():
                    print(band)
                    low, high = bands[band]
                    band_summary = generate_delta_band_summary_per_channel_full_duration(global_raw_ica.copy(),low,high)
            
                    band_response = client.chat.completions.create(
                    model="chatgpt-4o-latest",
                    messages=[
                            {"role": "system", "content": "You are an expert in neuroscience, specializing in EEG analysis and frequency band interpretation."},
                            {"role": "user", "content": f"""Given the following summary of EEG data focused on the 
                             {band} band: {band_summary}, 
                             please analyze the data and provide a short report with conclusions, 
                             specifically focusing on the {band} band. The participant is a {age}-year-old {gender}, 
                             named {name} having following known issues {known_issues}. The participant is taking 
                             medications: {medications}. Write the report in a way that addresses the 
                             participant directly, using their name when appropriate. The report should 
                             be structured into three sections (don't add any other heading/title): Introduction, 
                             Findings, and Conclusion. Do not sugar coat it, make sure to bring up anything alarming 
                             in the data in the conclusion, or any possible dignosis. Don't add aigning off remarks 
                             like, yours sincerely etc. The language should be formal, clear, concise, and suitable 
                             for a primary school-going child (aged {age} years), while maintaining proper report 
                             format. Make sure to explain what the findings suggest about brain activity.
                             Please write the response in British English"""}
                        ]
                        )
                    global_bands_openai[band] = band_response.choices[0].message.content
                #rel power topo openai
                relative_power_topomaps_summary = generate_detailed_relative_power_summary(raw_ica, 
                                                                                            bands, channel_groups)
                rel_pwr_topo_response = main_gpt_call("Relative Power spectra topomaps analysis", relative_power_topomaps_summary,
                                                 name, age, gender, known_issues,medications)
                global_relative_topo_openai = rel_pwr_topo_response
                
                #abs power topo openai
                detailed_absolute_power_summary = generate_detailed_absolute_power_summary(raw, bands, channel_groups)
                abs_pwr_topo_response = main_gpt_call("Absolute Power spectra topomaps analysis", detailed_absolute_power_summary,
                                                 name, age, gender, known_issues,medications)
                global_abs_topo_openai = abs_pwr_topo_response
                #rel spectra openai
                relative_spectra_summary = generate_detailed_relative_spectra_summary(raw_ica, bands)
                rel_spectra_response = main_gpt_call("Relative Power Spectra Analysis (area graphs)", relative_spectra_summary,
                                                 name, age, gender, known_issues,medications)

                global_rel_spectra_openai = rel_spectra_response
                #abs spectra opwnai
                abs_spectra_summary = generate_detailed_absolute_spectra_summary(raw_ica, bands)
                abs_spectra_response = main_gpt_call("Absolute Power spectra analysis (area graphs)", abs_spectra_summary,
                                                 name, age, gender, known_issues,medications)

                global_abs_spectra_openai = abs_spectra_response
                #theta beta ratio openai
                theta_beta_summary = generate_detailed_theta_beta_ratio_summary(raw_ica, bands)
                theta_beta_response = main_gpt_call("Theta/Beta ratio topomap analysis", theta_beta_summary,
                                                 name, age, gender, known_issues,medications)

                global_theta_beta_ratio_openai = theta_beta_response
                #brain mapping openai
                brain_mapping_summary = generate_detailed_brain_mapping_summary(raw_ica, bands)
                brain_mapping_response = main_gpt_call("Brain mapping topomap analysis with increased and decreased activity channels"
                                                       , brain_mapping_summary,
                                                        name, age, gender, known_issues,medications)

                global_brain_mapping_openai = brain_mapping_response
                #occi alpha peak openai
                occi_alpha_peak_summary = generate_detailed_occipital_alpha_peak_summary(raw_ica, alpha_band=(7.5, 14))
                occi_alpha_peak_response = main_gpt_call("EEG data focused on occipital alpha peaks"
                                                       , occi_alpha_peak_summary,
                                                        name, age, gender, known_issues,medications)

                global_occipital_alpha_peak_openai = occi_alpha_peak_response
                
                #chewing openai
                chewing_artifect_summary = generate_detailed_chewing_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['T3', 'T4', 'F7', 'F8'], 
                                                                                                    detect_chewing_artifacts, 
                                                                                                    5)
                chewing_artifect_response = main_gpt_call("EEG data focused on chewing artifact detection"
                                                       , chewing_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_chewing_artifect_openai = chewing_artifect_response
                
                #ecg openai
                ecg_artifect_summary = generate_detailed_ecg_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['T3', 'T4', 'Cz'], 
                                                                                                    detect_ecg_artifacts, 
                                                                                                    5)
                ecg_artifect_response = main_gpt_call("EEG data focused on ECG artifact detection"
                                                       , ecg_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_ecg_artifect_openai = ecg_artifect_response
                #rectus openai
                rectus_artifect_summary = generate_detailed_rectus_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['Fp1', 'Fp2'], 
                                                                                                    detect_rectus_artifacts, 
                                                                                                    5)
                rectus_artifect_response = main_gpt_call("EEG data focused on rectus artifact detection"
                                                       , rectus_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_rectus_artifect_openai = rectus_artifect_response
                #roving eye openai
                roving_artifect_summary = generate_detailed_roving_eye_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['Fp1', 'Fp2'], 
                                                                                                    detect_roving_eye_artifacts, 
                                                                                                    5)
                roving_artifect_response = main_gpt_call("EEG data focused on roving eye artifact detection"
                                                       , roving_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_roving_eye_artifect_openai = roving_artifect_response
                #muscle artifect openai
                muscle_artifect_summary = generate_detailed_muscle_tension_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['T3', 'T4', 'F7', 'F8'], 
                                                                                                    detect_muscle_tension_artifacts, 
                                                                                                    5)
                muscle_artifect_response = main_gpt_call("EEG data focused on muscle tension artifact detection"
                                                       , muscle_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_muscle_tension_artifect_openai = muscle_artifect_response
                #blink artifect openai
                blink_artifect_summary = generate_detailed_blink_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['Fp1', 'Fp2'], 
                                                                                                    detect_blink_artifacts, 
                                                                                                    5)
                blink_artifect_response = main_gpt_call("EEG data focused on blink artifact detection"
                                                       , blink_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_blink_artifect_openai = blink_artifect_response
                #rectus spike artifect openai
                rspike_artifect_summary = generate_detailed_rectus_spike_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['Fp1', 'Fp2'], 
                                                                                                    detect_rectus_spikes_artifacts, 
                                                                                                    5)
                rspike_artifect_response = main_gpt_call("EEG data focused on rectus spikes artifact detection"
                                                       , rspike_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_rectus_spike_artifect_openai = rspike_artifect_response
                #pdr artifect openai
                pdr_artifect_summary = generate_detailed_pdr_artifact_summary_full_duration(raw_ica, 
                                                                                                    ['O1', 'O2'], 
                                                                                                    detect_pdr_artifacts, 
                                                                                                    5)
                pdr_artifect_response = main_gpt_call("EEG data focused on PDR artifact detection"
                                                       , pdr_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_pdr_openai = pdr_artifect_response
                #impedance artifect openai
                impedance_artifect_summary = generate_detailed_impedance_artifact_summary_full_duration(raw_ica, 
                                                                                                    detect_impedance_artifacts, 
                                                                                                    5)
                impedance_artifect_response = main_gpt_call("EEG data focused on impedance artifact detection"
                                                       , impedance_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_impedance_openai = impedance_artifect_response
                #epileptic artifect openai
                epileptic_artifect_summary = generate_detailed_epileptic_pattern_summary_full_duration(raw_ica, 
                                                                                                    detect_epileptic_patterns, 
                                                                                                    5)
                epileptic_artifect_response = main_gpt_call("EEG data focused on epileptic patterns artifact detection"
                                                       , epileptic_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_epileptic_openai = epileptic_artifect_response
                #freq binz openai
                freq_bins_artifect_summary = generate_frequency_bin_summary(raw_ica, frequency_bins)
                freq_bins_artifect_response = main_gpt_call("EEG different frequency bin analysis"
                                                       , freq_bins_artifect_summary,
                                                        name, age, gender, known_issues,medications)

                global_frq_bins_openai = freq_bins_artifect_response
                # Determine the maximum time for the EEG data
                max_time = int(raw.times[-1])

                return render_template('upload_with_topomap_dropdown.html', max_time=max_time)

            except Exception as e:
                print(f"Error processing file: {e}")
                return "Error processing file", 500

    # If the request method is GET, render the upload page
    return render_template('upload_with_topomap_dropdown.html', max_time=0)



@socketio.on('slider_update')
def handle_slider_update(data):
    global global_raw, global_raw_ica

    try:
        start_time = int(data['start_time'])
        plot_type = data['plot_type']
        plot_url = None  # Initialize plot_url to avoid reference error
        openai_res = None

        if plot_type == 'raw' and global_raw:
            fig = global_raw.plot(start=start_time, duration=5, n_channels=19, show=False)
            openai_res = global_raw_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
            print(global_raw_openai)
        elif plot_type == 'cleaned' and global_raw_ica:
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            openai_res = global_raw_ica_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
            print(global_raw_ica_openai)
        elif plot_type == "ica_properties":
            figs = global_ica.plot_properties(global_raw_ica, show=False)
    
            # Save each figure to an image and send them to the client
            plot_urls = []
            for fig in figs:
                img = BytesIO()
                fig.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                #
                #
                #print (f'this is plot URL: {plot_url}')
                plot_urls.append(plot_url)
                

            openai_res = global_ica_components_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
            print(global_ica_components_openai)
        elif plot_type in ["delta", "theta", "alpha", "beta-1", "beta-2", "gamma"]:
            low, high = bands[plot_type]
            band_filtered = global_raw_ica.copy().filter(low, high, fir_design='firwin')
            fig = band_filtered.plot(start=start_time, duration=5, n_channels=19, show=False)
            
            openai_res = global_bands_openai[plot_type]
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "topomaps_relative":
            # Compute PSD and relative power for topomaps
            spectrum = global_raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
            psds, freqs = spectrum.get_data(return_freqs=True)
            band_powers = {}
            for band, (fmin, fmax) in bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_powers[band] = np.mean(psds[:, freq_mask], axis=1)
            total_power = np.sum(list(band_powers.values()), axis=0)
            relative_powers = {band: (power / total_power) * 100 for band, power in band_powers.items()}

            # Plot the topomaps for each band
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for ax, (band, power) in zip(axes, relative_powers.items()):
                plot_topomap(power, global_raw_ica.info, axes=ax, show=False, cmap=custom_cmap)
                ax.set_title(f'{band.capitalize()} Band')
            plt.suptitle('Relative Power Topomaps', fontsize=16)
            plt.colorbar(axes[0].images[0], ax=axes, orientation='horizontal', fraction=0.05, pad=0.07)
            
            openai_res = global_relative_topo_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "topomaps_absolute":
            # Compute PSD and absolute power for topomaps
            spectrum = global_raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
            psds, freqs = spectrum.get_data(return_freqs=True)
            band_powers = {}
            for band, (fmin, fmax) in bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_powers[band] = np.mean(psds[:, freq_mask], axis=1)

            # Plot the topomaps for each band (absolute power)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            for ax, (band, power) in zip(axes, band_powers.items()):
                plot_topomap(power, global_raw_ica.info, axes=ax, show=False, cmap=custom_cmap)
                ax.set_title(f'{band.capitalize()} Band')
            plt.suptitle('Absolute Power Topomaps', fontsize=16)
            plt.colorbar(axes[0].images[0], ax=axes, orientation='horizontal', fraction=0.05, pad=0.07)  
            
            openai_res = global_abs_topo_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == 'relative_spectra':
            spectrum = global_raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
            psds, freqs = spectrum.get_data(return_freqs=True)
            band_powers = {}
            for band, (fmin, fmax) in bands.items():
                freq_mask = (freqs >= fmin) & (freqs <= fmax)
                band_powers[band] = np.mean(psds[:, freq_mask], axis=1)
            total_power = np.sum(list(band_powers.values()), axis=0)
            fig, axes = plt.subplots(len(global_raw_ica.ch_names)//3, 3, figsize=(15, 15), constrained_layout=True)
            fig.suptitle('Relative power spectra (%P)', fontsize=16)
            for idx, (ax, ch_name) in enumerate(zip(axes.flatten(), global_raw_ica.ch_names)):
                ax.plot(freqs, psds[idx] / total_power[idx] * 100, color='black')
                ax.set_title(ch_name)
                for band, (fmin, fmax) in bands.items():
                    band_mask = (freqs >= fmin) & (freqs <= fmax)
                    ax.fill_between(freqs[band_mask], (psds[idx, band_mask] / total_power[idx]) * 100,
                                    color=band_colors[band], alpha=0.5)
                ax.set_xlim([0.5, 50])
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Relative Power (%)')
            openai_res = global_rel_spectra_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == 'absolute_spectra':
            spectrum = global_raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
            psds, freqs = spectrum.get_data(return_freqs=True)
            fig, axes = plt.subplots(len(global_raw_ica.ch_names)//3, 3, figsize=(15, 15), constrained_layout=True)
            fig.suptitle('Absolute power spectra (P)', fontsize=16)
            for idx, (ax, ch_name) in enumerate(zip(axes.flatten(), global_raw_ica.ch_names)):
                ax.plot(freqs, psds[idx], color='black')
                ax.set_title(ch_name)
                for band, (fmin, fmax) in bands.items():
                    band_mask = (freqs >= fmin) & (freqs <= fmax)
                    ax.fill_between(freqs[band_mask], psds[idx, band_mask],
                                    color=band_colors[band], alpha=0.5)
                ax.set_xlim([0.5, 50])
                ax.set_ylim([0, np.max(psds[idx]) * 1.1])
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Absolute Power')
            openai_res = global_abs_spectra_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "theta_beta_ratio":
            # Compute the Theta/Beta ratio
            spectrum = global_raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
            psds, freqs = spectrum.get_data(return_freqs=True)
            theta_power = np.mean(psds[:, (freqs >= bands['theta'][0]) & (freqs <= bands['theta'][1])], axis=1)
            beta_power = np.mean(psds[:, (freqs >= bands['beta-1'][0]) & (freqs <= bands['beta-1'][1])], axis=1)
            theta_beta_ratio = theta_power / beta_power

            # Plot the Theta/Beta ratio using topomaps
            fig, ax = plt.subplots(figsize=(8, 6))
            plot_topomap(theta_beta_ratio, global_raw_ica.info, axes=ax, show=False, cmap=custom_cmap)
            ax.set_title('Theta/Beta Ratio')
            plt.colorbar(ax.images[0], ax=ax, orientation='horizontal', fraction=0.05, pad=0.07)
            
            openai_res = global_theta_beta_ratio_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == 'brain_mapping':
            spectrum = global_raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
            psds, freqs = spectrum.get_data(return_freqs=True)
            theta_power = np.mean(psds[:, (freqs >= bands['theta'][0]) & (freqs <= bands['theta'][1])], axis=1)
            beta_power = np.mean(psds[:, (freqs >= bands['beta-1'][0]) & (freqs <= bands['beta-1'][1])], axis=1)
            theta_beta_ratio = theta_power / beta_power

            fig, ax = plt.subplots()

            # Set background color to white
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')

            # Plot empty topomap with standard 10-20 locations, without shading
            mne.viz.plot_topomap(np.zeros_like(theta_beta_ratio), global_raw_ica.info, axes=ax, show=False, contours=0, cmap=None)

            # Add circles to mark increased and decreased activity similar to image 2
            increased_activity_channels = np.where(theta_beta_ratio > np.mean(theta_beta_ratio) + np.std(theta_beta_ratio))[0]
            decreased_activity_channels = np.where(theta_beta_ratio < np.mean(theta_beta_ratio) - np.std(theta_beta_ratio))[0]

            # Draw circles for decreased activity (Red) and increased activity (Green)
            for ch_idx in increased_activity_channels:
                loc = global_raw_ica.info['chs'][ch_idx]['loc'][:2]
                ax.plot(loc[0], loc[1], 'o', markerfacecolor='green', markeredgecolor='green', markersize=15)
                ax.annotate(global_raw_ica.ch_names[ch_idx], xy=loc, xytext=(10, 10), textcoords='offset points', color='green', fontsize=10, fontweight='bold')
    
            for ch_idx in decreased_activity_channels:
                loc = global_raw_ica.info['chs'][ch_idx]['loc'][:2]
                ax.plot(loc[0], loc[1], 'o', markerfacecolor='red', markeredgecolor='red', markersize=15)
                ax.annotate(global_raw_ica.ch_names[ch_idx], xy=loc, xytext=(10, 10), textcoords='offset points', color='red', fontsize=10, fontweight='bold')

            # Remove any unnecessary elements
            ax.axis('off')
            ax.set_title('Theta/Beta Ratio Topographic Map', color='black')
            
            openai_res = global_brain_mapping_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "occipital_alpha_peak":
            # Compute the PSDs for Occipital channels
            # Compute PSD and relative power for topomaps and spectra
            spectrum = global_raw_ica.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
            psds, freqs = spectrum.get_data(return_freqs=True)
            occipital_channels = ['O1', 'O2']
            alpha_band = (7.5, 14)
            occipital_psds = psds[[global_raw_ica.ch_names.index(ch) for ch in occipital_channels], :]
            alpha_mask = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
            alpha_freqs = freqs[alpha_mask]
            occipital_psds_alpha = occipital_psds[:, alpha_mask]
            alpha_peaks = {}
            for idx, ch_name in enumerate(occipital_channels):
                alpha_psd = occipital_psds_alpha[idx]
                peak_idx = np.argmax(alpha_psd)
                alpha_peaks[ch_name] = alpha_freqs[peak_idx]

            fig, ax = plt.subplots()
            for idx, ch_name in enumerate(alpha_peaks.keys()):
                ax.plot(alpha_freqs, occipital_psds_alpha[idx], label=f'{ch_name} (Peak: {alpha_peaks[ch_name]:.2f} Hz)')
            ax.set_title('Occipital Alpha Peak Analysis')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Power')
            ax.legend()
            

            openai_res = global_occipital_alpha_peak_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "chewing_artifact_detection":
            # Chewing artifact detection logic
            chewing_channels = ['T3', 'T4','T5','T6']  # Channels focused on detecting rectus artifacts
            chewing_segments = detect_chewing_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in chewing_segments:
            #    ax.axvspan(segment[0], segment[1], color='red', alpha=0.3, label='Chewing Artifact')

            #ax.legend()
            ax = fig.axes[0]
            for segment in chewing_segments:
                for ch in chewing_channels:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='red', alpha=0.3, label=f'Chewing Artifact ({ch})', ymin=ch_index / 19, ymax=(ch_index + 1) / 19)

            #ax.legend()
            openai_res = global_chewing_artifect_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "ecg_artifact_detection":
            # ECG artifact detection logic
            ecg_channels = ['T3', 'T4', 'Cz']  # Channels focused on detecting rectus artifacts
            ecg_segments = detect_ecg_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in ecg_segments:
             #   ax.axvspan(segment[0], segment[1], color='blue', alpha=0.3, label='ECG Artifact')

            #ax.legend()
            # Get the current axes and highlight detected segments only on rectus channels
            ax = fig.axes[0]
            for segment in ecg_segments:
                for ch in ecg_channels:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='blue', alpha=0.3, label=f'ECG Artifact ({ch})', ymin=ch_index / 19, ymax=(ch_index + 1) / 19)
            
            openai_res = global_ecg_artifect_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "rectus_artifact_detection":
            # Rectus artifact detection logic
            #rectus_segments = detect_rectus_artifacts(global_raw, start_time, duration=5)

            # Plot EEG data
            #fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            #plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in rectus_segments:
             #   ax.axvspan(segment[0], segment[1], color='orange', alpha=0.3, label='Rectus Artifact')

            #ax.legend()
            rectus_channels = ['O1', 'O2']  # Channels focused on detecting rectus artifacts
            rectus_segments = detect_rectus_artifacts(global_raw_ica, start_time, duration=5)

            # Plot all channels
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments only on rectus channels
            ax = fig.axes[0]
            for segment in rectus_segments:
                for ch in rectus_channels:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='orange', alpha=0.3, label=f'Rectus Artifact ({ch})', ymin=ch_index / 19, ymax=(ch_index + 1) / 19)

            #ax.legend()

            openai_res = global_rectus_artifect_openai
            openai_res = re.sub(r'[*#]', '', openai_res) 
        elif plot_type == "roving_eye_artifact_detection":
            roving_channels = ['O1', 'O2']  # Channels focused on detecting rectus artifacts
            # Roving eye artifact detection logic
            roving_segments = detect_roving_eye_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in roving_segments:
            #    ax.axvspan(segment[0], segment[1], color='purple', alpha=0.3, label='Roving Eye Artifact')

            #ax.legend()  
            # Get the current axes and highlight detected segments only on rectus channels
            ax = fig.axes[0]
            for segment in roving_channels:
                for ch in roving_segments:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='purple', alpha=0.3, label=f'Roving Artifact ({ch})') 
                    
            openai_res = global_roving_eye_artifect_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "muscle_tension_artifact_detection":
            # Muscle tension artifact detection logic
            muscle_channels = ['T3', 'T4','T5','T6']  # Channels focused on detecting rectus artifacts
            muscle_segments = detect_muscle_tension_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in muscle_segments:
            #    ax.axvspan(segment[0], segment[1], color='magenta', alpha=0.3, label='Muscle Tension Artifact')
            #
            #ax.legend()
            ax = fig.axes[0]
            for segment in muscle_segments:
                for ch in muscle_channels:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='magenta', alpha=0.3, label=f'Blink Artifact ({ch})', ymin=ch_index / 19, ymax=(ch_index + 1) / 19)
                    
            openai_res = global_muscle_tension_artifect_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "blink_artifact_detection":
            # Blink artifact detection logic
            blink_channels = ['O1', 'O2']  # Channels focused on detecting rectus artifacts
            blink_segments = detect_blink_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in blink_segments:
            #    ax.axvspan(segment[0], segment[1], color='yellow', alpha=0.3, label='Blink Artifact')

            #ax.legend()
            ax = fig.axes[0]
            for segment in blink_segments:
                for ch in blink_channels:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='yellow', alpha=0.3, label=f'Blink Artifact ({ch})', ymin=ch_index / 19, ymax=(ch_index + 1) / 19)
                    
            openai_res = global_blink_artifect_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "rectus_spike_artifact_detection":
            # Rectus spike artifact detection logic
            rectus_spike_channels = ['O1', 'O2']  # Channels focused on detecting rectus artifacts
            
            rectus_spike_segments = detect_rectus_spikes_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in rectus_spike_segments:
             #   ax.axvspan(segment[0], segment[1], color='purple', alpha=0.3, label='Rectus Spike Artifact')
            ax = fig.axes[0]
            for segment in rectus_spike_segments:
                for ch in rectus_spike_channels:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='cyan', alpha=0.3, label=f'Blink Artifact ({ch})', ymin=ch_index / 19, ymax=(ch_index + 1) / 19)
            openai_res = global_rectus_spike_artifect_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        # PDR artifact detection logic
        elif plot_type == "pdr_artifact_detection":
            pdr_channels = ['Fp1', 'Fp2']  # Channels focused on detecting rectus artifacts
            
            pdr_segments = detect_pdr_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            #ax = fig.axes[0]
            #for segment in pdr_segments:
               # ax.axvspan(segment[0], segment[1], color='green', alpha=0.3, label='PDR Artifact')
            ax = fig.axes[0]
            for segment in pdr_segments:
                for ch in pdr_channels:
                    ch_index = global_raw_ica.ch_names.index(ch)
                    ax.axvspan(segment[0], segment[1], color='olive', alpha=0.3, label=f'Blink Artifact ({ch})', ymin=ch_index / 19, ymax=(ch_index + 1) / 19)
                    
            openai_res = global_pdr_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "impedance_artifact_detection":
            # Impedance artifact detection logic
            impedance_segments = detect_impedance_artifacts(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Get the current axes and highlight detected segments
            ax = fig.axes[0]
            for segment in impedance_segments:
                ax.axvspan(segment[0], segment[1], color='steelblue', alpha=0.3, label='Impedance Artifact')
                
            openai_res = global_impedance_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "epileptic_pattern_detection":
            epileptic_segments = detect_epileptic_patterns(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Highlight detected epileptic patterns
            ax = fig.axes[0]
            for segment in epileptic_segments:
                ax.axvspan(segment[0], segment[1], color='purple', alpha=0.3, label='Epileptic Pattern')
                
            openai_res = global_epileptic_openai
            openai_res = re.sub(r'[*#]', '', openai_res)
        elif plot_type == "dipole_analysis":
            fig = perform_dipole_analysis()
            openai_res = None
        elif plot_type == 'frequency_bins':
            fig = plot_frequency_bins(global_raw, frequency_bins)
                
            openai_res = global_frq_bins_openai
            openai_res = re.sub(r'[*#]', '', openai_res)

        else:
            return  # No action if the plot type is unrecognized or data is not loaded

        # Convert the plot to an image and send it to the client
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Emit the updated plot back to the client
        #emit('update_plot', {'plot_url': plot_url})
        # Include raw report to send back
        emit('update_plot', {'plot_url': plot_url, 'raw_report': openai_res})
        



    except Exception as e:
        print(f"Error generating plot: {e}")
        emit('update_plot', {'plot_url': None})  # Send a fallback response in case of error
        

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port= 5000)#, use_reloader=False)
