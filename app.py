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


# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Use global variables to store raw and ICA-cleaned EEG data
global_raw = None
global_raw_ica = None
global_ica = None

# Route for file upload and main dashboard
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global global_raw, global_raw_ica, global_ica

    if request.method == 'POST':
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

        if plot_type == 'raw' and global_raw:
            fig = global_raw.plot(start=start_time, duration=5, n_channels=19, show=False)
        elif plot_type == 'cleaned' and global_raw_ica:
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
        elif plot_type == "ica_properties":
            figs = global_ica.plot_properties(global_raw_ica, picks=global_ica.exclude, show=False)
    
            # Save each figure to an image and send them to the client
            plot_urls = []
            for fig in figs:
                img = BytesIO()
                fig.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode()
                print (f'this is plot URL: {plot_url}')
                plot_urls.append(plot_url)
        elif plot_type in ["delta", "theta", "alpha", "beta-1", "beta-2", "gamma"]:
            low, high = bands[plot_type]
            band_filtered = global_raw_ica.copy().filter(low, high, fir_design='firwin')
            fig = band_filtered.plot(start=start_time, duration=5, n_channels=19, show=False)
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
        elif plot_type == "epileptic_pattern_detection":
            epileptic_segments = detect_epileptic_patterns(global_raw_ica, start_time, duration=5)

            # Plot EEG data
            fig = global_raw_ica.plot(start=start_time, duration=5, n_channels=19, show=False)
            plt.close(fig)  # Close the interactive plot to prevent blocking

            # Highlight detected epileptic patterns
            ax = fig.axes[0]
            for segment in epileptic_segments:
                ax.axvspan(segment[0], segment[1], color='purple', alpha=0.3, label='Epileptic Pattern')
        elif plot_type == "dipole_analysis":
            fig = perform_dipole_analysis()
        elif plot_type == 'frequency_bins':
                frequency_bins = {
                    'Bin 1 (~0.98 Hz)': (0.97, 0.99),
                    'Bin 2 (~1.95 Hz)': (1.94, 1.96),
                    'Bin 3 (~8.30 Hz)': (8.29, 8.31),
                    'Bin 4 (~26.51 Hz)': (26.50, 26.62),
                    'Bin 5 (~30.76 Hz)': (30.75, 30.77)
                }
                fig = plot_frequency_bins(global_raw, frequency_bins)

        else:
            return  # No action if the plot type is unrecognized or data is not loaded

        # Convert the plot to an image and send it to the client
        img = BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        # Emit the updated plot back to the client
        emit('update_plot', {'plot_url': plot_url})



    except Exception as e:
        print(f"Error generating plot: {e}")
        emit('update_plot', {'plot_url': None})  # Send a fallback response in case of error
        

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)#, use_reloader=False)
