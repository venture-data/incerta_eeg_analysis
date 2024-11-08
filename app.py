from flask import Flask, render_template, request, flash, redirect
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
from mne.preprocessing import create_eog_epochs
from autoreject import AutoReject
from pyprep.prep_pipeline import PrepPipeline  # For ASR
from mne.preprocessing import ICA
import pywt
import pandas as pd
from datetime import datetime
import os
from pathlib import Path


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
    


# Use global variables to store raw and ICA-cleaned EEG data
global_raw = None
global_raw_ica = None
global_ica = None
global_channel_dict = None
global_decreased_combined_power_fig = None
global_decreased_activation_bandwise = None
global_increased_combined_power_fig = None
global_increased_activation_bandwise = None
global_abs_power_spectra_lines = None
global_abs_power_spectra_topo = None
global_rel_power_spectra_lines = None
global_rel_power_spectra_topo = None
global_theta_beta_ratio= None
global_asymmetry_fig = None
global_brodmann_dorsolateral = None
global_brodmann_dorsolateral_combined = None
global_brodmann_findings = None



# OpenAI API Key setup
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global global_raw, global_raw_ica, global_ica, global_channel_dict, global_decreased_combined_power_fig, \
    global_decreased_activation_bandwise, global_increased_combined_power_fig, global_increased_activation_bandwise, \
    global_abs_power_spectra_lines, global_abs_power_spectra_topo, global_rel_power_spectra_lines,global_rel_power_spectra_topo, \
    global_theta_beta_ratio, global_asymmetry_fig, global_brodmann_dorsolateral, global_brodmann_dorsolateral_combined, \
    global_brodmann_findings

    if request.method == 'POST':
       
        uploaded_file = request.files['file']        
        
        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.filename)[1]
            if file_ext.lower() != '.edf':
                flash('Invalid file format! Please upload a .edf file.', 'error')
                return redirect(request.url)
            # Handle file upload
            if 'file' in request.files:
                try:
                    f = request.files['file']
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                    f.save(filepath)
                
                    # Load the EEG data using MNE
                    raw = mne.io.read_raw_edf(filepath, preload=True, encoding='latin1')
                
                    channel_names = raw.info['ch_names']
                    print(channel_names)

                    mapping = {ch: ch.replace('EEG ', '') for ch in raw.info['ch_names'] if 'EEG ' in ch}
                    raw.rename_channels(mapping)
                    
                    # Step 1.2: Set non-EEG channels (e.g., ECG, Bio channels)
                    non_eeg_channels = ['Bio1-2', 'Bio3-4', 'ECG', 'Bio4', 'VSyn', 'ASyn', 'LABEL']
                    raw.set_channel_types({ch: 'misc' for ch in non_eeg_channels})
                    
                    """### Selecting desired 19 channels"""
                    
                    desired_channels = [
                        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5',
                        'P3', 'Pz', 'P4', 'T6', 'O1', 'O2'
                    ]
                    raw.pick_channels(desired_channels)
                    
                    montage = mne.channels.make_standard_montage('standard_1020')
                    raw.set_montage(montage)
                    
                    print(raw.info)
                    # Define the duration to remove (in seconds)
                    remove_duration = 10  # seconds
                    
                    # Define the sampling frequency of your data
                    sampling_frequency = raw.info['sfreq']  # Get sampling frequency from the raw object
                    
                    # Calculate the number of samples to remove
                    samples_to_remove = int(remove_duration * sampling_frequency)
                    
                    # Get the total number of samples in the raw data
                    total_samples = raw.n_times

                    # Check if the total samples are sufficient
                    if total_samples > 2 * samples_to_remove:
                        # Trim the raw data
                        trimmed_data = raw.copy().crop(tmin=samples_to_remove / sampling_frequency,
                                                                tmax=(total_samples - samples_to_remove) / sampling_frequency)
                    
                        # Optionally, you can update cleaned_raw with the trimmed data
                        raw = trimmed_data
                    else:
                        print("Not enough data to remove the specified duration.")

                    # Print new data info
                    print(f"New data length: {raw.n_times} samples, "
                          f"Duration: {raw.times[-1] - raw.times[0]:.2f} seconds")
                    
                    """## Filtering"""
                    
                    raw.filter(l_freq=0.53, h_freq=50)
                    raw.notch_filter(freqs=50)  # Assuming notch at 50 Hz to remove power line noise (modify if 50 Hz)

                    global_raw = raw

                    """## ICA"""

                    ica = ICA(n_components=15, random_state=13, max_iter='auto')
                    ica.fit(raw)

                    ica.exclude = [3]
                    cleaned_raw = ica.apply(raw.copy())  # Apply the ICA to the raw data

                    # Mark Fp1 and Fp2 as bad channels
                    cleaned_raw.info['bads'] = ['F7']
                    
                    # Interpolate bad channels
                    cleaned_raw.interpolate_bads(reset_bads=True)
                    # # Manually inspect the ICA components to find additional artifact-related components
                    # ica.plot_components()
                    
                    channel_names = cleaned_raw.info['ch_names']  # List of channel names
                    channel_dict = {name: idx for idx, name in enumerate(channel_names)}  # Create a dictionary with channel names and their indices
                
                    psd_object = cleaned_raw.compute_psd(fmin=0.5, fmax=40, n_fft=1024)  # This returns a PSD object
                    psds = psd_object.get_data()  # Extract the power spectral density
                    freqs = psd_object.freqs  # Get the frequency values

                    """# Set Band Frequency"""

                    # Define frequency bands
                    delta_band = (1.5, 4)      # Delta: 1.5-4 Hz
                    theta_band = (4, 7.5)      # Theta: 4-7.5 Hz
                    alpha_band = (7.5, 14)     # Alpha: 7.5- Hz
                    beta1_band = (14, 20)      # Beta1: 13-20 Hz
                    beta2_band = (20, 30)      # Beta2: 20-30 Hz
                    gamma_band = (30, 40)      # Gamma: 30-40 Hz

                    # Calculate power for each band
                    delta_power = np.mean(psds[:, (freqs >= delta_band[0]) & (freqs <= delta_band[1])], axis=1)
                    theta_power = np.mean(psds[:, (freqs >= theta_band[0]) & (freqs <= theta_band[1])], axis=1)
                    alpha_power = np.mean(psds[:, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])], axis=1)
                    beta1_power = np.mean(psds[:, (freqs >= beta1_band[0]) & (freqs <= beta1_band[1])], axis=1)
                    beta2_power = np.mean(psds[:, (freqs >= beta2_band[0]) & (freqs <= beta2_band[1])], axis=1)
                    gamma_power = np.mean(psds[:, (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])], axis=1)

                    """## Relative Power"""

                    total_power = np.sum(psds, axis=1)
                    relative_delta_power = (delta_power / total_power) * 100
                    relative_theta_power = (theta_power / total_power) * 100
                    relative_alpha_power = (alpha_power / total_power) * 100
                    relative_beta1_power = (beta1_power / total_power) * 100
                    relative_beta2_power = (beta2_power / total_power) * 100
                    relative_gamma_power = (gamma_power / total_power) * 100
                    zones = {
                    'Orbital Frontal': ['Fp1', 'Fp2'],
                    'Frontal': ['F7', 'F3', 'Fz', 'F4', 'F8'],
                    'Temporal': ['T3', 'T4', 'T5', 'T6'],
                    'Central': ['C3', 'Cz', 'C4'],
                    'Parietal': ['P3', 'Pz', 'P4'],
                    'Occipital': ['O1', 'O2']
                    }

                    """## Increased Power"""

                    def find_increased_power(power_band, threshold_percent=60):
                        threshold = np.percentile(power_band, threshold_percent)
                        increased_channels = [ch for ch, power in zip(channel_names, power_band) if power >= threshold]
                        return increased_channels

                    increased_delta = find_increased_power(delta_power)
                    increased_theta = find_increased_power(theta_power)
                    increased_alpha = find_increased_power(alpha_power)
                    increased_beta1 = find_increased_power(beta1_power)
                    increased_beta2 = find_increased_power(beta2_power)
                    increased_gamma = find_increased_power(gamma_power)

                    """## Group Findings by Zones"""

                    def map_channels_to_zones(increased_channels):
                        zone_report = {zone: [] for zone in zones}
                        for channel in increased_channels:
                            for zone, zone_channels in zones.items():
                                if channel in zone_channels:
                                    zone_report[zone].append(channel)
                        return zone_report

                    increased_delta_zones = map_channels_to_zones(increased_delta)
                    increased_theta_zones = map_channels_to_zones(increased_theta)
                    increased_alpha_zones = map_channels_to_zones(increased_alpha)
                    increased_beta1_zones = map_channels_to_zones(increased_beta1)
                    increased_beta2_zones = map_channels_to_zones(increased_beta2)
                    increased_gamma_zones = map_channels_to_zones(increased_gamma)

                    def generate_report(increased_zones, band_name):
                        report = f"Increased relative {band_name} power spectra in the "
                        zones_with_increases = {zone: channels for zone, channels in increased_zones.items() if channels}

                        for i, (zone, channels) in enumerate(zones_with_increases.items()):
                            channels_str = ', '.join(channels)  # List the channels in the zone
                            if i == len(zones_with_increases) - 1:
                                report += f"{zone.lower()} {channels_str} areas."
                            else:
                                report += f"{zone.lower()} {channels_str}, "
                        return report

                    """## Decreased Power"""

                    def find_decreased_power(power_band, threshold_percent=40):
                        threshold = np.percentile(power_band, threshold_percent)  # Use a lower threshold for decreased power
                        decreased_channels = [ch for ch, power in zip(channel_names, power_band) if power <= threshold]
                        return decreased_channels
                    
                    decreased_delta = find_decreased_power(delta_power)
                    decreased_theta = find_decreased_power(theta_power)
                    decreased_alpha = find_decreased_power(alpha_power)
                    decreased_beta1 = find_decreased_power(beta1_power)
                    decreased_beta2 = find_decreased_power(beta2_power)
                    decreased_gamma = find_decreased_power(gamma_power)

                    def map_channels_to_zones(decreased_channels):
                        zone_report = {zone: [] for zone in zones}
                        for channel in decreased_channels:
                            for zone, zone_channels in zones.items():
                                if channel in zone_channels:
                                    zone_report[zone].append(channel)
                        return zone_report

                    decreased_delta_zones = map_channels_to_zones(decreased_delta)
                    decreased_theta_zones = map_channels_to_zones(decreased_theta)
                    decreased_alpha_zones = map_channels_to_zones(decreased_alpha)
                    decreased_beta1_zones = map_channels_to_zones(decreased_beta1)
                    decreased_beta2_zones = map_channels_to_zones(decreased_beta2)
                    decreased_gamma_zones = map_channels_to_zones(decreased_gamma)

                    def generate_decrease_report(decreased_zones, band_name):
                        report = f"decreased relative {band_name} power spectra in the "
                        zones_with_decreases = {zone: channels for zone, channels in decreased_zones.items() if channels}

                        for i, (zone, channels) in enumerate(zones_with_decreases.items()):
                            channels_str = ', '.join(channels)  # List the channels in the zone
                            if i == len(zones_with_decreases) - 1:
                                report += f"{zone.lower()} {channels_str} areas."
                            else:
                                report += f"{zone.lower()} {channels_str}, "
                        return report

                    """## Generate Reports"""

                    delta_report = generate_report(increased_delta_zones, "delta")
                    theta_report = generate_report(increased_theta_zones, "theta")
                    alpha_report = generate_report(increased_alpha_zones, "alpha")
                    beta1_report = generate_report(increased_beta1_zones, "beta1")
                    beta2_report = generate_report(increased_beta2_zones, "beta2")
                    gamma_report = generate_report(increased_gamma_zones, "gamma")

                    delta_decrease_report = generate_decrease_report(decreased_delta_zones, "delta")
                    theta_decrease_report = generate_decrease_report(decreased_theta_zones, "theta")
                    alpha_decrease_report = generate_decrease_report(decreased_alpha_zones, "alpha")
                    beta1_decrease_report = generate_decrease_report(decreased_beta1_zones, "beta1")
                    beta2_decrease_report = generate_decrease_report(decreased_beta2_zones, "beta2")
                    gamma_decrease_report = generate_decrease_report(decreased_gamma_zones, "gamma")

                    # Sum power across all bands for each channel
                    combined_power = delta_power + theta_power + alpha_power + beta1_power + beta2_power + gamma_power

                    # Function to find channels with decreased combined power based on a threshold percentile
                    def find_decreased_power_combined(power, threshold_percent=40):
                        threshold = np.percentile(power, threshold_percent)
                        decreased_channels = [ch for ch, power_value in zip(channel_names, power) if power_value <= threshold]
                        return decreased_channels

                    # Find channels with decreased combined power
                    decreased_combined = find_decreased_power_combined(combined_power)

                    # Load the standard 10-20 montage for reference
                    montage = mne.channels.make_standard_montage('standard_1020')

                    # Extract positions for the channels in the montage
                    custom_channel_positions = {ch: montage.get_positions()['ch_pos'][ch] for ch in channel_names}

                    # Convert channel positions to an array for plotting
                    positions = np.array([custom_channel_positions[ch] for ch in channel_names])

                    # Function to plot brain activation based on combined decreased power in a single plot
                    def plot_combined_brain_activation(decreased_channels):
                        # Create a figure and axis
                        fig, ax = plt.subplots(figsize=(8, 8))
                        
                        # Set axis limits explicitly for more control
                        ax.set_xlim(-0.2, 0.2)
                        ax.set_ylim(-0.2, 0.2)
                        ax.set_aspect('equal')

                        # Draw the head circle
                        head_circle = plt.Circle((0, 0), 0.12, color='black', fill=False, linewidth=2)
                        ax.add_patch(head_circle)

                        # Draw perpendicular diameters (dotted lines)
                        ax.plot([-0.15, 0.15], [0, 0], 'k--', linewidth=1.5)
                        ax.plot([0, 0], [-0.15, 0.15], 'k--', linewidth=1.5)

                        # Plot each channel's position and label
                        for i, ch_name in enumerate(channel_names):
                            pos = positions[i]
                            ax.text(pos[0], pos[1], ch_name, fontsize=10, ha='center', va='center')

                        # Highlight channels with decreased combined power
                        for ch_name in decreased_channels:
                            pos = custom_channel_positions[ch_name]
                            circle = plt.Circle((pos[0], pos[1]), 0.02, color='red', fill=False, linewidth=1.5)
                            ax.add_patch(circle)

                        # Title and formatting
                        ax.set_title('Decreased Combined Power', fontsize=14)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                        # Return the figure object
                        return fig

                    # Use the function and store the figure globally
                    global_decreased_combined_power_fig = plot_combined_brain_activation(decreased_combined)

                    """## R1.1 Decreased Activation Bandwise"""

                    # Function to find decreased power channels based on a threshold percentile
                    def find_decreased_power(power_band, threshold_percent=40):
                        threshold = np.percentile(power_band, threshold_percent)  # Use a lower threshold for decreased power
                        decreased_channels = [ch for ch, power in zip(channel_names, power_band) if power <= threshold]
                        return decreased_channels

                    # Find increased power channels for all bands
                    decreased_delta = find_decreased_power(delta_power)
                    decreased_theta = find_decreased_power(theta_power)
                    decreased_alpha = find_decreased_power(alpha_power)
                    decreased_beta1 = find_decreased_power(beta1_power)
                    decreased_beta2 = find_decreased_power(beta2_power)
                    decreased_gamma = find_decreased_power(gamma_power)

                    # Create a custom 10-20 EEG montage using only the selected channels
                    montage = mne.channels.make_standard_montage('standard_1020')

                    # Custom montage with only the relevant channels
                    custom_channel_positions = {ch: montage.get_positions()['ch_pos'][ch] for ch in channel_names}

                    # Manually extract positions for plotting
                    positions = np.array([custom_channel_positions[ch] for ch in channel_names])

                    # Helper function to plot brain activation for a specific band in a subplot
                    def plot_brain_activation(ax, decreased_channels, band_name, type):
                        ax.set_xlim(-0.2, 0.2)  # Set axis limits explicitly for more control
                        ax.set_ylim(-0.2, 0.2)
                        ax.set_aspect('equal')  # Ensure circles are round

                        # Draw the head circle
                        head_circle = plt.Circle((0, 0), 0.12, color='black', fill=False, linewidth=2)
                        ax.add_patch(head_circle)

                        # Draw perpendicular diameters (dotted lines)
                        ax.plot([-0.15, 0.15], [0, 0], 'k--', linewidth=1.5)
                        ax.plot([0, 0], [-0.15, 0.15], 'k--', linewidth=1.5)

                        # Plot each channel's position and label
                        for i, ch_name in enumerate(channel_names):
                            pos = positions[i]
                            ax.text(pos[0], pos[1], ch_name, fontsize=10, ha='center', va='center')

                        # Highlight channels with decreased activation
                        for ch_name in decreased_channels:
                            pos = custom_channel_positions[ch_name]
                            if type == 'decreased':
                                colour = 'red'
                            else:
                                colour = 'green'
                            circle = plt.Circle((pos[0], pos[1]), 0.02, color=colour, fill=False, linewidth=1.5)
                            ax.add_patch(circle)

                        # Add title and remove axis ticks and labels
                        ax.set_title(f'{band_name} Power', fontsize=14)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                    # Function to generate and store the figure in global_decreased_activation_bandwise
                    def plot_decreased_activation_bandwise():
                        # Create a figure with 2 rows and 3 columns, adjust size
                        fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figure size

                        # Plot brain activation for each frequency band in its respective subplot
                        plot_brain_activation(axs[0, 0], decreased_delta, 'Delta','decreased')
                        plot_brain_activation(axs[0, 1], decreased_theta, 'Theta','decreased')
                        plot_brain_activation(axs[0, 2], decreased_alpha, 'Alpha','decreased')
                        plot_brain_activation(axs[1, 0], decreased_beta1, 'Beta1','decreased')
                        plot_brain_activation(axs[1, 1], decreased_beta2, 'Beta2','decreased')
                        plot_brain_activation(axs[1, 2], decreased_gamma, 'Gamma','decreased')

                        # Add a title for the entire figure
                        plt.suptitle('Decreased Power', fontsize=16)
                        # Adjust spacing to reduce space between subplots and improve overall plot size
                        plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)

                        # Return the figure
                        return fig

                    # Call the function and store the figure globally
                    global_decreased_activation_bandwise = plot_decreased_activation_bandwise()


                    # Function to find increased power channels based on a threshold percentile
                    def find_increased_power(power_band, threshold_percent=60):
                        threshold = np.percentile(power_band, threshold_percent)
                        increased_channels = [ch for ch, power in zip(channel_names, power_band) if power >= threshold]
                        return increased_channels

                    # Find increased power channels for all bands
                    increased_delta = find_increased_power(delta_power)
                    increased_theta = find_increased_power(theta_power)
                    increased_alpha = find_increased_power(alpha_power)
                    increased_beta1 = find_increased_power(beta1_power)
                    increased_beta2 = find_increased_power(beta2_power)
                    increased_gamma = find_increased_power(gamma_power)

                    # Create a custom 10-20 EEG montage using only the selected channels
                    montage = mne.channels.make_standard_montage('standard_1020')

                    # Custom montage with only the relevant channels
                    custom_channel_positions = {ch: montage.get_positions()['ch_pos'][ch] for ch in channel_names}

                    # Manually extract positions for plotting
                    positions = np.array([custom_channel_positions[ch] for ch in channel_names])

                    # Function to generate and store the figure in global_decreased_activation_bandwise
                    def plot_increased_activation_bandwise():
                        # Create a figure with 2 rows and 3 columns, adjust size
                        fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figure size

                        # Plot brain activation for each frequency band in its respective subplot
                        plot_brain_activation(axs[0, 0], increased_delta, 'Delta','increased')
                        plot_brain_activation(axs[0, 1], increased_theta, 'Theta','increased')
                        plot_brain_activation(axs[0, 2], increased_alpha, 'Alpha','increased')
                        plot_brain_activation(axs[1, 0], increased_beta1, 'Beta1','increased')
                        plot_brain_activation(axs[1, 1], increased_beta2, 'Beta2','increased')
                        plot_brain_activation(axs[1, 2], increased_gamma, 'Gamma','increased')

                        # Add a title for the entire figure
                        plt.suptitle('Increased Power', fontsize=16)
                        # Adjust spacing to reduce space between subplots and improve overall plot size
                        plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)

                        # Return the figure
                        return fig

                    # Call the function and store the figure globally
                    global_increased_activation_bandwise = plot_increased_activation_bandwise()


                    # Sum power across all bands for each channel
                    combined_power = delta_power + theta_power + alpha_power + beta1_power + beta2_power + gamma_power

                    # Function to find channels with increased combined power based on a threshold percentile
                    def find_increased_power_combined(power, threshold_percent=60):
                        threshold = np.percentile(power, threshold_percent)
                        increased_channels = [ch for ch, power_value in zip(channel_names, power) if power_value >= threshold]
                        return increased_channels

                    # Find channels with increased combined power
                    increased_combined = find_increased_power_combined(combined_power)

                    # Load the standard 10-20 montage for reference
                    montage = mne.channels.make_standard_montage('standard_1020')

                    # Extract positions for the channels in the montage
                    custom_channel_positions = {ch: montage.get_positions()['ch_pos'][ch] for ch in channel_names}

                    # Convert channel positions to an array for plotting
                    positions = np.array([custom_channel_positions[ch] for ch in channel_names])


                    # Function to plot combined brain activation and return the figure
                    def plot_increased_combined_power():
                        # Create a figure and axis
                        fig, ax = plt.subplots(figsize=(8, 8))

                        # Set axis limits explicitly for more control
                        ax.set_xlim(-0.2, 0.2)
                        ax.set_ylim(-0.2, 0.2)
                        ax.set_aspect('equal')

                        # Draw the head circle
                        head_circle = plt.Circle((0, 0), 0.12, color='black', fill=False, linewidth=2)
                        ax.add_patch(head_circle)

                        # Draw perpendicular diameters (dotted lines)
                        ax.plot([-0.15, 0.15], [0, 0], 'k--', linewidth=1.5)
                        ax.plot([0, 0], [-0.15, 0.15], 'k--', linewidth=1.5)

                        # Plot each channel's position and label
                        for i, ch_name in enumerate(channel_names):
                            pos = positions[i]
                            ax.text(pos[0], pos[1], ch_name, fontsize=10, ha='center', va='center')

                        # Highlight channels with increased combined power
                        for ch_name in increased_combined:
                            pos = custom_channel_positions[ch_name]
                            circle = plt.Circle((pos[0], pos[1]), 0.02, color='green', fill=False, linewidth=1.5)
                            ax.add_patch(circle)

                        # Title and formatting
                        ax.set_title('Increased Combined Power', fontsize=14)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        for spine in ax.spines.values():
                            spine.set_visible(False)

                        # Return the figure object
                        return fig

                    # Call the function and store the figure globally
                    global_increased_combined_power_fig = plot_increased_combined_power()
                    
                    num_channels = len(channel_names)
                    # Function to plot relative power lines and return the figure
                    def plot_absolute_power_lines():
                        # Determine the number of rows needed for the grid of subplots
                        # Number of channels
                        num_channels = len(channel_names)

                        
                        ncols = 3
                        nrows = (num_channels + ncols - 1) // ncols  # Calculate rows based on number of channels

                        # Create a figure with subplots arranged in a grid
                        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows), constrained_layout=True)

                        # Set up normalization manually
                        vmin, vmax = 0, 15  # Adjust these values as needed
                        cmap = 'jet'  # Colormap

                        for i, ch_name in enumerate(channel_names):
                            ax = axs[i // ncols, i % ncols]  # Get the subplot axis

                            # Plot the power spectrum for the current channel
                            ax.plot(freqs, psds[i], color='black', linewidth=1, label=f'{ch_name} Power Spectrum')

                            # Fill each frequency band with colors
                            ax.fill_between(freqs, psds[i], where=(freqs >= 0.5) & (freqs <= 4), color='brown', alpha=1.0, label='Delta Band')
                            ax.fill_between(freqs, psds[i], where=(freqs > 4) & (freqs <= 8), color='orange', alpha=1.0, label='Theta Band')
                            ax.fill_between(freqs, psds[i], where=(freqs > 8) & (freqs <= 13), color='blue', alpha=1.0, label='Alpha Band')
                            ax.fill_between(freqs, psds[i], where=(freqs > 13) & (freqs <= 20), color='green', alpha=1.0, label='Beta1 Band')
                            ax.fill_between(freqs, psds[i], where=(freqs > 20) & (freqs <= 30), color='yellow', alpha=1.0, label='Beta2 Band')
                            ax.fill_between(freqs, psds[i], where=(freqs > 30) & (freqs <= 40), color='cyan', alpha=1.0, label='Gamma Band')

                            # Add labels and grid for each subplot
                            ax.set_title(f'Absolute Power Spectrum - {ch_name}')
                            ax.set_xlabel('Frequency (Hz)')
                            ax.set_ylabel('Power (uV^2/Hz)')
                            ax.legend(loc='upper right', fontsize='small')
                            ax.grid(True)

                        # Hide any unused subplots
                        for j in range(num_channels, nrows * ncols):
                            axs[j // ncols, j % ncols].set_visible(False)

                        # Add a main title for the entire figure
                        plt.suptitle('Absolute Power Spectra for All Channels', fontsize=20)

                        # Return the figure object
                        return fig

                    # Call the function and store the figure globally
                    global_abs_power_spectra_lines = plot_absolute_power_lines()

                    # Function to plot absolute power spectra topomaps and return the figure
                    def plot_absolute_power_spectra():
                        # Create a figure with 2 rows and 3 columns
                        fig, axs = plt.subplots(2, 3, figsize=(16, 12))  # Adjusted figure size

                        # Set up normalization and colormap
                        vmin, vmax = 0, 15  # Adjust these values as needed
                        cmap = 'jet'

                        # Plot topomaps for each frequency band
                        mne.viz.plot_topomap(delta_power, cleaned_raw.info, axes=axs[0, 0], show=False, cmap=cmap)
                        axs[0, 0].set_title('Delta Power')
                        
                        mne.viz.plot_topomap(theta_power, cleaned_raw.info, axes=axs[0, 1], show=False, cmap=cmap)
                        axs[0, 1].set_title('Theta Power')
                        
                        mne.viz.plot_topomap(alpha_power, cleaned_raw.info, axes=axs[0, 2], show=False, cmap=cmap)
                        axs[0, 2].set_title('Alpha Power')
                        
                        mne.viz.plot_topomap(beta1_power, cleaned_raw.info, axes=axs[1, 0], show=False, cmap=cmap)
                        axs[1, 0].set_title('Beta1 Power')
                        
                        mne.viz.plot_topomap(beta2_power, cleaned_raw.info, axes=axs[1, 1], show=False, cmap=cmap)
                        axs[1, 1].set_title('Beta2 Power')
                        
                        mne.viz.plot_topomap(gamma_power, cleaned_raw.info, axes=axs[1, 2], show=False, cmap=cmap)
                        axs[1, 2].set_title('Gamma Power')

                        # Add the main title
                        plt.suptitle('Absolute Power Spectra', fontsize=16)

                        # Adjust layout to prevent overlap
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.90, bottom=0.1, wspace=0.3, hspace=0.4, right=0.85)  # Adjust for colorbar

                        # Create and position the colorbar
                        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Adjusted position for colorbar
                        norm = plt.Normalize(vmin=vmin, vmax=vmax)
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])  # Required for ScalarMappable
                        cbar = fig.colorbar(sm, cax=cbar_ax)
                        cbar.set_label('Power (µV²)')

                        # Return the figure object
                        return fig

                    # Call the function and store the figure globally
                    global_abs_power_spectra_topo = plot_absolute_power_spectra()


                    # Function to plot relative power spectra lines and return the figure
                    def plot_relative_power_spectra_lines():
                        # Calculate the total power for normalization
                        total_power = np.sum(psds, axis=1)

                        # Number of channels and grid dimensions
                        ncols = 3
                        nrows = (num_channels + ncols - 1) // ncols  # Calculate rows based on number of channels

                        # Create a figure with subplots arranged in a grid
                        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4 * nrows), constrained_layout=True)

                        # Plot relative power spectrum for each channel
                        for i, ch_name in enumerate(channel_names):
                            # Calculate relative power spectra as percentage of total power
                            relative_psd = (psds[i] / total_power[i]) * 100

                            ax = axs[i // ncols, i % ncols]  # Get the subplot axis

                            # Plot the relative power spectrum for the current channel
                            ax.plot(freqs, relative_psd, color='black', linewidth=1.5, label=f'{ch_name} Relative Power Spectrum (%P)')

                            # Fill each frequency band with colors
                            ax.fill_between(freqs, relative_psd, where=(freqs >= 0.5) & (freqs <= 4), color='brown', alpha=1.0, label='Delta Band')
                            ax.fill_between(freqs, relative_psd, where=(freqs > 4) & (freqs <= 8), color='orange', alpha=1.0, label='Theta Band')
                            ax.fill_between(freqs, relative_psd, where=(freqs > 8) & (freqs <= 13), color='blue', alpha=1.0, label='Alpha Band')
                            ax.fill_between(freqs, relative_psd, where=(freqs > 13) & (freqs <= 20), color='green', alpha=1.0, label='Beta1 Band')
                            ax.fill_between(freqs, relative_psd, where=(freqs > 20) & (freqs <= 30), color='yellow', alpha=1.0, label='Beta2 Band')
                            ax.fill_between(freqs, relative_psd, where=(freqs > 30) & (freqs <= 40), color='cyan', alpha=1.0, label='Gamma Band')

                            # Add labels and grid for each subplot
                            ax.set_title(f'Relative Power Spectrum (%P) - {ch_name}')
                            ax.set_xlabel('Frequency (Hz)')
                            ax.set_ylabel('Relative Power (%P)')
                            ax.legend(loc='upper right', fontsize='small')
                            ax.grid(True)

                        # Hide any unused subplots
                        for j in range(num_channels, nrows * ncols):
                            axs[j // ncols, j % ncols].set_visible(False)

                        # Add a main title for the entire figure
                        plt.suptitle('Relative Power Spectra for All Channels', fontsize=20)

                        # Return the figure object
                        return fig

                    # Call the function and store the figure globally
                    global_rel_power_spectra_lines = plot_relative_power_spectra_lines()

                    # Function to plot relative power spectra topomaps and return the figure
                    def plot_relative_power_spectra_topo():
                        # Create a figure with 2 rows and 3 columns layout
                        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

                        # Set up normalization and colormap
                        vmin, vmax = 0, 15  # Adjust these based on your dataset
                        cmap = 'jet'

                        # Plot topomaps for each frequency band
                        mne.viz.plot_topomap(relative_delta_power, cleaned_raw.info, axes=axs[0, 0], show=False, cmap=cmap)
                        axs[0, 0].set_title('Delta Power')
                        
                        mne.viz.plot_topomap(relative_theta_power, cleaned_raw.info, axes=axs[0, 1], show=False, cmap=cmap)
                        axs[0, 1].set_title('Theta Power')
                        
                        mne.viz.plot_topomap(relative_alpha_power, cleaned_raw.info, axes=axs[0, 2], show=False, cmap=cmap)
                        axs[0, 2].set_title('Alpha Power')
                        
                        mne.viz.plot_topomap(relative_beta1_power, cleaned_raw.info, axes=axs[1, 0], show=False, cmap=cmap)
                        axs[1, 0].set_title('Beta1 Power')
                        
                        mne.viz.plot_topomap(relative_beta2_power, cleaned_raw.info, axes=axs[1, 1], show=False, cmap=cmap)
                        axs[1, 1].set_title('Beta2 Power')
                        
                        mne.viz.plot_topomap(relative_gamma_power, cleaned_raw.info, axes=axs[1, 2], show=False, cmap=cmap)
                        axs[1, 2].set_title('Gamma Power')

                        # Add the main title
                        plt.suptitle('Relative Power Spectra (%P)', fontsize=16)

                        # Adjust layout to avoid overlap with the color bar
                        plt.tight_layout()
                        plt.subplots_adjust(top=0.90, bottom=0.1, right=0.85, wspace=0.3, hspace=0.4)  # Adjust for colorbar

                        # Create and position the colorbar
                        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # Position for colorbar
                        norm = plt.Normalize(vmin=vmin, vmax=vmax)
                        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                        sm.set_array([])  # Required for ScalarMappable
                        cbar = fig.colorbar(sm, cax=cbar_ax)
                        cbar.set_label('Power (%P)')

                        # Return the figure object
                        return fig

                    # Call the function and store the figure globally
                    global_rel_power_spectra_topo = plot_relative_power_spectra_topo()

                    # Function to plot the Theta/Beta1 ratio topomap and return the figure
                    def plot_theta_beta_ratio():
                        # Calculate Theta/Beta1 ratio for each channel
                        theta_beta_ratio = theta_power / beta1_power

                        # Create a figure and axis for the topomap
                        fig, ax = plt.subplots(figsize=(6, 6))

                        # Plot the topomap with the 'jet' colormap
                        im, _ = mne.viz.plot_topomap(theta_beta_ratio, cleaned_raw.info, axes=ax, show=False, cmap='jet')

                        # Add a color bar to the side
                        cbar = plt.colorbar(im, ax=ax, orientation='vertical')
                        cbar.set_label('Theta/Beta1 Ratio')

                        # Add a title
                        ax.set_title('Theta/Beta1 Ratio (Attention Index)', fontsize=14)

                        # Return the figure object
                        return fig

                    # Call the function and store the figure globally
                    global_theta_beta_ratio = plot_theta_beta_ratio()

                    # Function to plot asymmetry topomaps for each frequency band and return the figure
                    def plot_asymmetry():
                        frequency_bands = {
                            'Delta': delta_power,
                            'Theta': theta_power,
                            'Alpha': alpha_power,
                            'Beta1': beta1_power,
                            'Beta2': beta2_power,
                            'Gamma': gamma_power
                        }

                        # Define zones: pairs of left and right hemisphere channels
                        zones = {
                            'Frontal': (['F3', 'F7', 'Fp1'], ['F4', 'F8', 'Fp2']),
                            'Central': (['C3'], ['C4']),
                            'Temporal': (['T3', 'T5'], ['T4', 'T6']),
                            'Parietal': (['P3'], ['P4']),
                            'Occipital': (['O1'], ['O2'])
                        }

                        # Channel names in the standard 10-20 system
                        channel_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']

                        # Initialize a figure for the topomaps (2x3 grid for the 6 frequency bands)
                        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

                        # Iterate over each frequency band and calculate the asymmetry
                        for i, (band_name, power_data) in enumerate(frequency_bands.items()):
                            # Map power data to channels
                            channel_power_map = dict(zip(channel_names, power_data))

                            # Calculate asymmetry for each zone (left-right difference)
                            asymmetry_by_zone = {}
                            for zone_name, (left_channels, right_channels) in zones.items():
                                left_power = np.mean([channel_power_map[ch] for ch in left_channels])
                                right_power = np.mean([channel_power_map[ch] for ch in right_channels])
                                asymmetry_by_zone[zone_name] = left_power - right_power  # Asymmetry = left - right

                            # Create an array with the asymmetry values for topomap plotting
                            asymmetry_values = np.zeros(19)

                            # Assign asymmetry values to left and right hemisphere channels
                            for zone_name, (left_channels, right_channels) in zones.items():
                                left_asymmetry = asymmetry_by_zone[zone_name]
                                for ch in left_channels:
                                    asymmetry_values[channel_names.index(ch)] = left_asymmetry
                                for ch in right_channels:
                                    asymmetry_values[channel_names.index(ch)] = -left_asymmetry  # Right side gets negative value

                            # Plot the asymmetry topomap for the current frequency band
                            ax = axs[i // 3, i % 3]  # Select the subplot for the current frequency band
                            im, _ = mne.viz.plot_topomap(asymmetry_values, cleaned_raw.info, axes=ax, show=False, cmap='jet', contours=4)

                            # Set a title for each plot (the name of the frequency band)
                            ax.set_title(f'{band_name} Band', fontsize=14)

                        # Add a color bar for the entire figure
                        fig.subplots_adjust(right=0.9)  # Make space on the right for the color bar
                        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Position for the color bar
                        fig.colorbar(im, cax=cbar_ax)

                        # Add an overall title
                        plt.suptitle('Asymmetry', fontsize=16)

                        # Return the figure object
                        return fig

                    # Call the function and store the figure globally
                    global_asymmetry_fig = plot_asymmetry()

                    # Function to plot deviations for each Brodmann area and return the figures in a dictionary
                    def plot_brodmann_dorsolateral():
                        # Normative values (assuming you have calculated alpha_power, beta1_power, etc.)
                        normative_values = {
                            'Alpha': np.mean(alpha_power),
                            'Beta1': np.mean(beta1_power),
                            'Beta2': np.mean(beta2_power),
                            'Gamma': np.mean(gamma_power),
                            'Theta': np.mean(theta_power),
                            'Delta': np.mean(delta_power)
                        }

                        # Map EEG channels to Brodmann areas
                        brodmann_mapping = {
                            'Fp1': 'Brodmann Area 11', 'Fp2': 'Brodmann Area 11', 'F7': 'Brodmann Area 46',
                            'F3': 'Brodmann Area 46', 'Fz': 'Brodmann Area 9', 'F4': 'Brodmann Area 46',
                            'F8': 'Brodmann Area 46', 'T3': 'Brodmann Area 21', 'C3': 'Brodmann Area 4',
                            'Cz': 'Brodmann Area 4', 'C4': 'Brodmann Area 4', 'T4': 'Brodmann Area 21',
                            'T5': 'Brodmann Area 39', 'P3': 'Brodmann Area 39', 'Pz': 'Brodmann Area 7',
                            'P4': 'Brodmann Area 39', 'T6': 'Brodmann Area 39', 'O1': 'Brodmann Area 17',
                            'O2': 'Brodmann Area 17'
                        }

                        # Initialize dictionary to store average power for each Brodmann area
                        brodmann_area_power = {area: {band: 0 for band in normative_values.keys()} for area in set(brodmann_mapping.values())}

                        # Calculate average power for each Brodmann area
                        for area in brodmann_area_power.keys():
                            area_channels = [ch for ch, mapped_area in brodmann_mapping.items() if mapped_area == area]

                            for band in normative_values.keys():
                                power_values = []
                                for channel in area_channels:
                                    if channel in locals():  # Ensure the power variable exists
                                        power_values.append(locals()[f"{band.lower()}_power"][cleaned_raw.ch_names.index(channel)])

                                brodmann_area_power[area][band] = np.mean(power_values) if power_values else 0

                        # Calculate deviations from normative values
                        deviation_results = {}
                        for area, area_power in brodmann_area_power.items():
                            deviations = {}
                            for band, value in area_power.items():
                                deviation = value - normative_values[band]
                                deviations[band] = deviation
                            deviation_results[area] = deviations

                        # Brodmann area descriptions
                        brodmann_area_descriptions = {
                            'Brodmann Area 11': 'Orbital Gyrus Frontal Lobe',
                            'Brodmann Area 4': 'Precentral Gyrus Frontal Lobe',
                            'Brodmann Area 39': 'Angular Gyrus Parietal Lobe',
                            'Brodmann Area 21': 'Middle Temporal Gyrus Temporal Lobe',
                            'Brodmann Area 46': 'Dorsolateral Prefrontal Cortex',
                            'Brodmann Area 17': 'Primary Visual Cortex',
                            'Brodmann Area 7': 'Superior Parietal Lobule',
                            'Brodmann Area 9': 'Dorsolateral Prefrontal Cortex'
                        }

                        # Dictionary to store the figures
                        brodmann_figures = {}

                        # Plot deviations for each Brodmann area
                        for area, deviations in deviation_results.items():
                            description = brodmann_area_descriptions.get(area, "Description not found")
                            
                            # Create a new figure for each Brodmann area
                            fig, ax = plt.subplots(figsize=(10, 4))
                            ax.bar(deviations.keys(), deviations.values(), color='skyblue')
                            ax.set_title(f"{area}: {description}")
                            ax.set_xlabel("Frequency Band")
                            ax.set_ylabel("Deviation from Norm")
                            
                            # Store the figure in the dictionary
                            brodmann_figures[area] = fig

                        # Return the dictionary of figures
                        return brodmann_figures

                    # Call the function and store the result globally
                    global_brodmann_dorsolateral = plot_brodmann_dorsolateral()

                    # Function to create a combined figure with multiple Brodmann area plots
                    def plot_combined_brodmann_dorsolateral(brodmann_figures):
                        num_plots = len(brodmann_figures)
                        ncols = 3  # Set number of columns (adjustable)
                        nrows = (num_plots + ncols - 1) // ncols  # Calculate number of rows based on number of plots

                        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))  # Adjust the figure size

                        # Flatten the axes array for easy indexing
                        axs = axs.flatten()

                        # Plot each Brodmann area in a subplot
                        for i, (area, brodmann_fig) in enumerate(brodmann_figures.items()):
                            # Extract the bar chart data from each Brodmann area plot
                            deviations = brodmann_fig.axes[0].containers[0].datavalues
                            labels = [label.get_text() for label in brodmann_fig.axes[0].get_xticklabels()]
                            
                            axs[i].bar(labels, deviations, color='skyblue')
                            axs[i].set_title(area)
                            axs[i].set_xlabel("Frequency Band")
                            axs[i].set_ylabel("Deviation from Norm")

                        # Hide any remaining unused subplots
                        for j in range(i + 1, len(axs)):
                            axs[j].axis('off')

                        # Add a main title for the entire combined figure
                        plt.suptitle("Brodmann Area Deviations", fontsize=16)

                        # Return the combined figure
                        return fig

                    # Call the function and store the combined figure globally
                    global_brodmann_dorsolateral_combined = plot_combined_brodmann_dorsolateral(global_brodmann_dorsolateral)

                    #brodmann findings
                    def generate_brodmann_findings_figure():
                        # Generate the findings text as before
                        normative_values = {
                            'Alpha': np.mean(alpha_power),
                            'Beta1': np.mean(beta1_power),
                            'Beta2': np.mean(beta2_power),
                            'Gamma': np.mean(gamma_power),
                            'Theta': np.mean(theta_power),
                            'Delta': np.mean(delta_power)
                        }
                        

                        # Map EEG channels to Brodmann areas
                        brodmann_mapping = {
                            'Fp1': 'Brodmann Area 11',
                            'Fp2': 'Brodmann Area 11',
                            'F7': 'Brodmann Area 46',
                            'F3': 'Brodmann Area 46',
                            'Fz': 'Brodmann Area 9',
                            'F4': 'Brodmann Area 46',
                            'F8': 'Brodmann Area 46',
                            'T3': 'Brodmann Area 21',
                            'C3': 'Brodmann Area 4',
                            'Cz': 'Brodmann Area 4',
                            'C4': 'Brodmann Area 4',
                            'T4': 'Brodmann Area 21',
                            'T5': 'Brodmann Area 39',
                            'P3': 'Brodmann Area 39',
                            'Pz': 'Brodmann Area 7',
                            'P4': 'Brodmann Area 39',
                            'T6': 'Brodmann Area 39',
                            'O1': 'Brodmann Area 17',
                            'O2': 'Brodmann Area 17',
                        }

                        brodmann_area_power = {area: {band: 0 for band in normative_values.keys()} for area in set(brodmann_mapping.values())}

                        # Calculate average power for each Brodmann area
                        for area in brodmann_area_power.keys():
                            area_channels = [ch for ch, mapped_area in brodmann_mapping.items() if mapped_area == area]
                            for band in normative_values.keys():
                                power_values = []
                                for channel in area_channels:
                                    if channel in locals():
                                        power_values.append(locals()[f"{band.lower()}_power"][global_raw.ch_names.index(channel)])
                                brodmann_area_power[area][band] = np.mean(power_values) if power_values else 0

                        deviation_results = {}
                        for area, area_power in brodmann_area_power.items():
                            deviations = {}
                            for band, value in area_power.items():
                                deviation = value - normative_values[band]
                                deviations[band] = deviation
                            deviation_results[area] = deviations

                        brodmann_area_descriptions = {
                            'Brodmann Area 11': 'Orbital Gyrus Frontal Lobe',
                            'Brodmann Area 4': 'Precentral Gyrus Frontal Lobe',
                            'Brodmann Area 39': 'Angular Gyrus Parietal Lobe',
                            'Brodmann Area 21': 'Middle Temporal Gyrus Temporal Lobe',
                            'Brodmann Area 46': 'Dorsolateral Prefrontal Cortex',
                            'Brodmann Area 17': 'Primary Visual Cortex',
                            'Brodmann Area 7': 'Superior Parietal Lobule',
                            'Brodmann Area 9': 'Dorsolateral Prefrontal Cortex'
                        }

                        adjacent_areas = {
                            'Brodmann Area 11': ['Brodmann Area 47', 'Brodmann Area 10'],
                            'Brodmann Area 4': ['Brodmann Area 6', 'Brodmann Area 1', 'Brodmann Area 2', 'Brodmann Area 3'],
                            'Brodmann Area 39': ['Brodmann Area 40', 'Brodmann Area 7', 'Brodmann Area 19'],
                            'Brodmann Area 21': ['Brodmann Area 20', 'Brodmann Area 22'],
                            'Brodmann Area 46': ['Brodmann Area 9', 'Brodmann Area 10'],
                            'Brodmann Area 17': ['Brodmann Area 18', 'Brodmann Area 19'],
                            'Brodmann Area 7': ['Brodmann Area 39', 'Brodmann Area 40', 'Brodmann Area 5'],
                            'Brodmann Area 9': ['Brodmann Area 46', 'Brodmann Area 10']
                        }

                        # Generate findings text
                        brodmann_output_results = "Following deviations were calculated:\n\n"
                        for area, deviations in deviation_results.items():
                            if deviations:
                                adjacent_with_deviation = any(
                                    adjacent in deviation_results and deviation_results[adjacent]
                                    for adjacent in adjacent_areas.get(area, [])
                                )
                                if not adjacent_with_deviation:
                                    description = brodmann_area_descriptions.get(area, "Description not found")
                                    brodmann_output_results += f"{area}: {description}\n"

                        # Plot the findings text on a figure
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.text(0.5, 0.5, brodmann_output_results, ha='center', va='center', fontsize=12, wrap=True)
                        ax.axis('off')  # Hide the axes for a cleaner look
                        fig.suptitle("Brodmann Findings", fontsize=16)
                        return fig
                    
                    global_brodmann_findings = generate_brodmann_findings_figure()



                    global_channel_dict = channel_dict
                        
                    global_raw_ica = cleaned_raw                    
                    
                    global_ica = ica
                    
                    max_time = int(raw.times[-1])

                    # for ch, freqs in decreased_power_channels.items():
                    #     findings.append(f"Channel {ch}: Decreased power at frequencies {freqs}")
                    #flash('File successfully uploaded and processed.', 'success')
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
        openai_res_med = None

        if plot_type == 'raw' and global_raw:
            fig = global_raw.plot(start=start_time, duration=4, n_channels=19, show=False,scalings=70e-6)
        elif plot_type == 'cleaned' and global_raw_ica:
            scalings = {'eeg': 8e-6}  # Scale EEG channels to 20 µV
            fig = global_raw_ica.plot(start=start_time, duration=4, show=False,scalings=70e-6)
        # elif plot_type == "ica_properties":
        #     # fig = global_ica.plot_properties(global_raw_ica, picks=[0] ,show=False)
    
        #     # # Save each figure to an image and send them to the client
        #     plot_urls = []
        #     # for chann in ['T3','T4','F7','F8','Fp1','Fp2','Cz']:
        #     figs = global_ica.plot_properties(global_raw_ica, picks=[0] ,show=False)
        #     for fig in figs:
            
        #         img = BytesIO()
        #         fig.savefig(img, format='png')
        #         img.seek(0)
        #         plot_url = base64.b64encode(img.getvalue()).decode()
        #         #
        #         #
        #         #print (f'this is plot URL: {plot_url}')
        #         plot_urls.append(plot_url)
        
        elif plot_type == 'decrease_brain_power' and global_decreased_combined_power_fig:
            fig = global_decreased_combined_power_fig
        elif plot_type == 'decrease_brain_power_bandwise' and global_decreased_activation_bandwise:
            fig = global_decreased_activation_bandwise
        elif plot_type == 'increase_brain_power' and global_increased_combined_power_fig:
            fig = global_increased_combined_power_fig
        elif plot_type == 'increase_brain_power_bandwise' and global_increased_activation_bandwise:
            fig = global_increased_activation_bandwise
        elif plot_type == 'abs_power_spectra_lines' and global_abs_power_spectra_lines:
            fig = global_abs_power_spectra_lines
        elif plot_type == 'abs_power_spectra_topo' and global_abs_power_spectra_topo:
            fig = global_abs_power_spectra_topo
        elif plot_type == 'rel_power_spectra_lines' and global_rel_power_spectra_lines:
            fig = global_rel_power_spectra_lines
        elif plot_type == 'rel_power_spectra_topo' and global_rel_power_spectra_topo:
            fig = global_rel_power_spectra_topo
        elif plot_type == 'theta_beta_ratio' and global_theta_beta_ratio:
            fig = global_theta_beta_ratio
        elif plot_type == 'asymmetry' and global_asymmetry_fig:
            fig = global_asymmetry_fig
        elif plot_type == 'brodmann_dorsolateral' and global_brodmann_dorsolateral_combined:
            fig = global_brodmann_dorsolateral_combined
        elif plot_type == 'brodmann_findings' and global_brodmann_findings:
            fig = global_brodmann_findings
    
            
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
        emit('update_plot', {'plot_url': plot_url, 'raw_report': openai_res, 'raw_medical_report': openai_res_med})
        
    except Exception as e:
        print(f"Error generating plot: {e}")
        emit('update_plot', {'plot_url': None, 'raw_report': None, 'raw_medical_report': None})  # Send a fallback response in case of error
        

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port= 5000)#, use_reloader=False)
    # socketio.run(app, debug=False)#, use_reloader=False)
