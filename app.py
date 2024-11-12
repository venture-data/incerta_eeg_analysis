import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for Matplotlib

from flask import Flask, render_template, request, flash, redirect
from flask_socketio import SocketIO, emit
import mne
import os
from io import BytesIO
import base64
import numpy as np
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'C:/temp'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # Set max upload size to 50 MB

socketio = SocketIO(app)

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

global_raw = None
global_raw_ica = None
asymmetry_content_fig = None
combined_fig = None

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global global_raw, global_raw_ica, asymmetry_content_fig, combined_fig

    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.filename)[1]
            if file_ext.lower() != '.edf':
                flash('Invalid file format! Please upload a .edf file.', 'error')
                return redirect(request.url)

            try:
                # Save file and load EEG data
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
                uploaded_file.save(filepath)
                raw = mne.io.read_raw_edf(filepath, preload=True)
                raw.filter(0.5, 45, fir_design='firwin')

                # Drop unnecessary channels and set montage
                channels_to_drop = ['Bio1-2', 'Bio3-4', 'ECG', 'Bio4', 'VSyn', 'ASyn', 'LABEL', 'EEG Fpz', 'A1', 'A2', 'EEG Oz']
                mapping = {
                    'EEG Fp1': 'Fp1', 'EEG Fp2': 'Fp2', 'EEG F7': 'F7', 'EEG F3': 'F3', 'EEG Fz': 'Fz',
                    'EEG F4': 'F4', 'EEG F8': 'F8', 'EEG T3': 'T3', 'EEG C3': 'C3', 'EEG Cz': 'Cz',
                    'EEG C4': 'C4', 'EEG T4': 'T4', 'EEG T5': 'T5', 'EEG P3': 'P3', 'EEG Pz': 'Pz',
                    'EEG P4': 'P4', 'EEG T6': 'T6', 'EEG O1': 'O1', 'EEG O2': 'O2'
                }
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.drop_channels(channels_to_drop)
                raw.rename_channels(mapping)
                raw.set_montage(montage, on_missing='ignore')

                # Apply ICA to remove artifacts
                ica = ICA(n_components=19, random_state=97, max_iter='auto')
                ica.fit(raw)
                ica.exclude = [0,1,2,3]  # Set the component to exclude based on visual inspection
                cleaned_raw = ica.apply(raw.copy())
                global_raw_ica = cleaned_raw

                # Asymmetry Analysis
                def detect_asymmetry(data, sfreq, pairs, freq_bands, threshold_factor=1.5):
                    asymmetry_findings = []
                    psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=45, n_fft=2048)
                    for ch1, ch2 in pairs:
                        idx1, idx2 = raw.ch_names.index(ch1), raw.ch_names.index(ch2)
                        for band, (fmin, fmax) in freq_bands.items():
                            band_idx = (freqs >= fmin) & (freqs < fmax)
                            psd_diff = np.abs(psd[idx1, band_idx] - psd[idx2, band_idx])
                            median_band_power = np.median(psd[:, band_idx])
                            if np.any(psd_diff > threshold_factor * median_band_power):
                                asymmetry_findings.append((ch1, ch2, band))

                    asymmetry_content = "Significant Asymmetry over the:\n" + ", ".join(f"{ch1}-{ch2} ({band})" for ch1, ch2, band in asymmetry_findings)
                    fig, ax = plt.subplots()
                    ax.axis('off')
                    ax.text(0.5, 0.5, asymmetry_content, fontsize=12, ha='center', va='center', wrap=True)
                    # ax.set_title('Asymmetry Findings')
                    return fig

                # Define parameters and run asymmetry analysis
                asymmetry_pairs = [("F7", "F8"), ("T5", "T6"), ("O1", "O2"), ("F3", "F4")]
                bands = {"delta": (1.5, 4), "theta": (4, 7.5), "alpha": (7.5, 14), "beta-1": (14, 20), "beta-2": (20, 30), "gamma": (30, 40)}
                data, times = cleaned_raw[:]
                sfreq = cleaned_raw.info['sfreq']
                asymmetry_content_fig = detect_asymmetry(data, sfreq, asymmetry_pairs, bands)
                
                # Theta and Alpha Detection and Plotting
                # def detect_and_plot_theta_alpha_summary(data, sfreq, theta_range=(4, 8), alpha_range=(7.5, 14),
                #                                         beta2_range=(20,30), gamma_range = (30,40), threshold_factor=30.0):
                #     theta_channels = []
                #     alpha_channels = []
                #     beta2_channels = []
                #     gamma_channels = []
                #     psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=4, fmax=40, n_fft=2048)
                #     for ch_idx, channel_psd in enumerate(psd):
                #         theta_power = np.sum(channel_psd[(freqs >= theta_range[0]) & (freqs < theta_range[1])])
                #         alpha_power = np.sum(channel_psd[(freqs >= alpha_range[0]) & (freqs < alpha_range[1])])
                #         beta2_power = np.sum(channel_psd[(freqs >= beta2_range[0]) & (freqs < beta2_range[1])])
                #         gamma_power = np.sum(channel_psd[(freqs >= gamma_range[0]) & (freqs < gamma_range[1])])
                #         if theta_power > threshold_factor * np.median(psd):
                #             theta_channels.append(raw.ch_names[ch_idx])
                #         if beta2_power > threshold_factor * np.median(psd):
                #             beta2_channels.append(raw.ch_names[ch_idx])
                #         if alpha_power > threshold_factor * np.median(psd):
                #             alpha_channels.append(raw.ch_names[ch_idx])
                #         if gamma_power > threshold_factor * np.median(psd):
                #             gamma_channels.append(raw.ch_names[ch_idx])
                    
                #     # Group channels by region
                #     theta_regions = {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []}
                #     alpha_regions = {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []}
                #     beta2_regions = {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []}
                #     gamma_regions = {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []}
                                        
                #     region_mapping = {
                #         "Frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
                #         "Temporal": ["T3", "T4", "T5", "T6"],
                #         "Central": ["C3", "C4", "Cz"],
                #         "Parietal": ["P3", "P4", "Pz"],
                #         "Occipital": ["O1", "O2"]
                #     }

                #     # Collect theta and alpha channels by region
                #     for ch_name in theta_channels:
                #         for region, channels in region_mapping.items():
                #             if ch_name in channels:
                #                 theta_regions[region].append(ch_name)

                #     for ch_name in alpha_channels:
                #         for region, channels in region_mapping.items():
                #             if ch_name in channels:
                #                 alpha_regions[region].append(ch_name)
                #     for ch_name in gamma_channels:
                #         for region, channels in region_mapping.items():
                #             if ch_name in channels:
                #                 gamma_regions[region].append(ch_name)
                #     for ch_name in beta2_channels:
                #         for region, channels in region_mapping.items():
                #             if ch_name in channels:
                #                 beta2_regions[region].append(ch_name)

                #     # Format the summary output to include channel names by region
                #     summary_theta = "% Relative increased theta in " + ", ".join([
                #         f"{region.lower()} {', '.join(channels)}" for region, channels in theta_regions.items() if channels
                #     ])
                #     summary_alpha = "% Relative increase of alpha activity in " + ", ".join([
                #         f"{region.lower()} {', '.join(channels)}" for region, channels in alpha_regions.items() if channels
                #     ])
                #     summary_gamma = "% Relative increase of gamma activity in " + ", ".join([
                #         f"{region.lower()} {', '.join(channels)}" for region, channels in gamma_regions.items() if channels
                #     ])
                #     summary_beta2 = "% Relative increase of beta2 activity in " + ", ".join([
                #         f"{region.lower()} {', '.join(channels)}" for region, channels in beta2_regions.items() if channels
                #     ])

                #     # Plot summary in a single figure
                #     fig, ax = plt.subplots(figsize=(10, 5))
                #     ax.axis('off')  # Hide axes for a cleaner look
                #     text_content = f"Theta Activity Summary:\n{summary_theta}\n\nAlpha Activity Summary:\n{summary_alpha}\n\nGamma Activity Summary:\n{summary_gamma}\n\nBeta2 Activity Summary:\n{summary_beta2}\n\n"
                #     ax.text(0.5, 0.5, text_content, fontsize=12, ha='center', va='center', wrap=True)
                #     # ax.set_title('Theta/Alpha Findings')
                    
                #     return fig

                # def detect_and_plot_theta_alpha_summary(data, sfreq, theta_range=(4, 8), alpha_range=(7.5, 14),
                #                         delta_range=(1, 4), beta1_range=(14, 20), 
                #                         beta2_range=(20, 30), gamma_range=(30, 40), threshold_factor=30.0):
    
                #     # Initialize lists for each frequency band
                #     theta_channels = []
                #     alpha_channels = []
                #     delta_channels = []
                #     beta1_channels = []
                #     beta2_channels = []
                #     gamma_channels = []

                #     # Compute PSD for the data within specified frequency range
                #     psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=40, n_fft=2048)

                #     for ch_idx, channel_psd in enumerate(psd):
                #         # Calculate power for each frequency band
                #         theta_power = np.sum(channel_psd[(freqs >= theta_range[0]) & (freqs < theta_range[1])])
                #         alpha_power = np.sum(channel_psd[(freqs >= alpha_range[0]) & (freqs < alpha_range[1])])
                #         delta_power = np.sum(channel_psd[(freqs >= delta_range[0]) & (freqs < delta_range[1])])
                #         beta1_power = np.sum(channel_psd[(freqs >= beta1_range[0]) & (freqs < beta1_range[1])])
                #         beta2_power = np.sum(channel_psd[(freqs >= beta2_range[0]) & (freqs < beta2_range[1])])
                #         gamma_power = np.sum(channel_psd[(freqs >= gamma_range[0]) & (freqs < gamma_range[1])])

                #         # Compare power to threshold to detect increased activity
                #         if theta_power > threshold_factor * np.median(psd):
                #             theta_channels.append(raw.ch_names[ch_idx])
                #         if alpha_power > threshold_factor * np.median(psd):
                #             alpha_channels.append(raw.ch_names[ch_idx])
                #         if delta_power > threshold_factor * np.median(psd):
                #             delta_channels.append(raw.ch_names[ch_idx])
                #         if beta1_power > threshold_factor * np.median(psd):
                #             beta1_channels.append(raw.ch_names[ch_idx])
                #         if beta2_power > threshold_factor * np.median(psd):
                #             beta2_channels.append(raw.ch_names[ch_idx])
                #         if gamma_power > threshold_factor * np.median(psd):
                #             gamma_channels.append(raw.ch_names[ch_idx])

                #     # Define region mapping for grouping channels
                #     region_mapping = {
                #         "Frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
                #         "Temporal": ["T3", "T4", "T5", "T6"],
                #         "Central": ["C3", "C4", "Cz"],
                #         "Parietal": ["P3", "P4", "Pz"],
                #         "Occipital": ["O1", "O2"]
                #     }

                #     # Initialize dictionaries to hold channels by region for each band
                #     region_summaries = {
                #         'theta': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                #         'alpha': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                #         'delta': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                #         'beta1': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                #         'beta2': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                #         'gamma': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []}
                #     }

                #     # Assign channels to regions for each frequency band
                #     for band, channels in zip(['theta', 'alpha', 'delta', 'beta1', 'beta2', 'gamma'], 
                #                             [theta_channels, alpha_channels, delta_channels, beta1_channels, beta2_channels, gamma_channels]):
                #         for ch_name in channels:
                #             for region, region_channels in region_mapping.items():
                #                 if ch_name in region_channels:
                #                     region_summaries[band][region].append(ch_name)

                #     # Generate summary text for each frequency band by region
                #     summary_texts = {}
                #     for band in region_summaries.keys():
                #         summary_texts[band] = f"% Relative increase of {band} activity in " + ", ".join([
                #             f"{region.lower()} {', '.join(channels)}" for region, channels in region_summaries[band].items() if channels
                #         ])

                #     # Plot summary in a single figure
                #     fig, ax = plt.subplots(figsize=(10, 5))
                #     ax.axis('off')  # Hide axes for a cleaner look
                #     text_content = "\n\n".join([
                #         f"{band.capitalize()} Activity Summary:\n{summary}" for band, summary in summary_texts.items()
                #     ])
                #     ax.text(0.5, 0.5, text_content, fontsize=12, ha='center', va='center', wrap=True)
                #     fig.suptitle("Frequency Band Findings", fontsize=20)
                    
                #     return fig

                def detect_and_plot_theta_alpha_summary(data, sfreq, theta_range=(4, 8), alpha_range=(7.5, 14),
                                        delta_range=(1, 4), beta1_range=(14, 20), 
                                        beta2_range=(20, 30), gamma_range=(30, 40), 
                                        threshold_factor=30.0, low_threshold_factor=29.5):
    
                    # Initialize lists for increased and decreased activity for each frequency band
                    theta_increased_channels = []
                    alpha_increased_channels = []
                    delta_increased_channels = []
                    beta1_increased_channels = []
                    beta2_increased_channels = []
                    gamma_increased_channels = []
                    
                    theta_decreased_channels = []
                    alpha_decreased_channels = []
                    delta_decreased_channels = []
                    beta1_decreased_channels = []
                    beta2_decreased_channels = []
                    gamma_decreased_channels = []

                    # Compute PSD for the data within specified frequency range
                    psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=40, n_fft=2048)
                    median_psd = np.median(psd)

                    for ch_idx, channel_psd in enumerate(psd):
                        # Calculate power for each frequency band
                        theta_power = np.sum(channel_psd[(freqs >= theta_range[0]) & (freqs < theta_range[1])])
                        alpha_power = np.sum(channel_psd[(freqs >= alpha_range[0]) & (freqs < alpha_range[1])])
                        delta_power = np.sum(channel_psd[(freqs >= delta_range[0]) & (freqs < delta_range[1])])
                        beta1_power = np.sum(channel_psd[(freqs >= beta1_range[0]) & (freqs < beta1_range[1])])
                        beta2_power = np.sum(channel_psd[(freqs >= beta2_range[0]) & (freqs < beta2_range[1])])
                        gamma_power = np.sum(channel_psd[(freqs >= gamma_range[0]) & (freqs < gamma_range[1])])

                        # Compare power to thresholds for increased and decreased activity
                        if theta_power > threshold_factor * median_psd:
                            theta_increased_channels.append(raw.ch_names[ch_idx])
                        elif theta_power < low_threshold_factor * median_psd:
                            theta_decreased_channels.append(raw.ch_names[ch_idx])

                        if alpha_power > threshold_factor * median_psd:
                            alpha_increased_channels.append(raw.ch_names[ch_idx])
                        elif alpha_power < low_threshold_factor * median_psd:
                            alpha_decreased_channels.append(raw.ch_names[ch_idx])

                        if delta_power > threshold_factor * median_psd:
                            delta_increased_channels.append(raw.ch_names[ch_idx])
                        elif delta_power < low_threshold_factor * median_psd:
                            delta_decreased_channels.append(raw.ch_names[ch_idx])

                        if beta1_power > threshold_factor * median_psd:
                            beta1_increased_channels.append(raw.ch_names[ch_idx])
                        elif beta1_power < low_threshold_factor * median_psd:
                            beta1_decreased_channels.append(raw.ch_names[ch_idx])

                        if beta2_power > threshold_factor * median_psd:
                            beta2_increased_channels.append(raw.ch_names[ch_idx])
                        elif beta2_power < low_threshold_factor * median_psd:
                            beta2_decreased_channels.append(raw.ch_names[ch_idx])

                        if gamma_power > threshold_factor * median_psd:
                            gamma_increased_channels.append(raw.ch_names[ch_idx])
                        elif gamma_power < low_threshold_factor * median_psd:
                            gamma_decreased_channels.append(raw.ch_names[ch_idx])

                    # Define region mapping for grouping channels
                    region_mapping = {
                        "Frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
                        "Temporal": ["T3", "T4", "T5", "T6"],
                        "Central": ["C3", "C4", "Cz"],
                        "Parietal": ["P3", "P4", "Pz"],
                        "Occipital": ["O1", "O2"]
                    }

                    # Initialize dictionaries to hold channels by region for each band and activity type
                    region_summaries = {
                        'theta_increased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'theta_decreased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'alpha_increased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'alpha_decreased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'delta_increased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'delta_decreased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'beta1_increased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'beta1_decreased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'beta2_increased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'beta2_decreased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'gamma_increased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                        'gamma_decreased': {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []},
                    }

                    # Assign channels to regions for increased and decreased activity
                    for band, channels in zip(
                        ['theta_increased', 'theta_decreased', 'alpha_increased', 'alpha_decreased', 
                        'delta_increased', 'delta_decreased', 'beta1_increased', 'beta1_decreased', 
                        'beta2_increased', 'beta2_decreased', 'gamma_increased', 'gamma_decreased'], 
                        [theta_increased_channels, theta_decreased_channels, alpha_increased_channels, alpha_decreased_channels,
                        delta_increased_channels, delta_decreased_channels, beta1_increased_channels, beta1_decreased_channels,
                        beta2_increased_channels, beta2_decreased_channels, gamma_increased_channels, gamma_decreased_channels]):
                        
                        for ch_name in channels:
                            for region, region_channels in region_mapping.items():
                                if ch_name in region_channels:
                                    region_summaries[band][region].append(ch_name)

                    # Generate summary text for each frequency band by region and activity type
                    summary_texts = {}
                    for band in region_summaries.keys():
                        activity = "increase" if "increased" in band else "decrease"
                        freq_band = band.split('_')[0].capitalize()
                        summary_texts[band] = f"% Relative {activity} of {freq_band} activity in " + ", ".join([
                            f"{region.lower()} {', '.join(channels)}" for region, channels in region_summaries[band].items() if channels
                        ])

                    # Plot summary in a single figure
                    fig, ax = plt.subplots(figsize=(10, 10))
                    ax.axis('off')  # Hide axes for a cleaner look
                    text_content = "\n\n".join([
                        f"{summary}" for summary in summary_texts.values() if summary.split()[-1] != "in"
                    ])
                    ax.text(0.5, 0.5, text_content, fontsize=14, ha='center', va='center', wrap=True)
                    fig.suptitle("Frequency Band Findings - Increased and Decreased Activity", fontsize=16)
                    
                    return fig

                # Generate theta and alpha summary figure
                theta_alpha_content_fig = detect_and_plot_theta_alpha_summary(data, sfreq)

                # (Previous code remains the same until the combined figure section)
                
                # Function for Alpha Peak Analysis
                def alpha_peak_analysis(cleaned_raw):
                    spectrum = cleaned_raw.compute_psd(method='welch', fmin=1.5, fmax=40., n_fft=2048)
                    psds, freqs = spectrum.get_data(return_freqs=True)
                    occipital_channels = ['O1', 'O2']
                    alpha_band = (7.5, 14)
                    occipital_psds = psds[[cleaned_raw.ch_names.index(ch) for ch in occipital_channels], :]
                    alpha_mask = np.where((freqs >= alpha_band[0]) & (freqs <= alpha_band[1]))[0]
                    alpha_freqs = freqs[alpha_mask]
                    occipital_psds_alpha = occipital_psds[:, alpha_mask]
                    alpha_peaks = {ch: alpha_freqs[np.argmax(occipital_psds_alpha[i])] for i, ch in enumerate(occipital_channels)}
                    return f"O1 = {alpha_peaks['O1']:.2f} Hz / uV^2, O2 = {alpha_peaks['O2']:.2f} Hz / uV^2"

                # Function for Deviations Analysis
                def detect_deviations(data, sfreq, bands, threshold_factor=50.5):
                    deviation_results = []
                    psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=45, n_fft=2048)
                    for ch_idx, ch_name in enumerate(raw.ch_names):
                        for band, (fmin, fmax) in bands.items():
                            band_idx = (freqs >= fmin) & (freqs < fmax)
                            band_power = np.sum(psd[ch_idx, band_idx])
                            median_band_power = np.median(psd[:, band_idx])
                            if band_power > threshold_factor * median_band_power:
                                deviation_results.append((ch_name, band))
                    return deviation_results

                # Function for Leaky Gut Markers Analysis
                def detect_leaky_gut_markers(data, sfreq, gut_freq_range=(0.05, 0.1), threshold_factor=1.5):
                    gut_markers = []
                    psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=0.01, fmax=0.5, n_fft=2048)
                    for ch_idx, channel_psd in enumerate(psd):
                        gut_power = np.sum(channel_psd[(freqs >= gut_freq_range[0]) & (freqs < gut_freq_range[1])])
                        if gut_power > threshold_factor * np.median(psd):
                            gut_markers.append(raw.ch_names[ch_idx])
                    return gut_markers

                # Function to generate combined figure
                def plot_combined_analysis():
                    # Retrieve analysis data
                    alpha_results = alpha_peak_analysis(global_raw_ica)
                    deviation_results = detect_deviations(data, sfreq, bands)
                    leaky_gut_markers = detect_leaky_gut_markers(data, sfreq)

                    # Brodmann mapping for deviations analysis
                    brodmann_mapping = {
                        "Fp1": "Brodmann Area 10 (Anterior Prefrontal Cortex)",
                        "Fp2": "Brodmann Area 10 (Anterior Prefrontal Cortex)",
                        "F3": "Brodmann Area 8 (Frontal Eye Field)",
                        "F4": "Brodmann Area 8 (Frontal Eye Field)",
                        "Fz": "Brodmann Area 6 (Premotor Cortex)",
                        "F7": "Brodmann Area 9 (Dorsolateral Prefrontal Cortex)",
                        "F8": "Brodmann Area 9 (Dorsolateral Prefrontal Cortex)",
                        "T3": "Brodmann Area 21 (Middle Temporal Gyrus)",
                        "T4": "Brodmann Area 21 (Middle Temporal Gyrus)",
                        "T5": "Brodmann Area 37 (Occipitotemporal Area)",
                        "T6": "Brodmann Area 37 (Occipitotemporal Area)",
                        "C3": "Brodmann Area 4 (Primary Motor Cortex)",
                        "C4": "Brodmann Area 4 (Primary Motor Cortex)",
                        "Cz": "Brodmann Area 6 (Premotor Cortex)",
                        "P3": "Brodmann Area 7 (Superior Parietal Lobule)",
                        "P4": "Brodmann Area 7 (Superior Parietal Lobule)",
                        "Pz": "Brodmann Area 39 (Angular Gyrus)",
                        "O1": "Brodmann Area 17 (Primary Visual Cortex)",
                        "O2": "Brodmann Area 17 (Primary Visual Cortex)",
                        "Fpz": "Brodmann Area 11 (Orbital Prefrontal Cortex)",
                        "T7": "Brodmann Area 22 (Superior Temporal Gyrus)",
                        "T8": "Brodmann Area 22 (Superior Temporal Gyrus)"
                    }

                    # Format deviations summary for the figure
                    deviation_summary = {}
                    for ch_name, band in deviation_results:
                        brodmann_area = brodmann_mapping.get(ch_name, "Unknown Area")
                        if brodmann_area not in deviation_summary:
                            deviation_summary[brodmann_area] = []
                        deviation_summary[brodmann_area].append(f"{ch_name} ({band} band)")

                    sorted_deviations = sorted(deviation_summary.items(), key=lambda x: len(x[1]), reverse=True)[:4]
                    deviations_text = "\n".join([f"{area}: {', '.join(findings)}" for area, findings in sorted_deviations])

                    leaky_gut_text = f"Markers found in channels: {', '.join(leaky_gut_markers)}" if leaky_gut_markers else "No significant markers found for leaky gut syndrome."



                    # # Display Alpha Peaks
                    # fig_alpha, ax_alpha = plt.subplots(figsize=(10, 5))
                    # ax_alpha.axis('off')  # Hide axes for a cleaner look
                    # text_content_alpha = f"Alpha Peak Frequencies:\n{alpha_results}"
                    # ax_alpha.text(0.5, 0.5, text_content_alpha, fontsize=12, ha='center', va='center', wrap=True)
                    # # ax_alpha.set_title('Theta/Alpha Findings')

                    # # Display Deviations Analysis
                    # fig_dev, ax_dev = plt.subplots(figsize=(10, 5))
                    # ax_dev.axis('off')  # Hide axes for a cleaner look
                    # text_content_dev = f"Deviations Analysis:\n{deviations_text}"
                    # ax_dev.text(0.5, 0.5, text_content_dev, fontsize=12, ha='center', va='center', wrap=True)
                    # # ax_dev.set_title('Theta/Alpha Findings')

                    # # Display Leaky Gut Analysis
                    # fig_gut, ax_gut = plt.subplots(figsize=(10, 5))
                    # ax_gut.axis('off')  # Hide axes for a cleaner look
                    # text_content_gut = f"Deviations Analysis:\n{deviations_text}"
                    # ax_gut.text(0.5, 0.5, text_content_gut, fontsize=12, ha='center', va='center', wrap=True)
                    # ax.set_title('Theta/Alpha Findings')

                    # Create combined figure
                    # fig, axs = plt.subplots(3, 1, figsize=(12, 18))
                    fig, axs = plt.subplots(2, 1, figsize=(12, 18))

                    # Display Alpha Peaks
                    axs[0].axis('off')
                    axs[0].text(0.5, 0.5, f"Alpha Peak Frequencies:\n{alpha_results}", ha='center', va='center', wrap=True, fontsize=34)
                    # axs[0].set_title("Alpha Peak Frequencies")

                    # # Display Deviations Analysis
                    # axs[1].axis('off')
                    # axs[1].text(0.5, 0.5, f"Deviations Analysis:\n{deviations_text}", ha='center', va='center', wrap=True, fontsize=34)
                    # # axs[1].set_title("Deviations Analysis")

                    # Display Leaky Gut Analysis
                    # axs[2].axis('off')
                    # axs[2].text(0.5, 0.5, leaky_gut_text, ha='center', va='center', wrap=True, fontsize=34)
                    axs[1].axis('off')
                    axs[1].text(0.5, 0.5, leaky_gut_text, ha='center', va='center', wrap=True, fontsize=34)
                    # axs[2].set_title("Leaky Gut Syndrome Analysis")

                    plt.tight_layout()
                    # fig.suptitle("Combined Analysis: Alpha Peaks, Deviations, and Leaky Gut Markers", fontsize=36)
                    return fig#fig_alpha, fig_dev, fig_gut

                combined_fig = plot_combined_analysis()

                # psd_object = cleaned_raw.compute_psd(fmin=0.5, fmax=45, n_fft=1024)  # This returns a PSD object
                psd_object = cleaned_raw.compute_psd(fmin=1.5, fmax=40, n_fft=2048)  # This returns a PSD object
                psds = psd_object.get_data()  # Extract the power spectral density
                freqs = psd_object.freqs  # Get the frequency values

                """# Set Band Frequency"""

                #Define frequency bands
                delta_band = (1.5, 4)      # Delta: 1.5-4 Hz
                theta_band = (4, 7.5)      # Theta: 4-7.5 Hz
                alpha_band = (7.5, 14)     # Alpha: 7.5- Hz
                beta1_band = (14, 20)      # Beta1: 13-20 Hz
                beta2_band = (20, 30)      # Beta2: 20-30 Hz
                gamma_band = (30, 40)      # Gamma: 30-40 Hz

                # delta_band = (1, 4)      # Delta: 1.5-4 Hz
                # theta_band = (4, 8)      # Theta: 4-7.5 Hz
                # alpha_band = (8, 13)     # Alpha: 7.5- Hz
                # beta1_band = (13, 30)      # Beta1: 13-20 Hz
                # # beta2_band = (20, 30)      # Beta2: 20-30 Hz
                # # gamma_band = (30, 40)      # Gamma: 30-40 Hz

                # Calculate power for each band
                delta_power = np.mean(psds[:, (freqs >= delta_band[0]) & (freqs <= delta_band[1])], axis=1)
                theta_power = np.mean(psds[:, (freqs >= theta_band[0]) & (freqs <= theta_band[1])], axis=1)
                alpha_power = np.mean(psds[:, (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])], axis=1)
                beta1_power = np.mean(psds[:, (freqs >= beta1_band[0]) & (freqs <= beta1_band[1])], axis=1)
                beta2_power = np.mean(psds[:, (freqs >= beta2_band[0]) & (freqs <= beta2_band[1])], axis=1)
                gamma_power = np.mean(psds[:, (freqs >= gamma_band[0]) & (freqs <= gamma_band[1])], axis=1)

                #brodmann findings
                def generate_brodmann_findings_figure():
                    # Generate the findings text as before
                    # normative_values = {
                    #     'Alpha': np.mean(alpha_power),
                    #     'Beta1': np.mean(beta1_power),
                    #     'Beta2': np.mean(beta2_power),
                    #     'Gamma': np.mean(gamma_power),
                    #     'Theta': np.mean(theta_power),
                    #     'Delta': np.mean(delta_power)
                    # }

                    normative_values = {
                        'Alpha': alpha_power,
                        'Beta1': beta1_power,
                        'Beta2': beta2_power,
                        'Gamma': gamma_power,
                        'Theta': theta_power,
                        'Delta': delta_power
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
                        'Brodmann Area 9': ['Brodmann Area 46', 'Brodmann Area 10'],
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
                    ax.text(0.5, 0.5, brodmann_output_results, ha='center', va='center', fontsize=26, wrap=True)
                    ax.axis('off')  # Hide the axes for a cleaner look
                    fig.suptitle("Brodmann Findings", fontsize=26)
                    return fig
                    
                global_brodmann_findings = generate_brodmann_findings_figure()

                def plot_combined_analysis():
                    fig, axs = plt.subplots(4, 1, figsize=(12, 18))

                    def render_fig_to_array(sub_fig):
                        canvas = FigureCanvas(sub_fig)
                        canvas.draw()
                        width, height = canvas.get_width_height()
                        image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
                        return image

                    if asymmetry_content_fig:
                        asymmetry_img = render_fig_to_array(asymmetry_content_fig)
                        axs[0].imshow(asymmetry_img)
                        # axs[0].set_title("Asymmetry Analysis")
                        axs[0].axis('off')

                    if theta_alpha_content_fig:
                        theta_alpha_img = render_fig_to_array(theta_alpha_content_fig)
                        axs[1].imshow(theta_alpha_img)
                        axs[1].set_title("Power Spectra Findings")
                        axs[1].axis('off')
                    
                    if combined_fig:
                        combined_fig_img = render_fig_to_array(combined_fig)
                        axs[2].imshow(combined_fig_img)
                        axs[2].set_title("Pathological Sign Detection")
                        axs[2].axis('off')
                    if global_brodmann_findings:
                        global_brodmann_findings_img = render_fig_to_array(global_brodmann_findings)
                        axs[3].imshow(global_brodmann_findings_img)
                        axs[3].set_title("Brodmann Findings test")
                        axs[3].axis('off')
                    # if fig_dev:
                    #     fig_dev_img = render_fig_to_array(fig_dev)
                    #     axs[4].imshow(fig_dev_img)
                    #     axs[4].set_title("Deviations Analysis")
                    #     axs[4].axis('off')
                    # if fig_gut:
                    #     fig_gut_img = render_fig_to_array(fig_gut)
                    #     axs[5].imshow(fig_gut_img)
                    #     axs[5].set_title("Leaky Gut Syndrome Analysis")
                    #     axs[5].axis('off')

                    plt.tight_layout()
                    fig.suptitle("Combined Analysis: Asymmetry and Theta-Alpha Findings", fontsize=16)
                    return fig

                combined_fig = plot_combined_analysis()

                return render_template('upload_with_topomap_dropdown.html', max_time=int(raw.times[-1]))

            except Exception as e:
                print(f"Error processing file: {e}")
                return "Error processing file", 500

    return render_template('upload_with_topomap_dropdown.html', max_time=0)

@socketio.on('slider_update')
def handle_slider_update(data):
    global asymmetry_content_fig

    try:
        plot_type = data['plot_type']
        plot_url = None
        
        fig = None
        if plot_type == 'combined_analysis' and combined_fig:
            fig = combined_fig

        if fig:
            img = BytesIO()
            fig.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close(fig)

        emit('update_plot', {'plot_url': plot_url})
    except Exception as e:
        print(f"Error generating plot: {e}")
        emit('update_plot', {'plot_url': None})

if __name__ == '__main__':
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)
