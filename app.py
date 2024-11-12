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

                    asymmetry_content = "Asymmetry detected in channels:\n" + ", ".join(f"{ch1}-{ch2} ({band})" for ch1, ch2, band in asymmetry_findings)
                    fig, ax = plt.subplots()
                    ax.axis('off')
                    ax.text(0.5, 0.5, asymmetry_content, fontsize=12, ha='center', va='center', wrap=True)
                    # ax.set_title('Asymmetry Findings')
                    return fig

                # Define parameters and run asymmetry analysis
                asymmetry_pairs = [("F7", "F8"), ("T5", "T6"), ("O1", "O2"), ("F3", "F4")]
                bands = {"delta": (1.5, 4), "theta": (4, 7.5), "alpha": (7.5, 14), "beta-1": (14, 20), "beta-2": (20, 30), "gamma": (30, 40)}
                data, times = raw[:]
                sfreq = raw.info['sfreq']
                asymmetry_content_fig = detect_asymmetry(data, sfreq, asymmetry_pairs, bands)
                
                # Theta and Alpha Detection and Plotting
                def detect_and_plot_theta_alpha_summary(data, sfreq, theta_range=(4, 8), alpha_range=(7.5, 14), threshold_factor=1.5):
                    theta_channels = []
                    alpha_channels = []
                    psd, freqs = mne.time_frequency.psd_array_welch(data, sfreq=sfreq, fmin=1, fmax=45, n_fft=2048)
                    for ch_idx, channel_psd in enumerate(psd):
                        theta_power = np.sum(channel_psd[(freqs >= theta_range[0]) & (freqs < theta_range[1])])
                        alpha_power = np.sum(channel_psd[(freqs >= alpha_range[0]) & (freqs < alpha_range[1])])
                        if theta_power > threshold_factor * np.median(psd):
                            theta_channels.append(raw.ch_names[ch_idx])
                        if alpha_power > threshold_factor * np.median(psd):
                            alpha_channels.append(raw.ch_names[ch_idx])
                    
                    # Group channels by region
                    theta_regions = {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []}
                    alpha_regions = {"Frontal": [], "Temporal": [], "Central": [], "Parietal": [], "Occipital": []}
                                        
                    region_mapping = {
                        "Frontal": ["Fp1", "Fp2", "F3", "F4", "Fz"],
                        "Temporal": ["T3", "T4", "T5", "T6"],
                        "Central": ["C3", "C4", "Cz"],
                        "Parietal": ["P3", "P4", "Pz"],
                        "Occipital": ["O1", "O2"]
                    }

                    # Collect theta and alpha channels by region
                    for ch_name in theta_channels:
                        for region, channels in region_mapping.items():
                            if ch_name in channels:
                                theta_regions[region].append(ch_name)

                    for ch_name in alpha_channels:
                        for region, channels in region_mapping.items():
                            if ch_name in channels:
                                alpha_regions[region].append(ch_name)

                    # Format the summary output to include channel names by region
                    summary_theta = "% Relative increased theta in " + ", ".join([
                        f"{region.lower()} {', '.join(channels)}" for region, channels in theta_regions.items() if channels
                    ])
                    summary_alpha = "% Relative increase of alpha activity in " + ", ".join([
                        f"{region.lower()} {', '.join(channels)}" for region, channels in alpha_regions.items() if channels
                    ])

                    # Plot summary in a single figure
                    fig, ax = plt.subplots(figsize=(10, 5))
                    ax.axis('off')  # Hide axes for a cleaner look
                    text_content = f"Theta Activity Summary:\n{summary_theta}\n\nAlpha Activity Summary:\n{summary_alpha}"
                    ax.text(0.5, 0.5, text_content, fontsize=12, ha='center', va='center', wrap=True)
                    # ax.set_title('Theta/Alpha Findings')
                    
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
                def detect_deviations(data, sfreq, bands, threshold_factor=1.5):
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
                    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

                    # Display Alpha Peaks
                    axs[0].axis('off')
                    axs[0].text(0.5, 0.5, f"Alpha Peak Frequencies:\n{alpha_results}", ha='center', va='center', wrap=True, fontsize=34)
                    # axs[0].set_title("Alpha Peak Frequencies")

                    # Display Deviations Analysis
                    axs[1].axis('off')
                    axs[1].text(0.5, 0.5, f"Deviations Analysis:\n{deviations_text}", ha='center', va='center', wrap=True, fontsize=34)
                    # axs[1].set_title("Deviations Analysis")

                    # Display Leaky Gut Analysis
                    axs[2].axis('off')
                    axs[2].text(0.5, 0.5, leaky_gut_text, ha='center', va='center', wrap=True, fontsize=34)
                    # axs[2].set_title("Leaky Gut Syndrome Analysis")

                    plt.tight_layout()
                    fig.suptitle("Combined Analysis: Alpha Peaks, Deviations, and Leaky Gut Markers", fontsize=36)
                    return fig#fig_alpha, fig_dev, fig_gut

                combined_fig = plot_combined_analysis()

                def plot_combined_analysis():
                    fig, axs = plt.subplots(3, 1, figsize=(12, 18))

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
                        axs[1].set_title("Theta/Alpha Findings")
                        axs[1].axis('off')
                    
                    if combined_fig:
                        combined_fig_img = render_fig_to_array(combined_fig)
                        axs[2].imshow(combined_fig_img)
                        axs[2].set_title("Pathological Sign Detection")
                        axs[2].axis('off')
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
