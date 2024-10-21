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



# OpenAI API Key setup
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global global_raw, global_raw_ica, global_ica

    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file:
            file_ext = os.path.splitext(uploaded_file.filename)[1]
            if file_ext.lower() != '.edf':
                flash('Invalid file format! Please upload a .edf file.', 'error')
                return redirect(request.url)
            
            try:
                f = request.files['file']
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
                f.save(filepath)

                # Load the EEG data using MNE
                raw = mne.io.read_raw_edf(filepath, preload=True)
                raw.rename_channels({ch: ch.replace('EEG ', '') for ch in raw.ch_names if ch.startswith('EEG ')})
                # Remove unnecessary channels before applying the montage
                channels_to_drop = ['Bio1-2', 'Bio3-4', 'ECG', 'Bio4', 'VSyn', 'ASyn', 'LABEL']
                raw.drop_channels(channels_to_drop)
                
                # Now apply the montage
                montage = mne.channels.make_standard_montage('standard_1020')
                raw.set_montage(montage, on_missing='ignore')  # Use 'ignore' to skip missing channels if needed                
                # Filter and clean raw data
                raw.filter(l_freq=1., h_freq=120., picks='eeg')
                raw.notch_filter(freqs=50, picks='eeg')

                # ASR and ICA
                prep_params = {
                    "ref_chs": raw.ch_names, 
                    "reref_chs": raw.ch_names,
                    "line_freqs": [50], "max_iterations": 5
                }
                prep = PrepPipeline(raw, prep_params, montage='standard_1020')
                prep.fit()
                raw_clean_asr = prep.raw
                raw_clean_asr.set_eeg_reference(ref_channels='average')

                # ICA
                raw_filtered = raw_clean_asr.copy().filter(l_freq=0.3, h_freq=40.)
                ica = mne.preprocessing.ICA(n_components=min(25, len(raw.ch_names)), random_state=97, max_iter=1000)
                ica.fit(raw_filtered)
                ica.apply(raw_filtered)

                global_raw_ica = raw_filtered

                # Artifact detection based on thresholds
                threshold = 100e-6  # 100 µV threshold for detecting artifacts
                artifact_times = []
                for ch in raw_filtered.ch_names:
                    data, times = raw_filtered[ch, :]
                    peaks, _ = find_peaks(np.abs(data[0]), height=threshold)
                    for peak in peaks:
                        start = max(0, peak - int(5 * raw.info['sfreq']))
                        end = min(len(times) - 1, peak + int(5 * raw.info['sfreq']))
                        artifact_times.append((start, end))

                # Power Spectrum Analysis (FFT)
                fft_data = np.fft.rfft(raw_filtered.get_data())
                freqs = np.fft.rfftfreq(len(raw_filtered.times), d=1. / raw.info['sfreq'])
                power_spectra = np.abs(fft_data) ** 2

                power_threshold = np.mean(power_spectra, axis=1) + 2 * np.std(power_spectra, axis=1)
                increased_power_channels = {}
                for idx, ch in enumerate(raw_filtered.ch_names):
                    increased_freqs = freqs[power_spectra[idx] > power_threshold[idx]]
                    if increased_freqs.size > 0:
                        increased_power_channels[ch] = increased_freqs

                # Prepare output with increased power frequencies
                findings = []
                for ch, freqs in increased_power_channels.items():
                    findings.append(f"Channel {ch}: Increased power at frequencies {freqs}")
                
                # Render results
                return render_template('upload_with_topomap_dropdown.html', max_time=int(raw_filtered.times[-1]), findings=findings)

            except Exception as e:
                print(f"Error processing file: {e}")
                return "Error processing file", 500

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
            fig = global_raw.plot(start=start_time, duration=4, n_channels=19, show=False)
        elif plot_type == 'cleaned' and global_raw_ica:
            scalings = {'eeg': 8e-6}  # Scale EEG channels to 20 µV
            fig = global_raw_ica.plot(start=start_time, duration=4, show=False,scalings=scalings)
            
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
