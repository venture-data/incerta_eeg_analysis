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
                    
                   
                        
                    global_raw_ica = cleaned_raw                    
                    
                    global_ica = ica
                    
                    max_time = int(raw.times[-1])
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
        elif plot_type == "ica_properties":
            figs = global_ica.plot_properties(global_raw_ica ,show=False)
    
            
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
