from flask import Flask, render_template, request, redirect
import os
import tensorflow as tf
import librosa
import numpy as np
from werkzeug.utils import secure_filename

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

import librosa

import pandas as pd

import pywt

from scipy.stats import skew
from scipy.stats import kurtosis
import zipfile

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GRU, Dense, Dropout, BatchNormalization, Flatten
from tensorflow.keras.regularizers import l1_l2


df_extracted_feature_MI = pd.DataFrame()

df_extracted_feature_Normal = pd.DataFrame()

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = tf.keras.models.load_model('2CnnLSTMmodelTuning_dropped.h5')

    
def extract_zip(zip_path, extract_folder):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)    


# Feature Extraction

# Fungsi untuk ekstraksi fitur MFCC
def feature_extraction2(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfcc = librosa.feature.mfcc(y=x, sr=sample_rate)
    return mfcc

# Fungsi untuk ekstraksi entropi energi Shannon
def shannon_energy_entropy(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    x_abs = np.abs(x)
    x_squared = x_abs ** 2
    normalized_histogram = np.histogram(x_squared, bins=256, density=True)[0]
    entropy = -np.sum(normalized_histogram * np.log2(normalized_histogram + np.finfo(float).eps))
    return entropy

# Fungsi untuk ekstraksi menggunakan wavelet
def wavelet_extraction(data, coeff, dwt, db, level):
    N = np.array(data).size
    a, ds = dwt[0], list(reversed(dwt[1:]))
    if coeff == 'a':
        return pywt.upcoef('a', a, db, level=level)[:N]
    elif coeff == 'd':
        return pywt.upcoef('d', ds[level-1], db, level=level)[:N]
    else:
        raise ValueError("Invalid coefficients: {}".format(coeff))

# Fungsi untuk ekstraksi laju lintasan nol (Zero Crossing Rate)
def zcr_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    zcr = librosa.feature.zero_crossing_rate(x)
    return zcr

# Fungsi untuk ekstraksi kontras spektral
def spectral_contrast_extraction(file_path):
    x, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    spectral_contrast = librosa.feature.spectral_contrast(y=x, sr=sample_rate)
    return spectral_contrast

# Fungsi untuk ekstraksi fitur RMS per frame
def rms_extraction_per_frame(file_path, frame_length=1024, hop_length=512):
    x, sr = librosa.load(file_path, res_type='kaiser_fast')
    rms_values = librosa.feature.rms(y=x, frame_length=frame_length, hop_length=hop_length)
    return rms_values.flatten()



@app.route('/')
def index():
    return render_template('submit.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/noWav')
def no_wav():
    return render_template('noWav.html')


ALLOWED_EXTENSIONS = {'zip'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/process', methods=['POST'])
def process():
    if 'audioFile' in request.files:
        audio_file = request.files['audioFile']
        if audio_file.filename != '' and allowed_file(audio_file.filename):
            zip_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(audio_file.filename))
            audio_file.save(zip_path)

            # Extract the uploaded zip file
            extract_folder = os.path.join(app.config['UPLOAD_FOLDER'])
            extract_zip(zip_path, extract_folder)

            # Inisialisasi DataFrame untuk menyimpan hasil ekstraksi fitur

            df_extracted_feature_MI = pd.DataFrame()

            df_extracted_feature_Normal = pd.DataFrame()

            # Process audio files in the extracted folders

            # Process audio files in the extracted folders
            for root, dirs, files in os.walk(extract_folder):
                for folder_name in dirs:
                    folder_path = os.path.join(root, folder_name)
                    if folder_name == 'Myocardial':  # Process myocardial folder
                        for filename in os.listdir(folder_path):
                            if filename.endswith('.wav'):
                                audio_path = os.path.join(folder_path, filename)

                                # Feature Extraction
                                raw_feature_MI = feature_extraction2(audio_path)

                                # Ekstraksi fitur RMS
                                rms_per_frame = rms_extraction_per_frame(audio_path)

                                # Ekstraksi Zero Crossing Rate (ZCR)
                                zcr_MI = zcr_extraction(audio_path)

                                # Ekstraksi Spectral Contrast
                                spectral_contrast_MI = spectral_contrast_extraction(audio_path)


                                mfcc_mean = np.mean(raw_feature_MI, dtype=np.float64)
                                mfcc_std = np.std(raw_feature_MI)
                                mfcc_max = np.max(raw_feature_MI)
                                mfcc_min = np.min(raw_feature_MI)
                                mfcc_med = np.median(raw_feature_MI)
                                mfcc_var = np.var(raw_feature_MI)

                                mfcc_Skew = skew(raw_feature_MI, axis=0, bias=True)
                                mfcc_Skew_mean = np.mean(mfcc_Skew)

                                mfcc_Q1 = np.percentile(raw_feature_MI, 25)
                                mfcc_Q3 = np.percentile(raw_feature_MI, 75)
                                mfcc_IQR = mfcc_Q3 - mfcc_Q1
                                mfcc_Range = mfcc_max - mfcc_min

                                mfcc_Kurt = kurtosis(raw_feature_MI, axis=0, bias=True)
                                mfcc_Kurt_mean = np.mean(mfcc_Kurt)

                                entropy_raw = shannon_energy_entropy(audio_path)

                                db = 'db6'
                                level = 4
                                data_wave, sr = librosa.load(audio_path, res_type='kaiser_fast')
                                coeffs = pywt.wavedec(data_wave, db, level=level)
                                A4 = wavelet_extraction(data_wave, 'a', coeffs, db, level)
                                D4 = wavelet_extraction(data_wave, 'd', coeffs, db, level)
                                D3 = wavelet_extraction(data_wave, 'd', coeffs, db, 3)
                                D2 = wavelet_extraction(data_wave, 'd', coeffs, db, 2)
                                D1 = wavelet_extraction(data_wave, 'd', coeffs, db, 1)
                                wavelets = A4 + D4 + D3 + D2 + D1

                                wavelet_mean = np.mean(wavelets,dtype=np.float64)
                                wavelet_max = np.max(wavelets)
                                wavelet_min = np.min(wavelets)
                                wavelet_med = np.median(wavelets)
                                wavelet_Q1 = np.percentile(wavelets, 25)
                                wavelet_Q3 = np.percentile(wavelets, 75)
                                wavelet_IQR = wavelet_Q3 - wavelet_Q1

                                # Hitung statistik RMS
                                rms_mean = np.mean(rms_per_frame)
                                rms_max = np.max(rms_per_frame)
                                rms_min = np.min(rms_per_frame)
                                rms_med = np.median(rms_per_frame)
                                rms_Q1 = np.percentile(rms_per_frame, 25)
                                rms_Q3 = np.percentile(rms_per_frame, 75)

                                rms_std = np.std(rms_per_frame)
                                rms_var= np.var(rms_per_frame)
                                rms_kurt = kurtosis(rms_per_frame, axis=None, bias=True)
                                rms_range = np.max(rms_per_frame) - np.min(rms_per_frame)
                                rms_skew = skew(rms_per_frame, axis=None, bias=True)


                                # Hitung statistik untuk ZCR dan Spectral Contrast
                                zcr_mean = np.mean(zcr_MI)
                                zcr_std = np.std(zcr_MI)
                                zcr_max = np.max(zcr_MI)
                                zcr_min = np.min(zcr_MI)
                                zcr_med = np.median(zcr_MI)
                                zcr_var = np.var(zcr_MI)
                                zcr_Skew = skew(zcr_MI, axis=None, bias=True)
                                zcr_Kurt = kurtosis(zcr_MI, axis=None, bias=True)

                                spectral_contrast_mean = np.mean(spectral_contrast_MI)
                                spectral_contrast_std = np.std(spectral_contrast_MI)
                                spectral_contrast_max = np.max(spectral_contrast_MI)
                                spectral_contrast_min = np.min(spectral_contrast_MI)
                                spectral_contrast_med = np.median(spectral_contrast_MI)
                                spectral_contrast_var = np.var(spectral_contrast_MI)
                                spectral_contrast_Skew = skew(spectral_contrast_MI, axis=None, bias=True)
                                spectral_contrast_Kurt = kurtosis(spectral_contrast_MI, axis=None, bias=True)

                                # Gabungkan hasil ekstraksi fitur ke dalam DataFrame
                                df_extracted_feature_MI = pd.concat([df_extracted_feature_MI, pd.DataFrame({
                                    'MFCC Means': [mfcc_mean],
                                    'MFCC std': [mfcc_std],
                                    'MFCC max': [mfcc_max],
                                    'Med_mfcc': [mfcc_med],
                                    'Var_mfcc': [mfcc_var],
                                    'Skew_mean_mfcc': [mfcc_Skew_mean],
                                    'Q1_mfcc': [mfcc_Q1],
                                    'Q3_mfcc': [mfcc_Q3],
                                    'IQR_mfcc': [mfcc_IQR],
                                    'MinMax_mfcc': [mfcc_Range],
                                    'Kurt_mean_mfcc': [mfcc_Kurt_mean],


                                    'Entropy': [entropy_raw],


                                    'Wavelet Means': [wavelet_mean],
                                    'Wavelet max': [wavelet_max],
                                    'Wavelet min': [wavelet_min],
                                    'Med_wavelet': [wavelet_med],
                                    'Q1_wavelet': [wavelet_Q1],
                                    'Q3_wavelet': [wavelet_Q3],
                                    'IQR_wavelet': [wavelet_IQR],

                                    'RMS Mean': [rms_mean],
                                    'RMS Max': [rms_max],
                                    'RMS Min': [rms_min],
                                    'RMS Median': [rms_med],
                                    'RMS Q1': [rms_Q1],
                                    'RMS Q3': [rms_Q3],

                                    'RMS_std': [rms_std],
                                    'RMS_var': [rms_var],
                                    'RMS_Skew': [rms_skew],


                                    'RMS_Range': [rms_range],
                                    'RMS_Kurt': [rms_kurt],


                                    'ZCR Mean': [zcr_mean],
                                    'ZCR Std': [zcr_std],
                                    'ZCR Max': [zcr_max],
                                    'ZCR Min': [zcr_min],
                                    'ZCR Median': [zcr_med],
                                    'ZCR Variance': [zcr_var],
                                    'ZCR Skewness': [zcr_Skew],
                                    'ZCR Kurtosis': [zcr_Kurt],

                                    'Spectral Contrast Mean': [spectral_contrast_mean],
                                    'Spectral Contrast Std': [spectral_contrast_std],
                                    'Spectral Contrast Max': [spectral_contrast_max],
                                    'Spectral Contrast Min': [spectral_contrast_min],
                                    'Spectral Contrast Median': [spectral_contrast_med],
                                    'Spectral Contrast Variance': [spectral_contrast_var],
                                    'Spectral Contrast Skewness': [spectral_contrast_Skew],
                                    'Spectral Contrast Kurtosis': [spectral_contrast_Kurt]

                                })])




                                print(f"Processed file in Myocardial folder: {audio_path}")  # Add logging





                    elif folder_name == 'Normal':  # Process normal folder
                        for filename in os.listdir(folder_path):
                            if filename.endswith('.wav'):
                                audio_path = os.path.join(folder_path, filename)

                                # Feature Extraction
                                raw_feature_Normal = feature_extraction2(audio_path)

                                # Ekstraksi fitur RMS
                                rms_per_frame = rms_extraction_per_frame(audio_path)

                                # Ekstraksi Zero Crossing Rate (ZCR)
                                zcr_Normal = zcr_extraction(audio_path)

                                # Ekstraksi Spectral Contrast
                                spectral_contrast_Normal = spectral_contrast_extraction(audio_path)


                                mfcc_mean = np.mean(raw_feature_Normal, dtype=np.float64)
                                mfcc_std = np.std(raw_feature_Normal)
                                mfcc_max = np.max(raw_feature_Normal)
                                mfcc_min = np.min(raw_feature_Normal)
                                mfcc_med = np.median(raw_feature_Normal)
                                mfcc_var = np.var(raw_feature_Normal)

                                mfcc_Skew = skew(raw_feature_Normal, axis=0, bias=True)
                                mfcc_Skew_mean = np.mean(mfcc_Skew)

                                mfcc_Q1 = np.percentile(raw_feature_Normal, 25)
                                mfcc_Q3 = np.percentile(raw_feature_Normal, 75)
                                mfcc_IQR = mfcc_Q3 - mfcc_Q1
                                mfcc_Range = mfcc_max - mfcc_min

                                mfcc_Kurt = kurtosis(raw_feature_Normal, axis=0, bias=True)
                                mfcc_Kurt_mean = np.mean(mfcc_Kurt)

                                entropy_raw = shannon_energy_entropy(audio_path)

                                db = 'db6'
                                level = 4
                                data_wave, sr = librosa.load(audio_path, res_type='kaiser_fast')
                                coeffs = pywt.wavedec(data_wave, db, level=level)
                                A4 = wavelet_extraction(data_wave, 'a', coeffs, db, level)
                                D4 = wavelet_extraction(data_wave, 'd', coeffs, db, level)
                                D3 = wavelet_extraction(data_wave, 'd', coeffs, db, 3)
                                D2 = wavelet_extraction(data_wave, 'd', coeffs, db, 2)
                                D1 = wavelet_extraction(data_wave, 'd', coeffs, db, 1)
                                wavelets = A4 + D4 + D3 + D2 + D1

                                wavelet_mean = np.mean(wavelets,dtype=np.float64)
                                wavelet_max = np.max(wavelets)
                                wavelet_min = np.min(wavelets)
                                wavelet_med = np.median(wavelets)
                                wavelet_Q1 = np.percentile(wavelets, 25)
                                wavelet_Q3 = np.percentile(wavelets, 75)
                                wavelet_IQR = wavelet_Q3 - wavelet_Q1

                                # Hitung statistik RMS
                                rms_mean = np.mean(rms_per_frame)
                                rms_max = np.max(rms_per_frame)
                                rms_min = np.min(rms_per_frame)
                                rms_med = np.median(rms_per_frame)
                                rms_Q1 = np.percentile(rms_per_frame, 25)
                                rms_Q3 = np.percentile(rms_per_frame, 75)

                                rms_std = np.std(rms_per_frame)
                                rms_var= np.var(rms_per_frame)
                                rms_kurt = kurtosis(rms_per_frame, axis=None, bias=True)
                                rms_range = np.max(rms_per_frame) - np.min(rms_per_frame)
                                rms_skew = skew(rms_per_frame, axis=None, bias=True)


                                # Hitung statistik untuk ZCR dan Spectral Contrast
                                zcr_mean = np.mean(zcr_Normal)
                                zcr_std = np.std(zcr_Normal)
                                zcr_max = np.max(zcr_Normal)
                                zcr_min = np.min(zcr_Normal)
                                zcr_med = np.median(zcr_Normal)
                                zcr_var = np.var(zcr_Normal)
                                zcr_Skew = skew(zcr_Normal, axis=None, bias=True)
                                zcr_Kurt = kurtosis(zcr_Normal, axis=None, bias=True)

                                spectral_contrast_mean = np.mean(spectral_contrast_Normal)
                                spectral_contrast_std = np.std(spectral_contrast_Normal)
                                spectral_contrast_max = np.max(spectral_contrast_Normal)
                                spectral_contrast_min = np.min(spectral_contrast_Normal)
                                spectral_contrast_med = np.median(spectral_contrast_Normal)
                                spectral_contrast_var = np.var(spectral_contrast_Normal)
                                spectral_contrast_Skew = skew(spectral_contrast_Normal, axis=None, bias=True)
                                spectral_contrast_Kurt = kurtosis(spectral_contrast_Normal, axis=None, bias=True)

                                # Gabungkan hasil ekstraksi fitur ke dalam DataFrame
                                df_extracted_feature_Normal = pd.concat([df_extracted_feature_Normal, pd.DataFrame({
                                    'MFCC Means': [mfcc_mean],
                                    'MFCC std': [mfcc_std],
                                    'MFCC max': [mfcc_max],
                                    'Med_mfcc': [mfcc_med],
                                    'Var_mfcc': [mfcc_var],
                                    'Skew_mean_mfcc': [mfcc_Skew_mean],
                                    'Q1_mfcc': [mfcc_Q1],
                                    'Q3_mfcc': [mfcc_Q3],
                                    'IQR_mfcc': [mfcc_IQR],
                                    'MinMax_mfcc': [mfcc_Range],
                                    'Kurt_mean_mfcc': [mfcc_Kurt_mean],


                                    'Entropy': [entropy_raw],


                                    'Wavelet Means': [wavelet_mean],
                                    'Wavelet max': [wavelet_max],
                                    'Wavelet min': [wavelet_min],
                                    'Med_wavelet': [wavelet_med],
                                    'Q1_wavelet': [wavelet_Q1],
                                    'Q3_wavelet': [wavelet_Q3],
                                    'IQR_wavelet': [wavelet_IQR],

                                    'RMS Mean': [rms_mean],
                                    'RMS Max': [rms_max],
                                    'RMS Min': [rms_min],
                                    'RMS Median': [rms_med],
                                    'RMS Q1': [rms_Q1],
                                    'RMS Q3': [rms_Q3],

                                    'RMS_std': [rms_std],
                                    'RMS_var': [rms_var],
                                    'RMS_Skew': [rms_skew],


                                    'RMS_Range': [rms_range],
                                    'RMS_Kurt': [rms_kurt],


                                    'ZCR Mean': [zcr_mean],
                                    'ZCR Std': [zcr_std],
                                    'ZCR Max': [zcr_max],
                                    'ZCR Min': [zcr_min],
                                    'ZCR Median': [zcr_med],
                                    'ZCR Variance': [zcr_var],
                                    'ZCR Skewness': [zcr_Skew],
                                    'ZCR Kurtosis': [zcr_Kurt],

                                    'Spectral Contrast Mean': [spectral_contrast_mean],
                                    'Spectral Contrast Std': [spectral_contrast_std],
                                    'Spectral Contrast Max': [spectral_contrast_max],
                                    'Spectral Contrast Min': [spectral_contrast_min],
                                    'Spectral Contrast Median': [spectral_contrast_med],
                                    'Spectral Contrast Variance': [spectral_contrast_var],
                                    'Spectral Contrast Skewness': [spectral_contrast_Skew],
                                    'Spectral Contrast Kurtosis': [spectral_contrast_Kurt]

                                })])

                                print(f"Processed file in Normal folder: {audio_path}")  # Add logging
                    else:
                        print(f"Ignoring folder: {folder_name}")  # Add logging for any other folders


            # Reset index DataFrame
            df_extracted_feature_MI.reset_index(drop=True, inplace=True)

            df_extracted_feature_Normal.reset_index(drop=True, inplace=True)

            # Gabungkan DataFrame MI dan Normal
            df_combined = pd.concat([df_extracted_feature_MI, df_extracted_feature_Normal], ignore_index=True)

            # Tambahkan kolom label
            df_combined['Label'] = ['MI'] * len(df_extracted_feature_MI) + ['Normal'] * len(df_extracted_feature_Normal)


            label_encoder = LabelEncoder()
            df_combined['Encoded_Label'] = label_encoder.fit_transform(df_combined['Label'])


            columns_to_drop = [

                'IQR_wavelet',
                'Q3_wavelet',
                'Q1_wavelet',
                'ZCR Mean',

                'Wavelet min',
                'ZCR Min',
                'RMS Max',
                'RMS_Range',
                'RMS Min',

                'RMS Q1',
                'Wavelet Means',
                'Med_wavelet',
                'RMS_var'
            ]



            # Dropping columns that exist in the dataframe to avoid KeyErrors
            columns_to_drop = [col for col in columns_to_drop if col in df_combined.columns]

            # Dropping the specified columns
            df_combined = df_combined.drop(columns=columns_to_drop)

            X_train, X_test, y_train, y_test = train_test_split(
                df_combined.drop(['Label', 'Encoded_Label'], axis=1),
                df_combined['Encoded_Label'],
                test_size=0.3,
                random_state=42
            )

            # MinMax Scaler
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
            X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))


            mymodel = Sequential()
            # First Conv1D layer
            mymodel.add(Conv1D(filters=16, kernel_size=3, input_shape=(33, 1), activation='relu'))
            mymodel.add(BatchNormalization())
            mymodel.add(MaxPooling1D(pool_size=2))

            # Conv1D layer kedua
            mymodel.add(Conv1D(filters=16, kernel_size=3, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))
            mymodel.add(BatchNormalization())
            mymodel.add(MaxPooling1D(pool_size=2))
            mymodel.add(Dropout(0.2))

            # GRU layer
            mymodel.add(GRU(64, activation='relu', return_sequences=True))
            mymodel.add(Dropout(0.2))

            # GRU layer 2
            mymodel.add(GRU(32, activation='relu'))
            mymodel.add(Dropout(0.2))

            # Dense layer
            mymodel.add(Dense(128, activation='relu'))
            mymodel.add(Dropout(0.5))

            # Output layer
            mymodel.add(Dense(1, activation='sigmoid'))

            # Compile the model
            mymodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        
            epochs = 100
            batch_size = 32

            history = mymodel.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3)

            history


            lossTrain, accuracyTrain = mymodel.evaluate(X_train_reshaped, y_train)
            print(f"Loss: {lossTrain}, Accuracy: {accuracyTrain}")

            # Render the result on a new HTML page (you can customize this)
            return render_template('result.html', prediction=accuracyTrain, loss=lossTrain, df_extracted_feature=df_combined)

        else:
            return redirect('/noWav')
    
    return 'File upload failed.'

if __name__ == '__main__':
    app.run(debug=True)
