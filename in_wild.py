import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
import datetime
import warnings
import re
from sklearn import linear_model, metrics
from sklearn.metrics import r2_score
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from matplotlib.pyplot import figure
from helper_preprocess import actigraph_add_datetime, watch_add_datetime, get_intensity, get_met_fitbit, get_met_freedson, get_met_vm3, get_met_crouter, get_metcart, get_met_matcart, get_train_data, extract_features
from helper_extraction import generate_table
from helper_model import get_intensity_coef, build_classification_model, pred_activity, set_realistic_met_estimate
warnings.filterwarnings("ignore")

# Define Path
ROOT_PATH_FSM = 'Y:/PrevMed/Alshurafa_Lab/Lab_Common/CalorieHarmony/'
ACC_PATH = '/In Wild/Wrist/Clean/Resampled/Accelerometer/'
GYRO_PATH = '/In Wild/Wrist/Clean/Resampled/Gyroscope/'

wrist_valid_dic = {'409':['2019-11-7'],
                   '410':['2019-11-13', '2019-11-14'],
                   '411':['2019-11-12'],
                   '412':['2019-11-15'],
                   '415':['2019-11-26'],
                   '416':['2019-12-5', '2019-12-6'],
                   '417':['2019-12-6', '2019-12-7'],
                   '419':['2020-1-15', '2020-1-16'],
                   '420':['2020-1-24', '2020-1-25', '2020-1-26'],
                   '421':['2020-1-31', '2020-2-1'],
                   '423':['2020-2-7', '2020-2-8'],
                   '424':['2020-2-12'],
                   '425':['2020-2-15', '2020-2-17'],
                   '427':['2020-2-21', '2020-2-22'],
                   '429':['2020-3-3','2020-3-4','2020-3-5'],
                   '431':['2020-3-12']
}

# classification model
from xgboost import XGBClassifier
model_classification = XGBClassifier()
model_classification.load_model("trained_model/classification_model.json")
model_classification

# regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

loaded_rf = joblib.load("trained_model/regression_model.joblib")
loaded_rf

def get_rmssd(df, norm = 'l2'):
    df_temp = df.dropna()
    acc_x = list(df_temp['accX'])
    acc_y = list(df_temp['accY'])
    acc_z = list(df_temp['accZ'])
    data = np.array([df_temp['accX'], df_temp['accY'], df_temp['accZ']])

    if(len(acc_x) != 0):
        if(norm == 'l2'):
            processed = preprocessing.normalize(data, norm='l2')
            acc_x = processed[0]
            acc_y = processed[1]
            acc_z = processed[2]
        if(norm == 'l1'):
            processed = preprocessing.normalize(data, norm='l1')
            acc_x = processed[0]
            acc_y = processed[1]
            acc_z = processed[2]
        if(norm == 'minmax'):
            scaler = MinMaxScaler()
            scaler.fit(data)
            processed = scaler.transform(data)
            acc_x = processed[0]
            acc_y = processed[1]
            acc_z = processed[2]
            
        temp_sum = 0
        for i in range(1, len(acc_x)):
            temp_sum = temp_sum + (acc_x[i] - acc_x[i - 1])**2 + (acc_y[i] - acc_y[i - 1])**2 + (acc_z[i] - acc_z[i - 1])**2
        rmssd = (temp_sum/len(acc_x))**(1/2)
        return(rmssd)
    else:
        return np.nan
    
def get_freq_intensity(df, fs, top, plot=False):
    df_temp = df.dropna()
    acc_x = list(df_temp['accX'])
    acc_y = list(df_temp['accY'])
    acc_z = list(df_temp['accZ'])
    y = []
    if(len(acc_x) != 0):
        for i in range(1, len(acc_x)):
            y.append(np.sqrt(acc_x[i]**2 + acc_y[i]**2 + acc_z[i]**2))   
            n = len(y) # length of the signal
            k = np.arange(n)
            T = n/fs
            frq = k/T # two sides frequency range
            frq = frq[:len(frq)//2] # one side frequency range
            Y = np.fft.fft(y)/n # dft and normalization
            Y = Y[:n//2]   
            
        if(plot):
            plt.figure(figsize=(10,6))
            plt.plot(frq[1:],abs(Y[1:])) # plotting the spectrum without 0Hz
            plt.xlabel('Freq (Hz)')
            plt.ylabel('|Y(freq)|')

        top = top + 1
        fr = list(abs(Y))
        result = frq[fr.index(sorted(fr,reverse=True)[:top][top-1])]
        return(result)
    else:
        return np.nan

# Get in-wild data 
p_in_wild = ['404','409','410','411','412','415','416','417','419','420','421','423','424','425','427','429','431']
p = p_in_wild[0]
each_date = wrist_valid_dic[p][0]

full_path_acc = 'data_phase_1/' + p + '/' + each_date + '/' + 'acc_resample.csv'
full_path_gyro = 'data_phase_1/' + p + '/' + each_date + '/' + 'gyro_resample.csv'

df_acc_raw = pd.read_csv(full_path_acc)
df_gyro_raw = pd.read_csv(full_path_gyro)

df_acc = watch_add_datetime(df_acc_raw)
df_gyro = watch_add_datetime(df_gyro_raw)
df_gyro.columns = ['Time','rotX', 'rotY', 'rotZ', 'Datetime']

# newvalue= (max'-min')/(max-min)*(value-max)+max'
# max/min = observed , max'/min' = normalized into
participant_list = ['1000','1002','1003','1004','1005','1006','1007','1008','1009','1010','1011','1012','1013','1014','1015','1016', '1017', '1018', '1019', '1020', '1021', '1022', '1023', '1024', '1025', '1026']
max_acc = 0
min_acc = 0
targets = ['accX', 'accY', 'accZ']

for each_col in targets:
    for each in participant_list:
        df_acc_sample = pd.read_csv('data_phase_2/' + each + '/Wild/Wrist/Clean/Resampled/Accelerometer/acc_resample.csv')
        max_acc += np.max(df_acc_sample[each_col])
        min_acc += np.min(df_acc_sample[each_col])
        
    max_acc = max_acc / len(participant_list)
    min_acc = min_acc / len(participant_list)
    max_old = np.max(df_acc[each_col])
    min_old = np.min(df_acc[each_col])
    value = df_acc[each_col]
    
    df_acc[each_col] = (max_acc-min_acc)/(max_old-min_old)*(value-max_old)+max_acc
    
df_acc.head(3)

def get_freq_intensity(df, fs, top, plot=False):
    df_temp = df.dropna()
    acc_x = list(df_temp['accX'])
    acc_y = list(df_temp['accY'])
    acc_z = list(df_temp['accZ'])
    y = []
    if(len(acc_x) != 0):
        for i in range(1, len(acc_x)):
            y.append(np.sqrt(acc_x[i]**2 + acc_y[i]**2 + acc_z[i]**2))   
            n = len(y) # length of the signal
            k = np.arange(n)
            T = n/fs
            frq = k/T # two sides frequency range
            frq = frq[:len(frq)//2] # one side frequency range
            Y = np.fft.fft(y)/n # dft and normalization
            Y = Y[:n//2]   
            
        if(plot):
            plt.figure(figsize=(10,6))
            plt.plot(frq[1:],abs(Y[1:])) # plotting the spectrum without 0Hz
            plt.xlabel('Freq (Hz)')
            plt.ylabel('|Y(freq)|')

        top = top + 1
        fr = list(abs(Y))
        result = frq[fr.index(sorted(fr,reverse=True)[:top][top-1])]
        return(result)
    else:
        return np.nan

def get_rmssd(df, norm = 'l2'):
    df_temp = df.dropna()
    acc_x = list(df_temp['accX'])
    acc_y = list(df_temp['accY'])
    acc_z = list(df_temp['accZ'])
    data = np.array([df_temp['accX'], df_temp['accY'], df_temp['accZ']])

    if(len(acc_x) != 0):
        if(norm == 'l2'):
            processed = preprocessing.normalize(data, norm='l2')
            acc_x = processed[0]
            acc_y = processed[1]
            acc_z = processed[2]
        if(norm == 'l1'):
            processed = preprocessing.normalize(data, norm='l1')
            acc_x = processed[0]
            acc_y = processed[1]
            acc_z = processed[2]
        if(norm == 'minmax'):
            scaler = MinMaxScaler()
            scaler.fit(data)
            processed = scaler.transform(data)
            acc_x = processed[0]
            acc_y = processed[1]
            acc_z = processed[2]
            
        temp_sum = 0
        for i in range(1, len(acc_x)):
            temp_sum = temp_sum + (acc_x[i] - acc_x[i - 1])**2 + (acc_y[i] - acc_y[i - 1])**2 + (acc_z[i] - acc_z[i - 1])**2
        rmssd = (temp_sum/len(acc_x))**(1/2)
        return(rmssd)
    else:
        return np.nan
    
def get_train_data(df, st, window_size, input_type='gyro'):
    et = st + pd.DateOffset(minutes=window_size/60)
    temp = df.loc[(df['Datetime'] >= st) & (df['Datetime'] < et)].reset_index(drop=True)
    if(input_type == 'gyro'):
        this_min_data = [temp['rotX'], temp['rotY'], temp['rotZ']]
    elif(input_type == 'acc'):
        this_min_data = [temp['accX'], temp['accY'], temp['accZ']]
    else:
        print('Invalid input data type')
        return
    return(this_min_data)

window_size = 60
st = start_time

# used for metadata for each window
l_participant = []
l_start = []
l_end = []

# used for classification model
data_training = [] 

# used for regression model
# 'gender', 'age', 'BMI', 'Intensity (Freq)', 'gender_age','gender_BMI','age_BMI','age_gender_BMI', 'Intensity (RMSSD_l1)'
l_intensity_freq = []
l_intensity_rmssd_l1 = []


range_time = int((end_time - start_time).seconds / window_size) - 1
for i in range(range_time):
    print(i)
    l_participant.append(p)
    
    if(i < 60/window_size*2): # exclude first two minutes
        st += pd.DateOffset(minutes=window_size/60)   
    else: 
        l_start.append(st)
        et = st + pd.DateOffset(minutes=window_size/60)
        l_end.append(et)
        temp = df_acc.loc[(df_acc['Datetime'] >= start_time) & (df_acc['Datetime'] < et)].reset_index(drop=True)
        l_intensity_freq.append(get_freq_intensity(temp, 100, 1, False))
        #l_intensity_rmssd_l1.append(get_rmssd(temp, norm = 'l1'))
        #data_training.append(get_train_data(df_gyro, st, window_size, 'gyro') + get_train_data(df_acc, st, window_size, 'acc'))
        st += pd.DateOffset(minutes=window_size/60)  

start_time = df_acc['Datetime'][0]
end_time = df_acc['Datetime'][df_acc.shape[0]-1]
df_result = pd.DataFrame({'Participant': l_participant, 'Start':l_start, 'End':l_end, 'Intensity (Freq)': l_intensity_freq, 'Intensity (RMSSD_l1)': l_intensity_rmssd_l1})


window_size = 60
st = start_time

# used for metadata for each window
l_participant = []
l_start = []
l_end = []

# used for classification model
data_training = [] 

# used for regression model
# 'gender', 'age', 'BMI', 'Intensity (Freq)', 'gender_age','gender_BMI','age_BMI','age_gender_BMI', 'Intensity (RMSSD_l1)'
l_intensity_freq = []
l_intensity_rmssd_l1 = []

for i in range(int((end_time - start_time).seconds / window_size) - 1):
    print(i)
    l_participant.append(p)
    
    if(i < 60/window_size*2 - 1): # exclude first two minutes
        st += pd.DateOffset(minutes=window_size/60)   
    else: 
        l_start.append(st)
        et = st + pd.DateOffset(minutes=window_size/60)
        l_end.append(et)
        
        temp = df_acc.loc[(df_acc['Datetime'] >= start_time) & (df_acc['Datetime'] < et)].reset_index(drop=True)
        l_intensity_freq.append(get_freq_intensity(temp, 100, 1, False))
        l_intensity_rmssd_l1.append(get_rmssd(temp, norm = 'l1'))
        data_training.append(get_train_data(df_gyro, st, window_size, 'gyro') + get_train_data(df_acc, st, window_size, 'acc'))
        st += pd.DateOffset(minutes=window_size/60) 
        
for i in range(len(data_training)):
        for j in range(6):
            if(data_training[i][j].shape[0] != window_size * 20):
                s_temp = pd.Series([0]*int(window_size * 20-data_training[i][j].shape[0]))
                data_training[i][j] = data_training[i][j].append(s_temp, ignore_index=True)

np_training = np.array(data_training)
data_train = extract_features(np_training)

# non-sedentary = 1
# sedentary = 0
classification = model_classification.predict(data_train)

data_regression = np.array(df_result[['gender', 'age', 'BMI', 'Intensity (Freq)', 'gender_age','gender_BMI','age_BMI','age_gender_BMI', 'Intensity (RMSSD_l1)']])

prediction = []
for i in range(len(classification)):
    if(classification[i] == 0):
        prediction.append(1) # sedentary = 1.0 MET
    else:
        prediction.append(loaded_rf.predict(data_regression[i].reshape(1,9))) # regression model

# Visualizing the results
import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
import warnings
from sklearn import linear_model, metrics
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
import scipy.stats as st
import datetime as dt
pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore")

intensity_c = {'Sedentary': '#a8d0e3', # lightblue
               'Light': '#f6b79b', # peach
               'Moderate / Vigorous': '#DE6454' # red
              }

intensity_legend = {'Sedentary': '#a8d0e3', # lightblue
               'Light': '#f6b79b', # peach
               'Moderate /\n Vigorous': '#DE6454' # red
              }


def set_intensity(row, col_name):
    if row[col_name]<1.5:
        return 'Sedentary'
    elif 1.5<=row[col_name]<3:
        return 'Light'
    elif row[col_name]>=3:
        return 'Moderate / Vigorous'
    
def inwild_preprocess(inlab, target_col, result_col):
    inlab = inlab[['Participant', 'Datetime','estimation',target_col]]
    inlab['Intensity'] = inlab.apply(set_intensity, col_name=target_col, axis=1)
    #inlab['avgMetCartEst'] = inlab[['MET (MetCart)', 'estimation']].mean(axis=1)
    inlab[result_col] = inlab.apply(lambda x: x[target_col] - x['estimation'], axis=1)
    return(inlab)

def plot_subplot(ax, xdata, ydata, df, xlabel, ylabel, title, color_dic, color_by, rmin, rmax):
    i=3
    for key in color_dic:
        temp = df.loc[df[color_by]==key]
        ax.scatter(temp[xdata], temp[ydata], c = color_dic[key], s=20, zorder=i)
        i-=1
        
    # x and y label
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    l = df[ydata].tolist()
    md = df[ydata].mean()
    sd = df[ydata].std()
    
    # calculating 95%CI around the mean, mean+-1.96s
    ci_mean = st.t.interval(0.95, len(l)-1, loc=np.mean(l), scale=st.sem(l))
    ci_plus1z = st.t.interval(0.95, len(l)-1, loc=np.mean(l) + 1.96*np.std(l), scale=(3*np.std(l)**2/len(l))**0.5)
    ci_minus1z = st.t.interval(0.95, len(l)-1, loc=np.mean(l) - 1.96*np.std(l), scale=(3*np.std(l)**2/len(l))**0.5)

    # 3 lines at mean, mean+1.96s, mean-1.96s
    ax.axhline(md,           color='black', linestyle='-', zorder=20)
    ax.axhline(md + 1.96*sd, color='gray', linestyle='--', zorder=20)
    ax.axhline(md - 1.96*sd, color='gray', linestyle='--', zorder=20)

    # x and y lim
    xmin, xmax = ax.get_xlim()
    xrange = xmax-xmin
    ax.set_xlim([xmin, xmax+xrange*0.21])
    ymin, ymax = ax.get_ylim()
    yrange = ymax-ymin
    
    textpos = yrange*0.024
    numpos = yrange*0.06
    rightmargin = xrange*0.2

    #text on the lines
    ax.text(xmax+rightmargin, md+textpos,'Mean', ha='right', zorder=20)
    #ax.text(xmax+rightmargin, md-numpos,round(md,2), ha='right', zorder=20)
    ax.text(xmax+rightmargin, md+textpos+1.96*sd,'+1.96 SD', ha='right', zorder=20)
    #ax.text(xmax+rightmargin, md-numpos+1.96*sd,round(md + 1.96*sd,2), ha='right', zorder=20)
    ax.text(xmax+rightmargin, md+textpos-1.96*sd,'-1.96 SD', ha='right', zorder=20)
    #ax.text(xmax+rightmargin, md-numpos-1.96*sd,round(md - 1.96*sd,2), ha='right', zorder=20)
    
    #shaded bands for 95% CI
    ax.axhspan(ci_mean[0], ci_mean[1], alpha=0.5, color='lightgray', zorder=1)
    ax.axhspan(ci_plus1z[0], ci_plus1z[1], alpha=0.5, color='lightgray', zorder=1)
    ax.axhspan(ci_minus1z[0], ci_minus1z[1], alpha=0.5, color='lightgray', zorder=1)
    
    ax.set_xticks((2, 4, 6, 8))
    
    ax.set_ylim([rmin, rmax])
    
    patches = [mpatches.Patch(color=color, label=label) for color, label in zip(intensity_legend.values(), intensity_legend.keys())]
    ax.legend(handles=patches, labels=[label for _, label in zip(intensity_legend.values(), intensity_legend.keys())], loc='upper right', ncol = 1)
    
    print('Upper 1.96 SD:', round(md + 1.96*sd,2))
    print(df[df[ydata] > round(md + 1.96*sd,2)].shape)
    
    print('Lower 1.96 SD:', round(md - 1.96*sd,2))
    print(df[df[ydata] < round(md - 1.96*sd,2)].shape)
    
df_kerr_wild = inwild_preprocess(df_new, 'kerr', 'Kerr-Estimation')
df_vm3_wild = inwild_preprocess(df_new, 'MET (VM3)', 'VM3-Estimation')

fig = plt.figure(figsize = (12,10))

color_dic = intensity_c
color_by = 'Intensity'

ax1 = fig.add_subplot(2, 2, 1)
#ax2 = fig.add_subplot(2, 2, 2, sharey=ax1)
#ax3 = fig.add_subplot(2, 2, 3, sharey=ax1)
#ax4 = fig.add_subplot(2, 2, 4)

ax1 = plot_subplot(ax1, 'kerr', 'Kerr-Estimation', df_kerr_wild,
                   'Kerr et al Estimation \n\n(a)', 'Kerr et al METs - WRIST METs',
                   'Kerr et al METs vs WRIST METs',
                   color_dic, color_by, -2, 2)

'''
ax2 = plot_subplot(ax2, 'MET (VM3)', 'VM3-Estimation', df_vm3_wild,
                   'VM3 METs \n\n(b)', 'VM3 METs - ActiGraph - WRIST METs', 
                   'VM3 METs vs WRIST METs',
                   color_dic, color_by, -6, 6)
'''
fig.tight_layout(pad=4)
plt.suptitle('In-wild Bland-Altman Plot')
plt.show()

# extract under- and over-estimated minutes
df_under = df_kerr_wild[df_kerr_wild['Kerr-Estimation'] > 0.62]
df_over = df_kerr_wild[df_kerr_wild['Kerr-Estimation'] < -0.63]
df_under.head(3)

df_under.to_csv('result/df_under.csv', index=False)
df_over.to_csv('result/df_over.csv', index=False)


def get_sorted_RGB(path, start_unix=None, end_unix=None):
    file_number_list_complete = []
    if((start_unix!=None) & (end_unix!=None)):
        for f in sorted(glob.glob(os.path.join(path, '*.jpg'))):
            if((int(f[-17:-4]) >= start_unix) and (int(f[-17:-4]) <= end_unix)):
                file_number_list_complete.append(f.replace('\\', '/'))
    return(file_number_list_complete)

df_under = pd.read_csv('result/df_under.csv')
df_over = pd.read_csv('result/df_over.csv')
df_under.shape[0]

df_under['Datetime'] = pd.to_datetime(df_under['Datetime'])
df_under['Unix'] = df_under['Datetime'].apply(lambda x: x.timestamp()*1000 + 21600000)

df_over['Datetime'] = pd.to_datetime(df_over['Datetime'])
df_over['Unix'] = df_over['Datetime'].apply(lambda x: x.timestamp()*1000 + 21600000)

df_under.head(3)

df_under = df_over

# Extract the video clips for over- and under-estimated minutes
import cv2
import ffmpeg
from os import listdir
from os.path import isfile, join
import glob
import shutil
#i = 1

df_under = df_over

for i in range(210, df_under.shape[0]):
    video_path = 'Y:/PrevMed/Alshurafa_Lab/Lab_Common/CalorieHarmony/' + str(df_under['Participant'][i]) + '/In Wild/Camera/Frame/' + str(str(df_under['Datetime'][i])[0:4]) + '_' + str(str(df_under['Datetime'][i])[5:7]) + '_' + str(str(df_under['Datetime'][i])[8:10]) + '/ ' + str(df_under['Datetime'][i])[11:13]
    print(video_path)

    file_number = get_sorted_RGB(video_path, start_unix=df_under['Unix'][i], end_unix=df_under['Unix'][i]+60000)
    if(len(file_number) == 0):
        continue
    else:
        # create temp image folder
        j = 1
        image_path_save = 'temp/'

        if(os.path.exists(image_path_save) == False):
            os.mkdir(image_path_save)  

        for path in file_number:
            frame = cv2.imread(path)
            cv2.imwrite(str(image_path_save + str(j).zfill(7) + '.jpg'), frame)
            j = j+1

        save_video_to = 'in_wild_inspect/over/' + str(df_under['Participant'][i]) + '/' 
        save_video_as = save_video_to + 'video_' + str(i) + '.mp4'

        if(os.path.exists(save_video_to) == False):
            os.mkdir(save_video_to)   

        # write into video
        images = [img for img in os.listdir(image_path_save) if img.endswith(".jpg")]
        frame = cv2.imread(os.path.join(image_path_save, images[0]))
        height, width, layers = frame.shape
        video = cv2.VideoWriter(save_video_as, 0, 1, frameSize=(width,height), fps=len(file_number)/60)
        for image in images:
            video.write(cv2.imread(os.path.join(image_path_save, image)))
        cv2.destroyAllWindows()
        video.release()

        #remove folder  
        shutil.rmtree(image_path_save)

df_over.shape[0]

file_number = get_sorted_RGB(video_path, start_unix=df_under['Unix'][i], end_unix=df_under['Unix'][i]+60000)
j = 1
image_path_save = 'temp/'
for path in file_number:
    frame = cv2.imread(path)
    cv2.imwrite(str(image_path_save + str(j).zfill(7) + '.jpg'), frame)
    j = i+1
shutil.rmtree(image_path_save)

save_video_to = 'temp_2/' + 'video.mp4'

image_folder = 'temp/'
video_name = 'video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, frameSize=(width,height), fps=len(file_number)/60)

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
