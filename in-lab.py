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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from matplotlib.pyplot import figure
from helper_preprocess import actigraph_add_datetime, watch_add_datetime, get_intensity, get_met_fitbit, get_met_freedson, get_met_vm3, get_met_crouter, get_metcart, get_met_matcart, get_train_data, extract_features
from helper_extraction import generate_table
from helper_model import get_intensity_coef, build_classification_model, pred_activity, set_realistic_met_estimate, regression

warnings.filterwarnings("ignore")

# Define Path
ROOT_PATH_FSM = 'Y:/PrevMed/Alshurafa_Lab/Lab_Common/CalorieHarmony/A. Phase 2 Participants/'
PATH_RESAMPLE_ACC = '/Wild/Wrist/Clean/Resampled/Accelerometer/'
PATH_RESAMPLE_GYRO = '/Wild/Wrist/Clean/Resampled/Gyroscope/'
participant_list = ['1000','1002','1003','1004','1005','1006','1007','1008','1009','1010','1011','1012','1013','1014','1015','1016', '1017', '1018', '1019', '1020', '1021', '1022', '1023', '1024', '1025', '1026']

# Define met_cart_dic and activity_estimate dictionaries
met_cart_dic = {'1000':'01/27/2022 13:43:15', '1001':'NA', '1002':'02/09/2022 10:30:45', '1003':'01/12/2022 15:47:36', '1004':'01/21/2022 13:49:59', '1005':'02/12/2022 11:57:42', '1006':'03/22/2022 10:23:05', '1007':'03/28/2022 12:49:21', '1008':'04/12/2022 12:11:31', '1009':'04/15/2022 12:37:46', '1010':'05/06/2022 11:14:05', '1011':'05/11/2022 12:02:56', '1012':'05/24/2022 16:42:47', '1013':'05/27/2022 11:34:26', '1014':'06/03/2022 11:50:35', '1015':'06/06/2022 10:41:15', '1016':'08/01/2022 12:47:00', '1017':'08/16/2022 13:11:40', '1018':'08/26/2022 11:54:56', '1019':'09/21/2022 11:41:39', '1020':'10/04/2022 11:17:21', '1021':'10/14/2022 13:52:46', '1022':'10/19/2022 18:31:47', '1023':'10/21/2022 11:42:49', '1024':'11/03/2022 11:40:41', '1025':'11/07/2022 10:46:36', '1026':'11/16/2022 11:07:17'}

# Compendium estimation
activity_estimate = {'Typing on a computer while seated':1.3, 'Rest':1, 'Walking 2 mph on treadmill':2.8, 'Walking 3.5 mph on treadmill':4.3, 'Standing while fidgeting':1.8, 'Squats (shoulder length legs, get down to 90 degree angle)':5, 'Reading a book or magazine while reclining':1.3, 'General aerobics video':7.3, 'Sweeping slowly ':2.3, 'Push-ups against the wall':3.8, 'Running 4 mph on a treadmill':6, 'Lying down while doing nothing':1.3, 'Chester Step Test (0.25 m step at a rate of 30 steps per minute)':8}

# Get demographic info
df_weight = pd.read_excel(ROOT_PATH_FSM + 'Participant Measurement Record.xlsx')
weight_list = []
mix_list = list(df_weight['Met Cart H/W'])
for i in range(len(mix_list)):
    x = re.search(r"/", mix_list[i])
    weight_list.append(float(mix_list[i][x.span()[0]+1:])*0.45359237) # convert lbs to kg

height_list = []
for i in range(len(mix_list)):
    x = mix_list[i].split('/')
    num = re.findall(r'\d+', x[0]) 
    num = ".".join(num)
    height_list.append(float(num) * 2.54) # convert inch to cm
    
participant_weight = {list(df_weight['P ID'])[i]: weight_list[i] for i in range(len(list(df_weight['P ID'])))}
participant_height = {list(df_weight['P ID'])[i]: height_list[i] for i in range(len(list(df_weight['P ID'])))}

df_age = pd.read_excel(ROOT_PATH_FSM + 'Participant Log.xlsx')
age_list = []
today = datetime.date.today()
for each_age in list(df_age['BD']):
    age_list.append(today.year - each_age.year)
participant_age = {list(df_age['P ID'])[i]: age_list[i] for i in range(len(list(df_age['P ID'])))}

# male = 1, female = 0
df_gender = pd.read_excel(ROOT_PATH_FSM + 'Participant Genders.xlsx')
gender_list = []
for each_gender in list(df_gender['Gender']):
    if(each_gender[0] == 'M'):
        gender_list.append(1)
    else:
        gender_list.append(0)
participant_gender = {list(df_gender['P ID'])[i]: gender_list[i] for i in range(len(list(df_gender['P ID'])))}

df_demographic = pd.DataFrame({'weight':pd.Series(participant_weight),'height':pd.Series(participant_height), 'gender':pd.Series(participant_gender),'age':pd.Series(participant_age)})
df_demographic.reset_index(inplace=True)
df_demographic = df_demographic.rename(columns = {'index':'Participant'})
df_demographic['BMI'] = df_demographic['weight'] / (df_demographic['height'] / 100)**2
# additional interaction terms
df_demographic['gender_age'] = df_demographic['age'] * df_demographic['gender']
df_demographic['gender_BMI'] = df_demographic['BMI'] * df_demographic['gender']
df_demographic['age_BMI'] = df_demographic['age'] * df_demographic['BMI']
df_demographic['age_gender_BMI'] = df_demographic['age'] * df_demographic['gender'] * df_demographic['BMI']

print('Number of participants:',len(participant_list))
df_demographic.head(3)
df_demographic.tail(3)

# Coefficient Part
def get_train(table_all, i):
    return(pd.concat(table_all[:i]+table_all[i+1:]))

def get_intensity_coef(df_table_all, gt_type='MetCart'):
    """
    This function takes the aggregated table and build a linear regression model.
    Parameters:
        :param df_table_all: the aggregated table
    """
    sedentary_activities = ['Rest', 'Typing on a computer while seated', 'Reading a book or magazine while reclining', 'lie down']    
    df_table_all = df_table_all.loc[~df_table_all['Activity'].isin(sedentary_activities)].reset_index()
    
    if(gt_type == 'MetCart'):
        l_met = df_table_all['MET (MetCart)'].tolist() 
    else:
        l_met = df_table_all['MET (Ainsworth)'].tolist()
        
    l_intensity = df_table_all['Intensity (ACC)'].tolist()
    l_met_final = [l_met[i] for i in range(len(l_met)) if not np.isnan(l_met[i]) and not np.isnan(l_intensity[i])]
    l_intensity_final = [l_intensity[i] for i in range(len(l_intensity)) if not np.isnan(l_met[i]) and not np.isnan(l_intensity[i])]

    met_reshaped = np.array(l_met_final).reshape(-1, 1)
    instensity_reshaped = np.array(l_intensity_final).reshape(-1, 1)

    regr = linear_model.LinearRegression(fit_intercept=False)
    regr.fit(instensity_reshaped, met_reshaped)
    return(instensity_reshaped, met_reshaped, regr.coef_[0][0], regr.intercept_)

# Visualization helper (IMU)
def plot_acc(df):
    plt.figure(figsize=(10,6))
    plt.plot(df[['Time','accX']].Time, df[['Time','accX']].accX, label='accX')
    plt.plot(df[['Time','accY']].Time, df[['Time','accY']].accY, label='accY')
    plt.plot(df[['Time','accZ']].Time, df[['Time','accZ']].accZ, label='accZ')
    plt.legend()
    plt.ylim(-15, 15)
    plt.show()
    
def smoothing(df, window):
    temp = df.copy(deep=True)
    temp['accX'] = temp['accX'].rolling(window).mean()
    temp['accY'] = temp['accY'].rolling(window).mean()
    temp['accZ'] = temp['accZ'].rolling(window).mean()
    return temp

def old_intensity(temp):
    """
    Calculate and return the minute level intensity value of given watch using the Panasonic equation.
    :param watch_df: the watch dataframe, used to calculate the intensity value
    :param st: the start time, data of the next minute will be used for calculation
    :return: the intensity value of the next minute using the Panasonic equation
    """
    sum_x_sq = 0
    sum_y_sq = 0
    sum_z_sq = 0
    sum_x = 0
    sum_y = 0
    sum_z = 0
    count = 0
    for i in range(0, len(temp)):
        if not np.isnan(temp['accX'][i]):
            sum_x_sq += temp['accX'][i] ** 2
            sum_y_sq += temp['accY'][i] ** 2
            sum_z_sq += temp['accZ'][i] ** 2
            sum_x += temp['accX'][i]
            sum_y += temp['accY'][i]
            sum_z += temp['accZ'][i]
            count += 1
    if count != 0:
        Q = sum_x_sq + sum_y_sq + sum_z_sq
        P = sum_x ** 2 + sum_y ** 2 + sum_z ** 2
        K = ((Q - P / count) / (count - 1)) ** 0.5
        return K
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

# Generate the train/test data for building in-lab model
def generate_table_test(PATH_RESAMPLE_ACC, PATH_RESAMPLE_GYRO, ROOT_PATH_FSM, met_cart_dic, participant_weight, activity_estimate, p, window_size, smoothed = False):

    if(window_size != 60):
        df_actigraph = pd.read_csv(ROOT_PATH_FSM + 'P' + str(p) + '/Actigraph/1 Sec Files/' + 'P' + str(p) + '_inlab_VM3.csv', skiprows=1)
    else:
        df_actigraph = pd.read_csv(ROOT_PATH_FSM + 'P' + str(p) + '/Actigraph/60 Sec Files/' + 'P' + str(p) + '_inlab_VM3.csv', skiprows=1)

    df_acc = pd.read_csv(os.path.join('data_phase_2/'+str(p) + PATH_RESAMPLE_ACC, 'acc_resample.csv'))
    df_gyro = pd.read_csv(os.path.join('data_phase_2/'+str(p) + PATH_RESAMPLE_GYRO, 'gyro_resample.csv'))
    df_actigraph = actigraph_add_datetime(df_actigraph)
    df_acc = watch_add_datetime(df_acc)
    df_gyro.columns = ['Time','rotX','rotY','rotZ'] # rename
    df_gyro = watch_add_datetime(df_gyro)
    df_logs = pd.read_excel(ROOT_PATH_FSM + 'P' + str(p) + '/Calorie Harmony Phase 2 Activity Log.xlsx',usecols="A:L",skiprows=1,nrows=24)
    df_logs.columns.values[0] = 'Activity' #rename the first column
    df_logs['Start Date'] = met_cart_dic[str(p)][:10]
    df_met = get_metcart(ROOT_PATH_FSM, met_cart_dic, p)

    l_participant = []
    l_datetime = []
    l_activity = []
    l_fit = []
    l_intensity = []
    l_intensity_rmssd = []
    l_intensity_freq = []
    l_intensity_rmssd_l1 = []
    l_intensity_rmssd_l2 = []
    l_intensity_rmssd_minmax = []
    l_mets_freedson = []
    l_mets_vm3 = []
    l_mets_crouter = []
    l_mets_ainsworth = []
    l_mets_metcart = []
    l_target = []
    data_training = []
    l_acc_data = []
    sedentary_activities = ['Rest', 'Typing on a computer while seated', 'Reading a book or magazine while reclining', 'Lying down while doing nothing']

    print('Processing: ', str(p))
    weight = participant_weight[int(p)]

    for i in range(len(df_logs['Activity'])): 
        try:
            if(df_logs['Include'][i]==1):
                st = pd.to_datetime(df_logs['Start Time'][i], format='%H:%M:%S')
                et = pd.to_datetime(df_logs['Expected Stop Time'][i], format='%H:%M:%S')

                ### DEBUG
                #print(df_logs['Activity'][i], 'Start:', st, 'End:', et)

                start_time = pd.Timestamp.combine(pd.to_datetime(df_logs['Start Date'][i]).date(), st.time())
                for j in range(int((et - st).seconds / window_size)):
                    if(j < 60/window_size*2 - 1): # exclude first two minutes
                        start_time += pd.DateOffset(minutes=window_size/60)   
                    else:
                        l_participant.append(p)
                        l_datetime.append(start_time)
                        l_activity.append(df_logs['Activity'][i])

                        cal = (df_logs['Google Fit Calorie Expenditure Reading at Stop'][i] - df_logs['Google Fit Calorie Expenditure Reading at Start'][i]) / int((et - st).seconds / 60)
                        l_fit.append(get_met_fitbit(cal, weight))

                        l_intensity.append(get_intensity(df_acc, start_time, window_size))

                        # get raw ECG data
                        et = start_time + pd.DateOffset(minutes=window_size/60)
                        temp = df_acc.loc[(df_acc['Datetime'] >= start_time) & (df_acc['Datetime'] < et)].reset_index(drop=True)

                        l_acc_data.append(temp)

                        l_intensity_rmssd.append(get_rmssd(temp, norm = None))
                        l_intensity_rmssd_l1.append(get_rmssd(temp, norm = 'l1'))
                        l_intensity_rmssd_l2.append(get_rmssd(temp, norm = 'l2'))
                        l_intensity_rmssd_minmax.append(get_rmssd(temp, norm = 'minmax'))
                        l_intensity_freq.append(get_freq_intensity(temp, 100, 1, False))

                        l_mets_freedson.append(get_met_freedson(df_actigraph, start_time, window_size))
                        l_mets_vm3.append(get_met_vm3(df_actigraph, start_time, window_size))
                        l_mets_crouter.append(get_met_crouter(df_actigraph, start_time, window_size))
                        l_mets_ainsworth.append(activity_estimate[df_logs['Activity'][i]])
                        l_mets_metcart.append(get_met_matcart(df_met, start_time, window_size))


                        # Acc/Gyro Feature Extraction Part for Activity Classification
                        if(df_logs['Activity'][i] in sedentary_activities):
                            l_target.append(0)
                        else:
                            l_target.append(1)

                        # extract both gyroscope and accelerometer data
                        data_training.append(get_train_data(df_gyro, start_time, window_size, 'gyro') + get_train_data(df_acc, start_time, window_size, 'acc'))           
                        start_time += pd.DateOffset(minutes=window_size/60)
            else:
                print('Participant', p, 'Activity', df_logs['Activity'][i], 'is not included')
        except:
            print('Participant', p, 'Activity', df_logs['Activity'][i], start_time, 'encounters a problem')
            continue

    df_result = pd.DataFrame({'Participant': l_participant, 'Datetime': l_datetime, 'Activity': l_activity, 
                              'Intensity (ACC)': l_intensity, 'Intensity (RMSSD)': l_intensity_rmssd,
                              'Intensity (RMSSD_l1)': l_intensity_rmssd_l1, 'Intensity (RMSSD_l2)': l_intensity_rmssd_l2, 
                              'Intensity (RMSSD_minmax)': l_intensity_rmssd_minmax, 'Intensity (Freq)': l_intensity_freq, 
                              'MET (GoogleFit)': l_fit, 'MET (Freedson)': l_mets_freedson, 'MET(Crouter)': l_mets_crouter, 
                              'MET (VM3)': l_mets_vm3, 'MET (Ainsworth)': l_mets_ainsworth, 'MET (MetCart)': l_mets_metcart})

    # replace value below 1.0 to 1.0 for MetCart MET values
    df_result['MET (MetCart)'].mask(df_result['MET (MetCart)'] < 1.0 ,'1.0', inplace=True)
    df_result['MET (MetCart)'] = pd.to_numeric(df_result['MET (MetCart)'],errors = 'coerce')
    
    # sanity check: array length equal to each other 
    for i in range(len(data_training)):
        for j in range(6):
            if(data_training[i][j].shape[0] != window_size * 20):
                s_temp = pd.Series([0]*int(window_size * 20-data_training[i][j].shape[0]))
                data_training[i][j] = data_training[i][j].append(s_temp, ignore_index=True)

    np_target_training = np.array(l_target)
    np_training = np.array(data_training)

    l_acc_data_final = l_acc_data
    
    # extract features
    data_train = extract_features(np_training)
    y_train = np_target_training
    temp_train = []
    y_temp = []
    idx_null_2 = []
    for i in range(len(data_train)):
        if(len(data_train[i])!=0):
            temp_train.append(data_train[i])
            y_temp.append(y_train[i])
        else:
            idx_null_2.append(i)

    data_train = np.array(temp_train)
    y_train = np.array(y_temp)
    df_result_final = df_result.drop(df_result.index[idx_null_2]).reset_index(drop=True)

    return(data_train, y_train, df_result_final, l_acc_data_final)

# Example P1000:
data_train, y_train, df_result_final, l_acc_data = generate_table_test(PATH_RESAMPLE_ACC, PATH_RESAMPLE_GYRO, ROOT_PATH_FSM, met_cart_dic, participant_weight, activity_estimate, '1000', window_size=60, smoothed=True)

# Entire Pipeline
participant_list = ['1000','1002','1003','1004','1005','1006','1007','1008','1009','1010','1011','1012','1013','1014','1015','1016', '1017', '1018', '1019', '1020', '1021', '1022', '1023', '1024', '1025', '1026']
window_size= 15 # define sliding window size

l_participant = []
train_list = []
y_list = []
table_all= []

for p in participant_list:
    l_participant.append(p)
    data_train, y_train, table, l_acc = generate_table_test(PATH_RESAMPLE_ACC, PATH_RESAMPLE_GYRO, ROOT_PATH_FSM, met_cart_dic, participant_weight, activity_estimate, p, window_size=window_size, smoothed=True)
    train_list.append(data_train)
    y_list.append(y_train)
    table_all.append(table)
    
# Save the processed files
df_table_all = pd.concat(table_all)
df_table_all['Participant'] = df_table_all['Participant'].astype(str).astype(int)
df_table_all =pd.merge(df_table_all,df_demographic, on='Participant', how='left')
print(df_table_all.shape)

# Save locally
df_table_all.to_csv('result/all_data_15sec.csv', index=False) # based on sliding window size

######################################################################################################################################################################################
# Activity Classification (Sedentary vs Non-sendentary)
# Regression Model (Estimate the METs)
# Building model based on LOSO

l_score_proposed = []
l_score_vm3 = []
l_score_freedson = []
l_score_googlefit = []
l_score_crouter = []
l_features = []
combinations = []

import itertools

features = ['gender', 'age', 'BMI', 'Intensity (Freq)', 'gender_age','gender_BMI','age_BMI','age_gender_BMI']
fixed_features = ['gender', 'age', 'BMI', 'Intensity (Freq)', 'gender_age','gender_BMI','age_BMI','age_gender_BMI']
for r in range(len(features)+1):
    for combination in itertools.combinations(features, r):
        combinations.append(combination)


#combinations = [['gender_age'], ['gender_BMI'], ['age_BMI'], ['gender_age', 'gender_BMI'], ['gender_age', 'age_BMI'],
#               ['gender_BMI','age_BMI'], ['gender_age', 'gender_BMI', 'age_BMI'], ['gender', 'age', 'BMI', 'Intensity (Freq)', 'gender_age','gender_BMI','age_BMI','age_gender_BMI']]

each = fixed_features 
#for each in combinations:
print('Using:', each)
subset = list(each)
l_features.append(' '.join(['Intensity (RMSSD_l1)'] + subset))

df_list = []
for i in range(len(participant_list)):
    leftout = int(participant_list[i])
    df_table_all = pd.concat(table_all)
    df_table_all['Participant'] = df_table_all['Participant'].astype(str).astype(int)
    df_table_all =pd.merge(df_table_all,df_demographic, on='Participant', how='left')

    # build classification model
    data_train_all = np.concatenate(train_list)
    y_train_all = np.concatenate(y_list)
    #data_train_all = np.concatenate(train_list[:i]+train_list[i+1:])
    #y_train_all = np.concatenate(y_list[:i]+y_list[i+1:])

    model = build_classification_model(data_train_all, y_train_all)
    
    # get the test data for classification (step 1)
    table_test = df_table_all[df_table_all['Participant']==leftout]
    data_test = train_list[i]
    table_pred = pred_activity(data_test, model, table_test)

    # classification result
    df_sedentary = table_pred[table_pred['model_classification']==0]
    df_sedentary['estimation'] = 1

    df_nonsedentary = table_pred[table_pred['model_classification']==1]
    
    if(df_nonsedentary.shape[0] != 0):
        df_nonsedentary = regression(p, df_table_all, df_nonsedentary, 'Intensity (RMSSD_l1)', model = 'rdf', features = subset)
        df_result = pd.concat([df_sedentary, df_nonsedentary])
    else:
        df_result = df_sedentary
    df_list.append(df_result)

df_table_est = pd.concat(df_list)

df_proposed = df_table_est[['Participant', 'Activity', 'estimation', 'MET (MetCart)']].dropna()

# Train and save classification and regression model (all data)
# classification model
df_table_all = pd.concat(table_all)
df_table_all['Participant'] = df_table_all['Participant'].astype(str).astype(int)
df_table_all =pd.merge(df_table_all,df_demographic, on='Participant', how='left')
data_train_all = np.concatenate(train_list)
y_train_all = np.concatenate(y_list)
model = build_classification_model(data_train_all, y_train_all)

# regression model
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import joblib

sedentary_activities = ['Rest', 'Typing on a computer while seated', 'Reading a book or magazine while reclining', 'Lying down while doing nothing']
light_activities = ['Walking 2 mph on treadmill', 'Standing while fidgeting', 'Sweeping slowly ']
vigorous_activities = ['Walking 3.5 mph on treadmill', 'Push-ups against the wall', 'Squats (shoulder length legs, get down to 90 degree angle)', 'General aerobics video', 'Running 4 mph on a treadmill', 'Chester Step Test (0.25 m step at a rate of 30 steps per minute)']

df_sed = df_table_all[df_table_all['Activity'].isin(sedentary_activities)]
df_light = df_table_all[df_table_all['Activity'].isin(light_activities)]
df_vig = df_table_all[df_table_all['Activity'].isin(vigorous_activities)]

df_nonsedentary = pd.concat([df_light, df_vig]).reset_index(drop=True)

# train regression
features = ['gender', 'age', 'BMI', 'Intensity (Freq)', 'gender_age','gender_BMI','age_BMI','age_gender_BMI', 'Intensity (RMSSD_l1)']
target = 'MET (MetCart)' 
x_train = np.array(df_nonsedentary[features])
y_train = np.array(df_nonsedentary[[target]])

regr = RandomForestRegressor()
regr.fit(x_train, y_train)
#regr.predict(x_train[0].reshape(1,9))

# save
joblib.dump(regr, "./trained_model/regression_model.joblib")