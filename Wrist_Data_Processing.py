import pandas as pd
import os
import numpy as np
import pytz
from datetime import datetime
from sort_resample_organize import resample
import warnings
warnings.filterwarnings("ignore")

# Define Path
ROOT_PATH_ACC = '/Wild/Wrist/Clean/Original/Accelerometer/'
ROOT_PATH_GYRO = '/Wild/Wrist/Clean/Original/Gyroscope/'
ROOT_PATH_FSM = 'Y:/PrevMed/Alshurafa_Lab/Lab_Common/CalorieHarmony/A. Phase 2 Participants/'
PATH_RESAMPLE_ACC = '/Wild/Wrist/Clean/Resampled/Accelerometer/'
PATH_RESAMPLE_GYRO = '/Wild/Wrist/Clean/Resampled/Gyroscope/'

# Define list of participants + start datetime
participant_list = ['1000','1001','1002','1003','1004','1005','1006','1007','1008','1009','1010','1011','1012','1013','1014','1015']
participant_list_2 = ['1016', '1017', '1018', '1019', '1020', '1021', '1022', '1023']
participant_list_3 = ['1019', '1020', '1022']
participant_list_4 = ['1024', '1025', '1026']
met_cart_dic = {'1000':'01/27/2022 13:43:15',
                '1001':'NA',
                '1002':'02/09/2022 10:30:45',
                '1003':'01/12/2022 15:47:36',
                '1004':'01/21/2022 13:49:59',
                '1005':'02/12/2022 11:57:42',
                '1006':'03/22/2022 10:23:05',
                '1007':'03/28/2022 12:49:21',
                '1008':'04/12/2022 12:11:31',
                '1009':'04/15/2022 12:37:46',
                '1010':'05/06/2022 11:14:05',
                '1011':'05/11/2022 12:02:56',
                '1012':'05/24/2022 16:42:47',
                '1013':'05/27/2022 11:34:26',
                '1014':'06/03/2022 11:50:35',
                '1015':'06/06/2022 10:41:15',
                '1016':'08/01/2022 12:47:00',
                '1017':'08/16/2022 13:11:40',
                '1018':'08/26/2022 11:54:56',
                '1019':'09/21/2022 11:41:39',
                '1020':'10/04/2022 11:17:21',
                '1021':'10/14/2022 13:52:46',
                '1022':'10/19/2022 18:31:47',
                '1023':'10/21/2022 11:42:49',
                '1024':'11/03/2022 11:40:41',
                '1025':'11/07/2022 10:46:36',
                '1026':'11/16/2022 11:07:17'
}


def merge_files(path):
    dir_list = os.listdir(path)
    dir_list.sort()
    df_list = []
    for each_dir in dir_list:
        df_list.append(pd.read_csv(path + each_dir))
    df_merged = pd.concat(df_list)    
    df_merged['date'] = pd.to_datetime(df_merged['datetime']).dt.date
    df_merged = df_merged.sort_values(by=['datetime'],ascending=True).reset_index(drop=True)
    return(df_merged)

def get_utc_date(local_time_string):
    # local_time_string: met_cart_dic['1000']
    local = pytz.timezone("America/Chicago")
    naive = datetime.strptime(local_time_string, '%m/%d/%Y %H:%M:%S')
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return(utc_dt.strftime("%Y-%m-%d"))

# Saving directory
ROOT_PATH_FSM = 'Y:/PrevMed/Alshurafa_Lab/Lab_Common/CalorieHarmony/'
ACC_PATH = '/In Wild/Wrist/Clean/Resampled/Accelerometer/'
GYRO_PATH = '/In Wild/Wrist/Clean/Resampled/Gyroscope/'

for p in participant_list_4:
    print('processing: ', p)
    try:
        df_acc = merge_files('data_phase_2/' + str(p) + ROOT_PATH_ACC)
        df_gyro = merge_files('data_phase_2/' + str(p) + ROOT_PATH_GYRO)
        
        date1 = get_utc_date(met_cart_dic[str(p)])
        if(p == '1022'):
            date2 = '2022-10-20'
        else:
            date2 = date1[:-1] + str(int(date1[-1])+1)
        df_acc_corrected = df_acc[(df_acc['date']==pd.to_datetime(date1))|(df_acc['date']==pd.to_datetime(date2))].reset_index(drop=True)
        df_gyro_corrected = df_gyro[(df_gyro['date']==pd.to_datetime(date1))|(df_gyro['date']==pd.to_datetime(date2))].reset_index(drop=True)

        # resample to 20 Hz
        df_resample_acc = resample(df_acc_corrected[['Time','accX','accY','accZ']], 'Time', 20)
        df_resample_gyro = resample(df_gyro_corrected[['Time','rotX', 'rotY', 'rotZosboxe']], 'Time', 20)

        df_resample_acc.to_csv(os.path.join('data_phase_2/'+str(p)+PATH_RESAMPLE_ACC, 'acc_resample.csv'),index=False)
        df_resample_gyro.to_csv(os.path.join('data_phase_2/'+str(p)+PATH_RESAMPLE_GYRO, 'gyro_resample.csv'),index=False)
    except:
        print('Invalid:', p)

def merge_files_inwild(path):
    dir_list = os.listdir(path)
    dir_list.sort()
    df_list = []
    for each_dir in dir_list:
        df_list.append(pd.read_csv(path + each_dir + '/accel_data.csv'))      
    df_merged = pd.concat(df_list)    
    df_merged['date'] = pd.to_datetime(df_merged['DT']).dt.date
    df_merged = df_merged.sort_values(by=['DT'],ascending=True).reset_index(drop=True)
    return(df_merged)

def get_utc_date(local_time_string):
    # local_time_string: met_cart_dic['1000']
    local = pytz.timezone("America/Chicago")
    naive = datetime.strptime(local_time_string, '%m/%d/%Y %H:%M:%S')
    local_dt = local.localize(naive, is_dst=None)
    utc_dt = local_dt.astimezone(pytz.utc)
    return(utc_dt.strftime("%Y-%m-%d"))

wrist_valid_dic = {'404':['2019-10-8', '2019-10-10'],
                   '409':['2019-11-7'],
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

p_in_wild = ['404','409','410','411','412','415','416','417','419','420','421','423','424','425','427','429','431']
for p in p_in_wild:
    print('processing', p)
    dates = wrist_valid_dic[str(p)]
    for each_date in dates:
        full_path_acc = ROOT_PATH_FSM + 'P' + p + ACC_PATH + each_date + '/'
        full_path_gyro = ROOT_PATH_FSM + 'P' + p + GYRO_PATH + each_date + '/'
        
        df_acc = merge_files_inwild(full_path_acc) # merge acc files for the entire day
        df_acc_resampled = resample(df_acc[['Time','accX','accY','accZ']], 'Time', 20)
        df_gyro = merge_files_inwild(full_path_gyro) # merge acc files for the entire day
        try:
            df_gyro_resampled = resample(df_gyro[['Time','accX','accY','accZ']], 'Time', 20)
        except:
            df_gyro_resampled = resample(df_gyro[['Time','rotX','rotY','rotZ']], 'Time', 20)

        
        outdir = os.path.join('data_phase_1/' + str(p) + '/')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outdir = os.path.join('data_phase_1/' + str(p) + '/'  + str(each_date) + '/')
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        df_acc_resampled.to_csv(outdir + 'acc_resample.csv', index=False)
        df_gyro_resampled.to_csv(outdir + 'gyro_resample.csv', index=False)