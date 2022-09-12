import mne
import os
import os.path as op
import numpy as np
import pandas as pd

freq_range = '2_40_step_2_time_bandwidth_by_default_4_early_log'

subjects = []
for i in range(0,63):
    if i < 10:
        subjects += ['P00' + str(i)]
    else:
        subjects += ['P0' + str(i)]
    
subjects_new = ['P301', 'P304', 'P307', 'P308', 'P309', 'P311', 'P312', 'P313', 'P314', 'P316', 'P318', 'P320', 'P321', 'P322', 'P323', 'P324', 'P325', 'P326', 'P327', 'P328', 'P329', 'P330', 'P331', 'P332', 'P333', 'P334', 'P335', 'P336', 'P338', 'P340', 'P341', 'P342', 'P063', 'P064', 'P065', 'P066', 'P067']

subjects = subjects + subjects_new

rounds = [1, 2, 3, 4, 5, 6]

trial_type = ['norisk', 'prerisk', 'risk', 'postrisk']

os.makedirs('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}'.format(freq_range), exist_ok = True)
os.makedirs('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}/ave_into_subj'.format(freq_range), exist_ok = True)

# донор (donor creation see make_donor_for_tfr_plot.ipynb)
temp = mne.time_frequency.read_tfrs('/net/server/data/Archive/prob_learn/vtretyakova/Nikita_mio_cleaned/tfr_plots/2_40_step_2_time_bandwidth_by_default_4_early_log/donor_evoked.h5', condition=None)[0]

n = temp.data.shape[2] # количество временных точек (берем у донора, если донор из тех же данных.
fr = temp.data.shape[1] # number of frequencies  (берем у донора, если донор из тех же данных.


for subj in subjects:
    for t in trial_type:
    
####################### IF DATA OF SINGLE EPOCHES  ##########################################
        '''
        ################################ Positive feedback ################################
        positive_fb = np.empty((0, 306, fr, n)) # эпохи х каналы х частоты (например: от 2 до 40 Гц, с шагом 2 Гц - 2, 4 ... 40, получается 20 частот) х временные точки
        for r in rounds:
            try:
                epochs_positive = mne.time_frequency.read_tfrs('/net/server/data/Archive/prob_learn/data_processing/TF_plots/{0}_epo/{1}_run{2}_{3}_fb_cur_positive_{0}-epo.h5'.format(freq_range, subj, r, t), condition=None)[0]  
                
                positive_fb = np.vstack([positive_fb, epochs_positive.data])
               
                
            except (OSError):
                
                print(f'{subj} {t} run {r} fb negative is not exist')

        ###### Шаг 1. Усреднили все положительные фидбеки внутри испытуемого (между блоками 1 -6) #################
        if positive_fb.size != 0:
            positive_fb_mean = positive_fb.mean(axis = 0) # усредняем по оси количества эпох
            positive_fb_mean = positive_fb_mean.reshape(1, 306, fr, n) # добавляем ось для фидбека
            
        else:
            positive_fb_mean = np.empty((0, 306, fr, n))
            
            
        ########################## Negative feedback #############################
        negative_fb = np.empty((0, 306, fr, n))
        for r in rounds:
            try:
                
                epochs_negative = mne.time_frequency.read_tfrs('/net/server/data/Archive/prob_learn/data_processing/TF_plots/{0}_epo/{1}_run{2}_{3}_fb_cur_negative_{0}-epo.h5'.format(freq_range, subj, r, t), condition=None)[0]  
                
                           
                negative_fb = np.vstack([negative_fb, epochs_negative.data])
             
                
            except (OSError):
                print(f'{subj} {t} run {r} fb negative is not exist')

        ###### Шаг 1. Усреднили все отрицательные фидбеки внутри испытуемого (между блоками 1 -6) #################
        if negative_fb.size != 0:
            negative_fb_mean = negative_fb.mean(axis = 0) # усредняем по оси количества эпох
            
            negative_fb_mean = negative_fb_mean.reshape(1, 306, fr, n) # добавляем ось для фидбека
        else:
            negative_fb_mean = np.empty((0, 306, fr, n))
        ####################### Шаг 2 усреднения. Усредняем данные внутри испытуемого #####################################
        data_into_subj = np.vstack([negative_fb_mean, positive_fb_mean])
        
        if data_into_subj.size != 0:
            temp.data = data_into_subj.mean(axis = 0) # усредняем данные для двух фидбеков внутри испытуемого (т.е. по оси фидбека)
        
            # сохраняем данные, усредненные внутри испытуемого. Шаг усредения 3, это усреднение между испытуемыми делается при рисовании топомапов
            
            temp.save('/net/server/data/Archive/prob_learn/data_processing/TF_plots/{0}/ave_into_subj/{1}_{2}_{0}_resp.h5'.format(freq_range, subj, t))
            
        else:
            print(f'{subj} has no feedbacks in {t}')
            pass
        '''
######################## IF DATA OF AVERAGED EPOCHES  ##########################################
        ################################ Positive feedback ################################
        positive_fb = np.empty((0, 306, fr, n)) # RUNS х каналы х частоты (например: от 2 до 40 Гц, с шагом 2 Гц - 2, 4 ... 40, получается 20 частот) х временные точки
        positive_fb_itc = np.empty((0, 306, fr, n))
        
        for r in rounds:
            try:
                # power
                positive = mne.time_frequency.read_tfrs('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}/power_{1}_run{2}_{3}_fb_cur_positive_{0}.h5'.format(freq_range, subj, r, t), condition=None)[0]
                positive_data = positive.data.reshape(1, 306, fr, n) # добавляем ось для run
                positive_fb = np.vstack([positive_fb, positive_data])
               
                #itc
                positive_itc = mne.time_frequency.read_tfrs('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}/itc_{1}_run{2}_{3}_fb_cur_positive_{0}.h5'.format(freq_range, subj, r, t), condition=None)[0]
                positive_data_itc = positive_itc.data.reshape(1, 306, fr, n) # добавляем ось для run
                positive_fb_itc = np.vstack([positive_fb_itc, positive_data_itc])                
                                
            except (OSError):
                
                print(f'{subj} {t} run {r} fb negative is not exist')

        ###### Шаг 1. Усреднили все положительные фидбеки внутри испытуемого (между блоками 1 -6) #################
        #power
        if positive_fb.size != 0:
            positive_fb_mean = positive_fb.mean(axis = 0) # усредняем по оси runs
            positive_fb_mean = positive_fb_mean.reshape(1, 306, fr, n) # добавляем ось для фидбека
            
        else:
            positive_fb_mean = np.empty((0, 306, fr, n))
            
        #itc    
        if positive_fb_itc.size != 0:
            positive_fb_mean_itc = positive_fb_itc.mean(axis = 0) # усредняем по оси runs
            positive_fb_mean_itc = positive_fb_mean_itc.reshape(1, 306, fr, n) # добавляем ось для фидбека
            
        else:
            positive_fb_mean_itc = np.empty((0, 306, fr, n))    
        ########################## Negative feedback #############################
        negative_fb = np.empty((0, 306, fr, n))
        negative_fb_itc = np.empty((0, 306, fr, n))
        for r in rounds:
            try:
                #power
                negative = mne.time_frequency.read_tfrs('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}/power_{1}_run{2}_{3}_fb_cur_negative_{0}.h5'.format(freq_range, subj, r, t), condition=None)[0]  
                negative_data = negative.data.reshape(1, 306, fr, n) # добавляем ось для run
                           
                negative_fb = np.vstack([negative_fb, negative_data])
                
                #itc
                
                negative_itc = mne.time_frequency.read_tfrs('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}/itc_{1}_run{2}_{3}_fb_cur_negative_{0}.h5'.format(freq_range, subj, r, t), condition=None)[0]  
                negative_data_itc = negative_itc.data.reshape(1, 306, fr, n) # добавляем ось для run
                           
                negative_fb_itc = np.vstack([negative_fb_itc, negative_data_itc])
             
                
            except (OSError):
                print(f'{subj} {t} run {r} fb negative is not exist')

        ###### Шаг 1. Усреднили все отрицательные фидбеки внутри испытуемого (между блоками 1 -6) #################
        #power
        if negative_fb.size != 0:
            negative_fb_mean = negative_fb.mean(axis = 0) # усредняем по оси runs
            
            negative_fb_mean = negative_fb_mean.reshape(1, 306, fr, n) # добавляем ось для фидбека
        else:
            negative_fb_mean = np.empty((0, 306, fr, n))
            
        #itc
        if negative_fb_itc.size != 0:
            negative_fb_mean_itc = negative_fb_itc.mean(axis = 0) # усредняем по оси runs
            
            negative_fb_mean_itc = negative_fb_mean_itc.reshape(1, 306, fr, n) # добавляем ось для фидбека
        else:
            negative_fb_mean_itc = np.empty((0, 306, fr, n))        
        
        
        ####################### Шаг 2 усреднения. Усредняем данные внутри испытуемого #####################################
        
        #power
        data_into_subj = np.vstack([negative_fb_mean, positive_fb_mean])
        
        if data_into_subj.size != 0:
            temp.data = data_into_subj.mean(axis = 0) # усредняем данные для двух фидбеков внутри испытуемого (т.е. по оси фидбека)
        
            # сохраняем данные, усредненные внутри испытуемого. Шаг усредения 3, это усреднение между испытуемыми делается при рисовании топомапов
            
            temp.save('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}/ave_into_subj/power_{1}_{2}_resp.h5'.format(freq_range, subj, t))
            
        else:
            print(f'{subj} has no feedbacks in {t}')
            pass

        #itc
        data_into_subj_itc = np.vstack([negative_fb_mean_itc, positive_fb_mean_itc])
        
        if data_into_subj_itc.size != 0:
            temp.data = data_into_subj_itc.mean(axis = 0) # усредняем данные для двух фидбеков внутри испытуемого (т.е. по оси фидбека)
        
            # сохраняем данные, усредненные внутри испытуемого. Шаг усредения 3, это усреднение между испытуемыми делается при рисовании топомапов
            
            temp.save('/net/server/data/Archive/prob_learn/data_processing/TF_plots_power_itc/{0}/ave_into_subj/itc_{1}_{2}_resp.h5'.format(freq_range, subj, t))
            
        else:
            print(f'{subj} has no feedbacks in {t}')
            pass


