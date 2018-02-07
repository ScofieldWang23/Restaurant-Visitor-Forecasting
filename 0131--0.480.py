#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 15:01:22 2018

@author: wsf
"""
import operator
import glob, re
import numpy as np
import pandas as pd
from sklearn import *
from sklearn.ensemble.forest import RandomForestRegressor
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
from datetime import datetime
from xgboost import XGBRegressor
import xgboost as xgb
from keras.layers import Embedding, Input, Dense
import keras
import keras.backend as K
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest

PATH = '/Users/wangshaofei/Desktop/Kaggle/Restaurant/input/'
data = { #字典，value是一个dataframe
    'tra': pd.read_csv(PATH+'air_visit_data.csv'),      #训练集，air旗下餐厅就餐数据
    'as': pd.read_csv(PATH+'air_store_info.csv'),       #air旗下餐厅数据，一共829个记录，而测试集一共有821个餐厅
    'hs': pd.read_csv(PATH+'hpg_store_info.csv'),       #hpg旗下餐厅数据
    'ar': pd.read_csv(PATH+'air_reserve.csv'),          #air旗下餐厅预定数据
    'hr': pd.read_csv(PATH+'hpg_reserve.csv'),          #hpg旗下餐厅预定数据
    'id': pd.read_csv(PATH+'store_id_relation.csv'),    #air和hpg都注册过的餐厅的id（两列数据） 
    'tes': pd.read_csv(PATH+'sample_submission.csv'),   #测试集,air旗下的餐厅每日预测就餐人数
    'hol': pd.read_csv(PATH+'date_info.csv').rename(columns={'calendar_date':'visit_date'}) #visit_date数据（一共三列）
    }


data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id']) #筛出hpg旗下同为air旗下的餐厅

''' air_store_id和visit_date作为分组键,计算出air/hpg旗下的餐厅每日总（就餐-预定）相隔天数--rs1，就餐总人数--rv1；
    平均相隔天数（总相隔天数/不同预定天数）--rs2，平均人数（总人数/不同预定天数）--rv2
'''
    
for df in ['ar','hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_dow'] = data[df]['visit_datetime'].dt.dayofweek # 一周中的周几（Monday=0, Sunday=6）
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date # 类型转换？
   
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    # Exclude same-week reservations - from aharless kernel, 下面是新加的，为何要排除一周之内的预定
    data[df] = data[df][data[df]['reserve_datetime_diff'] > data[df]['visit_dow']] #没太懂
    
    #air_store_id和visit_date作为分组键求sum
    tmp1 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors':'rv1'})
    #air_store_id和visit_date作为分组键求mean
    tmp2 = data[df].groupby(['air_store_id','visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean().rename(columns={'visit_datetime':'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors':'rv2'})
    
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=['air_store_id','visit_date'])


data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek # 一周中的周几（Monday=0, Sunday=6）
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(lambda x: '_'.join(x.split('_')[:2]))

data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek # 一周中的周几（Monday=0, Sunday=6）
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date



unique_stores = data['tes']['air_store_id'].unique() #unique()函数用法:去除重复的air_store_id，测试集一共有821家店铺
#构建一个DataFrame，一共两列。每个店铺id对应一周7天（821*7 = 5747）
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 
                                  'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

tra = data['tra'] #6列
tra = tra.loc[tra['air_store_id'].isin(list(unique_stores))] #筛选出test集中存在的821家餐厅的就餐信息,250468行
tes = data['tes'] #32109行

######### 餐厅风格、地理位置构造特征，应该会很有用 #########
air_store = data['as'] 
air_store = air_store.loc[air_store['air_store_id'].isin(list(unique_stores))]

air_store['air_genre_name'] = air_store['air_genre_name'].map(lambda x: str(str(x).replace('/',' ')))
air_store['air_area_name'] = air_store['air_area_name'].map(lambda x: str(str(x).replace('-',' ')))

#对一些features进行LabelEncoder编码
lbl = preprocessing.LabelEncoder()

for i in range(10): #10我觉得只是随便设置的一个数（大于分割后字符串的个数即可），for循环目的是为每个字符串编码（每一列编码）
    air_store['air_genre_name'+str(i)] = lbl.fit_transform(air_store['air_genre_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
    air_store['air_area_name'+str(i)] = lbl.fit_transform(air_store['air_area_name'].map(lambda x: str(str(x).split(' ')[i]) if len(str(x).split(' '))>i else ''))
#餐厅风格、位置编码
air_store['air_genre_name'] = lbl.fit_transform(air_store['air_genre_name'])
air_store['air_area_name'] = lbl.fit_transform(air_store['air_area_name'])


tra =  pd.merge(tra, air_store, how='left', on=['air_store_id']) #tra合并air_store, 30列，方便后面计算基于air_area_name和air_genre_name的顾客数
tes =  pd.merge(tes, air_store, how='left', on=['air_store_id']) #tes合并air_store, 31列，方便后面计算基于air_area_name和air_genre_name的顾客数

# NEW FEATURES FROM JMBULL
tra['date_int'] = tra['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int) #这个特征效果很好 ！！！！
tes['date_int'] = tes['visit_date'].apply(lambda x: x.strftime('%Y%m%d')).astype(int) #这个特征效果很好 ！！！！

Day_tra = tra['date_int'].unique().size #跨度478天，478-39 = 439天 = 409+30
print(Day_tra)
Day_tes = tes['date_int'].unique().size #跨度39天
print(Day_tes) 

'''
temp = tra.loc[tra['date_int']>=20160401]
temp = temp.loc[temp['date_int']<=20160531]
A = list(temp['air_store_id'].unique()) #训练集中在测试集时间对应的时间段只有311家餐厅在营业
A1 = []
for i in range(7):
    A1.append(tra.loc[tra['dow']==i]['air_store_id'].unique().size)
A2 = []   
for i in range(12):
    A2.append(tra.loc[tra['month']==i+1]['air_store_id'].unique().size)
'''


#sure it can be compressed...
######### 统计每家餐厅一周7天内每天不同的顾客数据 #########

#每家餐厅周一到周六每日最少顾客数量
tmp = tra.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_store_id', 'dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_store_id', 'dow'])

#每家餐厅周一到周六每日顾客数量的平均数
tmp = tra.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_store_id', 'dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_store_id', 'dow'])

#每家餐厅周一到周六每日最少顾客数量的平均数
tmp = tra.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_store_id', 'dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_store_id', 'dow'])

#每家餐厅周一到周六每日最多顾客数量
tmp = tra.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_store_id', 'dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_store_id', 'dow'])

#每家餐厅周一到周六每日一共有多少记录（多少个Sunday,Monday...）
tmp = tra.groupby(['air_store_id', 'dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
tra = pd.merge(tra, tmp, how='left', on=['air_store_id', 'dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_store_id', 'dow'])



############### 每种air_area_name每天不同的顾客数据 ###############

#每家餐厅周一到周六每日最少顾客数量
tmp = tra.groupby(['air_area_name','month','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'area_min_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_area_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_area_name','month','dow'])

#每家餐厅周一到周六每日顾客数量的平均数
tmp = tra.groupby(['air_area_name','month','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'area_mean_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_area_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_area_name','month','dow'])

#每家餐厅周一到周六每日最少顾客数量的平均数
tmp = tra.groupby(['air_area_name','month','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'area_median_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_area_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_area_name','month','dow'])

#每家餐厅周一到周六每日最多顾客数量
tmp = tra.groupby(['air_area_name','month','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'area_max_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_area_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_area_name','month','dow'])

#每家餐厅周一到周六每日一共有多少记录（多少个Sunday,Monday...）
tmp = tra.groupby(['air_area_name','month','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'area_count_observations'})
tra = pd.merge(tra, tmp, how='left', on=['air_area_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_area_name','month','dow'])


############### 统计一周7天内每种air_genre_name每天不同的顾客数据 ############### 

#每家餐厅周一到周六每日最少顾客数量
tmp = tra.groupby(['air_genre_name','month','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'genre_min_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_genre_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_genre_name','month','dow'])

#每家餐厅周一到周六每日顾客数量的平均数
tmp = tra.groupby(['air_genre_name','month','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'genre_mean_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_genre_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_genre_name','month','dow'])

#每家餐厅周一到周六每日最少顾客数量的平均数
tmp = tra.groupby(['air_genre_name','month','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'genre_median_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_genre_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_genre_name','month','dow'])

#每家餐厅周一到周六每日最多顾客数量
tmp = tra.groupby(['air_genre_name','month','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'genre_max_visitors'})
tra = pd.merge(tra, tmp, how='left', on=['air_genre_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_genre_name','month','dow'])

#每家餐厅周一到周六每日一共有多少记录（多少个Sunday,Monday...）
tmp = tra.groupby(['air_genre_name','month','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'genre_count_observations'})
tra = pd.merge(tra, tmp, how='left', on=['air_genre_name','month','dow'])
tes = pd.merge(tes, tmp, how='left', on=['air_genre_name','month','dow'])

'''
######## 统计每个月不同的顾客数据，然后统一将新特征添加到stores#########
#每家餐厅每月最少顾客数量
tmp = tra.groupby(['air_store_id','month'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id']) 
#每家餐厅每月顾客数量的平均数
tmp = tra.groupby(['air_store_id','month'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','month'])
#每家餐厅每月最少顾客数量的平均数
tmp = tra.groupby(['air_store_id','month'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','month'])
#每家餐厅每月最多顾客数量
tmp = tra.groupby(['air_store_id','month'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','month'])
#每家餐厅每月一共有多少记录（多少个Sunday,Monday...）
tmp = tra.groupby(['air_store_id','month'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
stores = pd.merge(stores, tmp, how='left', on=['air_store_id','month']) 
'''


#### 基本日期数据构造特征 ####
data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week']) #注意：这里day_of_week和之前的dow编码不一样，但不知道这个后面有用？？
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
# 一个疑问：后面都对holiday_flag做了特殊处理，这儿不需要？？


##################### 训练集，测试集构造 #####################
#合并基本日期数据信息
train = pd.merge(tra, data['hol'], how='left', on=['visit_date']) #252108行
test = pd.merge(tes, data['hol'], how='left', on=['visit_date']) #32019行 


#### 合并air、hpg旗下餐厅预定数据（rs1, rv1; rs2, rv2）###

for df in ['ar','hr']:
    #下面合并会产生很多空值
    train = pd.merge(train, data[df], how='left', on=['air_store_id','visit_date']) 
    test = pd.merge(test, data[df], how='left', on=['air_store_id','visit_date'])
    
#‘id’ is the air_store_id and visit_date with an underscore
train['id'] = train.apply(lambda r: '_'.join([str(r['air_store_id']), str(r['visit_date'])]), axis=1) #题目要求，测试集就有一列'id'是由air_store_id和visit_date组成的
train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y'] #rv1_x和rv1_y应该是因为有两列一样的特征存在，所以自动生成的；下面同理
train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2 #rs比较特殊，具体含义见上面'ar','hr'构造

test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2


#下面特征效果很好？？居然在经纬度做文章
train['var_max_lat'] = train['latitude'].max() - train['latitude']
train['var_max_long'] = train['longitude'].max() - train['longitude']
test['var_max_lat'] = test['latitude'].max() - test['latitude']
test['var_max_long'] = test['longitude'].max() - test['longitude']


train['lon_plus_lat'] = train['longitude'] + train['latitude'] 
test['lon_plus_lat'] = test['longitude'] + test['latitude']


#air_store_id编码：0~828 = air_store_id2
lbl = preprocessing.LabelEncoder()
train['air_store_id2'] = lbl.fit_transform(train['air_store_id']) #62+2列
test['air_store_id2'] = lbl.transform(test['air_store_id'])       #62+2列

col = [c for c in train if c not in ['id', 'air_store_id', 'visit_date','visitors', #去掉id是因为不好回归？？
                                     'air_area_name7', 'air_area_name8', 'air_area_name9',
                                     'air_genre_name3', 'air_genre_name4','air_genre_name5', 'air_genre_name6', 'air_genre_name7', 'air_genre_name8', 'air_genre_name9']] #除去列表中4项，一共50列的列名，后面模型训练要用到

train = train.fillna(-1)
test = test.fillna(-1)

#train[col].to_csv('train.csv', index=False)

train.sort_values(by=['visit_date'],inplace=True)
train.reset_index(inplace=True,drop=True)
test.sort_values(by=['visit_date'],inplace=True)
test.reset_index(inplace=True,drop=True)



######################### 模型建立，模型训练，模型调参 #########################

'''
Following function implements the Keras neural network model.

Basic structure:

(1) categorical columns get independent inputs, passed through embedding layer and then flattened.
(2) numeric columns are simply taken as float32 inputs
(3) the final tensors of categorical and numerical are then concatenated together
(4) following the concatenated layer and simple feed forward neural network is implemented. #前馈神经网络
(5) output layer has 'ReLU' activation function #激活函数选择：修正线性单元，对于某一输入，当它小于 0 时，输出为 0，否则不变

'''

def get_nn_complete_model(train, hidden1_neurons=35, hidden2_neurons=15): #35，15是怎么设置的？
    """
    Input:
        train:           train dataframe(used to define the input size of the embedding layer)
        hidden1_neurons: number of neurons in the first hidden layer
        hidden2_neurons: number of neurons in the first hidden layer 注：我觉得应该是第二层的神经元数
    Output:
        return 'keras neural network model'
    """
    #hidden1_neurons=35
    #hidden2_neurons=15
    
    K.clear_session()#清空之前的模型，应该是初始化
    
    #整体过程（对每一个特征来说）Input-> embedding-> Flatten-> (concatenate)->Dense
    
   
    #air_store_id_emb
    air_store_id = Input(shape=(1,), dtype='int32', name='air_store_id')#expect输入向量一维
    air_store_id_emb = Embedding(len(train['air_store_id2'].unique()) + 1, 15, input_shape=(1,),
                                 name='air_store_id_emb')(air_store_id)#前一层的输出就是后一层的输入
    #Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小。
    air_store_id_emb = keras.layers.Flatten(name='air_store_id_emb_flatten')(air_store_id_emb)
   

    #dow_emb
    dow = Input(shape=(1,), dtype='int32', name='dow')
    dow_emb = Embedding(8, 3, input_shape=(1,), name='dow_emb')(dow)#embedding的第一个参数是输入个数（维度），第二个是输出个数
    dow_emb = keras.layers.Flatten(name='dow_emb_flatten')(dow_emb)

        
    #month_emb
    month = Input(shape=(1,), dtype='int32', name='month')
    month_emb = Embedding(13, 3, input_shape=(1,), name='month_emb')(month)
    month_emb = keras.layers.Flatten(name='month_emb_flatten')(month_emb)


    #air_area_name, air_genre_name
    air_area_name, air_genre_name = [], []
    air_area_name_emb, air_genre_name_emb = [], []
    for i in range(7):#air_area_name只有0-6是有大于1的，虽然train里面有air_area_name7-air_area_name9但是全是0，我也不知道为什么
        area_name_col = 'air_area_name' + str(i)
        air_area_name.append(Input(shape=(1,), dtype='int32', name=area_name_col))
        tmp = Embedding(len(train[area_name_col].unique()), 3, input_shape=(1,),
                        name=area_name_col + '_emb')(air_area_name[-1])#每次将最后一个（就是刚加进去的那个）特征加入网络
        tmp = keras.layers.Flatten(name=area_name_col + '_emb_flatten')(tmp)
        air_area_name_emb.append(tmp)

        if i > 4:
            continue#而air_genre_name只有0-3有数字的，所以超过四就不要后面了直接continue
        area_genre_col = 'air_genre_name' + str(i)
        air_genre_name.append(Input(shape=(1,), dtype='int32', name=area_genre_col))
        tmp = Embedding(len(train[area_genre_col].unique()), 3, input_shape=(1,),
                        name=area_genre_col + '_emb')(air_genre_name[-1])
        tmp = keras.layers.Flatten(name=area_genre_col + '_emb_flatten')(tmp)
        air_genre_name_emb.append(tmp)#循环结束，应该是完成了air_area_name, air_genre_name有数据的特征们的几层的构建
    
    #dense后代表输出维度为4 ,Dense层就是全连接层，所有的卷积层之后都要加上全连接层，全连接层的每一个节点都与上一层每个节点连接，是把前一层的输出特征都综合起来，下面是全连接层的链接
    #https://www.zhihu.com/question/41037974
    #concatenate：该Merge层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。（大概就是合并吧）
    air_genre_name_emb = keras.layers.concatenate(air_genre_name_emb)
    air_genre_name_emb = Dense(4, activation='sigmoid', name='final_air_genre_emb')(air_genre_name_emb)

    air_area_name_emb = keras.layers.concatenate(air_area_name_emb)
    air_area_name_emb = Dense(4, activation='sigmoid', name='final_air_area_emb')(air_area_name_emb)
    
    air_area_code = Input(shape=(1,), dtype='int32', name='air_area_code')
    air_area_code_emb = Embedding(len(train['air_area_name'].unique()), 8, input_shape=(1,), name='air_area_code_emb')(air_area_code)
    air_area_code_emb = keras.layers.Flatten(name='air_area_code_emb_flatten')(air_area_code_emb)
    
    air_genre_code = Input(shape=(1,), dtype='int32', name='air_genre_code')
    air_genre_code_emb = Embedding(len(train['air_genre_name'].unique()), 5, input_shape=(1,),
                                   name='air_genre_code_emb')(air_genre_code)
    air_genre_code_emb = keras.layers.Flatten(name='air_genre_code_emb_flatten')(air_genre_code_emb)
    
    
    
    #以下一堆输入，最后放在inp_ten
    holiday_flg = Input(shape=(1,), dtype='float32', name='holiday_flg')
    year = Input(shape=(1,), dtype='float32', name='year')
    
    min_visitors = Input(shape=(1,), dtype='float32', name='min_visitors')
    mean_visitors = Input(shape=(1,), dtype='float32', name='mean_visitors')
    median_visitors = Input(shape=(1,), dtype='float32', name='median_visitors')
    max_visitors = Input(shape=(1,), dtype='float32', name='max_visitors')
    count_observations = Input(shape=(1,), dtype='float32', name='count_observations')
    
    area_min_visitors = Input(shape=(1,), dtype='float32', name='area_min_visitors')
    area_mean_visitors = Input(shape=(1,), dtype='float32', name='area_mean_visitors')
    area_median_visitors = Input(shape=(1,), dtype='float32', name='area_median_visitors')
    area_max_visitors = Input(shape=(1,), dtype='float32', name='area_max_visitors')
    area_count_observations = Input(shape=(1,), dtype='float32', name='area_count_observations')
    
    genre_min_visitors = Input(shape=(1,), dtype='float32', name='genre_min_visitors')
    genre_mean_visitors = Input(shape=(1,), dtype='float32', name='genre_mean_visitors')
    genre_median_visitors = Input(shape=(1,), dtype='float32', name='genre_median_visitors')
    genre_max_visitors = Input(shape=(1,), dtype='float32', name='genre_max_visitors')
    genre_count_observations = Input(shape=(1,), dtype='float32', name='genre_count_observations')
    
    
    rs1_x = Input(shape=(1,), dtype='float32', name='rs1_x')
    rv1_x = Input(shape=(1,), dtype='float32', name='rv1_x')
    rs2_x = Input(shape=(1,), dtype='float32', name='rs2_x')
    rv2_x = Input(shape=(1,), dtype='float32', name='rv2_x')
    rs1_y = Input(shape=(1,), dtype='float32', name='rs1_y')
    rv1_y = Input(shape=(1,), dtype='float32', name='rv1_y')
    rs2_y = Input(shape=(1,), dtype='float32', name='rs2_y')
    rv2_y = Input(shape=(1,), dtype='float32', name='rv2_y')
    total_reserv_sum = Input(shape=(1,), dtype='float32', name='total_reserv_sum')
    total_reserv_mean = Input(shape=(1,), dtype='float32', name='total_reserv_mean')
    total_reserv_dt_diff_mean = Input(shape=(1,), dtype='float32', name='total_reserv_dt_diff_mean')
    date_int = Input(shape=(1,), dtype='float32', name='date_int')
    var_max_lat = Input(shape=(1,), dtype='float32', name='var_max_lat')
    var_max_long = Input(shape=(1,), dtype='float32', name='var_max_long')
    #lon_plus_lat = Input(shape=(1,), dtype='float32', name='lon_plus_lat')
    
    #对[dow_emb, month_emb, year, holiday_flg]做个结合成为data_emb
    date_emb = keras.layers.concatenate([dow_emb, month_emb, year, holiday_flg])
    date_emb = Dense(5, activation='sigmoid', name='date_merged_emb')(date_emb)

    cat_layer = keras.layers.concatenate([holiday_flg, min_visitors, mean_visitors,median_visitors, max_visitors, count_observations, 
                    area_min_visitors, area_mean_visitors,area_median_visitors, area_max_visitors, area_count_observations,
                    genre_min_visitors, genre_mean_visitors,genre_median_visitors, genre_max_visitors, genre_count_observations,
                    rs1_x, rv1_x,rs2_x, rv2_x, rs1_y, rv1_y, rs2_y, rv2_y,
                    total_reserv_sum, total_reserv_mean, total_reserv_dt_diff_mean,
                    date_int, var_max_lat, var_max_long, 
                    date_emb, air_area_name_emb, air_genre_name_emb,
                    air_area_code_emb, air_genre_code_emb, air_store_id_emb])
    #m的输出维度为hidden1_neurons = 15
    m = Dense(hidden1_neurons, name='hidden1',
             kernel_initializer=keras.initializers.RandomNormal(mean=0.0,
                            stddev=0.05, seed=None))(cat_layer)
    m = keras.layers.LeakyReLU(alpha=0.2)(m)#LeakyReLU：避免ReLU可能出现的神经元“死亡”现象。。。不懂
    m = keras.layers.BatchNormalization()(m)#即使得其输出数据的均值接近0，其标准差接近1
    #m1的输出维度为hidden2_neurons = 35
    m1 = Dense(hidden2_neurons, name='hidden2')(m)
    m1 = keras.layers.LeakyReLU(alpha=0.2)(m1)
    m = Dense(1, activation='relu')(m1)#输出维度为1，激活函数是relu（不知道什么意思哈哈）
    #inp_ten是模型输入的几个特征
    inp_ten = [
        holiday_flg, min_visitors, mean_visitors, median_visitors, max_visitors, count_observations,
        area_min_visitors, area_mean_visitors,area_median_visitors, area_max_visitors, area_count_observations,
        genre_min_visitors, genre_mean_visitors,genre_median_visitors, genre_max_visitors, genre_count_observations,
        rs1_x, rv1_x, rs2_x, rv2_x, rs1_y, rv1_y, rs2_y, rv2_y, total_reserv_sum, total_reserv_mean,
        total_reserv_dt_diff_mean, date_int, var_max_lat, var_max_long,
        dow, year, month, air_store_id, air_area_code, air_genre_code
    ]
    inp_ten += air_area_name
    inp_ten += air_genre_name
    #函数式模型，例 model = Model(inputs=[a1, a2], outputs=[b1, b3, b3])，拥有输入inp_ten和输出m
    model = keras.Model(inp_ten, m)
    #编译模型 参数：损失函数loss，优化器optimizer，评估指标metrics
    model.compile(loss='mse', optimizer='rmsprop', metrics=['acc'])

    return model


############## 神经网络输入数据准备 ##############
'''
Here we prepare data required for the neural network model.

value_col: taken as float input(which are normalized)

nn_col - value_col: taken as categorical inputs(embedding layers used) #嵌入层将正整数（下标）转换为具有固定大小的向量

'''
#直觉上效果不太好的特征有：count_observations，rs1,rs2 
value_col = ['holiday_flg','min_visitors','mean_visitors','median_visitors','max_visitors','count_observations',
             'area_min_visitors','area_mean_visitors','area_median_visitors','area_max_visitors','area_count_observations',
             'genre_min_visitors','genre_mean_visitors','genre_median_visitors','genre_max_visitors','genre_count_observations',
             'rs1_x','rv1_x','rs2_x','rv2_x','rs1_y','rv1_y','rs2_y','rv2_y','total_reserv_sum','total_reserv_mean',
             'total_reserv_dt_diff_mean','date_int','var_max_lat','var_max_long']

#nn_col一共39列
nn_col = value_col + ['dow', 'year', 'month', 'air_store_id2', 'air_area_name', 'air_genre_name',
                      'air_area_name0', 'air_area_name1', 'air_area_name2', 'air_area_name3', 'air_area_name4',
                      'air_area_name5', 'air_area_name6', 'air_genre_name0', 'air_genre_name1','air_genre_name2','air_genre_name3','air_genre_name4']

########### local_test--keras神经网络输入数据准备 ###########
# 我在想这样划分有没有问题，因为没有考虑holiday的分布
train_val = train.tail(int(len(train)*0.1)) #用于测试的数据集
train_tra = train.head(len(train)- int(len(train)*0.1)) #用于训练的数据集

local_X = train_tra.copy()
local_X_test = train_val[nn_col].copy()

value_scaler = preprocessing.MinMaxScaler() #最小最大值标准化：将数据缩放至给定的最小值与最大值之间，通常是0与１之间
for vcol in value_col:
    local_X[vcol] = value_scaler.fit_transform(local_X[vcol].values.astype(np.float64).reshape(-1, 1)) #reshape(-1,1)可以让原矩阵变成只有一列，行数不知道多少的矩阵
    local_X_test[vcol] = value_scaler.transform(local_X_test[vcol].values.astype(np.float64).reshape(-1, 1))

local_X_train = list(local_X[nn_col].T.as_matrix()) #as_matrix() 用来dataframe和numpy矩阵转换
local_Y_train = np.log1p(local_X['visitors']).values
local_nn_train = [local_X_train, local_Y_train]
local_nn_test = [list(local_X_test[nn_col].T.as_matrix())]
print("Local Train and test data prepared")



########### online_test--keras神经网络输入数据准备 ###########

X = train.copy()
X_test = test[nn_col].copy()

value_scaler = preprocessing.MinMaxScaler() #最小最大值标准化：将数据缩放至给定的最小值与最大值之间，通常是0与１之间
for vcol in value_col:
    X[vcol] = value_scaler.fit_transform(X[vcol].values.astype(np.float64).reshape(-1, 1)) #reshape(-1,1)可以让原矩阵变成只有一列，行数不知道多少的矩阵
    X_test[vcol] = value_scaler.transform(X_test[vcol].values.astype(np.float64).reshape(-1, 1))

X_train = list(X[nn_col].T.as_matrix()) #as_matrix() 用来dataframe和numpy矩阵转换
Y_train = np.log1p(X['visitors']).values
nn_train = [X_train, Y_train]
nn_test = [list(X_test[nn_col].T.as_matrix())]
print("Train and test data prepared")


#评价指标
def RMSLE(y, pred):
    return 'evaluation', metrics.mean_squared_error(y, pred)**0.5


################################# 模型训练 #################################

################### 本地评测 -- local_test ###################

#local_model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3,
                    #n_estimators=200, subsample=0.8, max_depth =10)
local_model2 = XGBRegressor(learning_rate=0.2, n_estimators=200, subsample=0.8, 
                      colsample_bytree=0.8, max_depth =10,feval = RMSLE)

#local_model3 = RandomForestRegressor(random_state=10)
#local_model4 = linear_model.LinearRegression()
#local_model5 = linear_model.Lasso(alpha=0.1)

local_model6 = get_nn_complete_model(train_tra, hidden2_neurons=12) #15

#local_model1.fit(train_tra[col], np.log1p(train_tra['visitors'].values))
#print("local_Model1 trained")
local_model2.fit(train_tra[col], np.log1p(train_tra['visitors'].values))
print("local_Model2 trained")

local_model3.fit(train_tra[col], np.log1p(train_tra['visitors'].values))
print("local_Model3 trained")
'''
local_model4.fit(train_tra[col], np.log1p(train_tra['visitors'].values))
print("local_Model4 trained")
local_model5.fit(train_tra[col], np.log1p(train_tra['visitors'].values))
print("local_Model5 trained")
'''
for i in range(5):
    local_model6.fit(local_nn_train[0], local_nn_train[1], epochs=3, verbose=1,
        batch_size=256, shuffle=True, validation_split=0.15)
    local_model6.fit(local_nn_train[0], local_nn_train[1], epochs=8, verbose=0,
        batch_size=256, shuffle=True)
    local_model6.fit(local_nn_train[0], local_nn_train[1], epochs=2, verbose=0,
        batch_size=512, shuffle=True)
print("local_Model6 trained")


#local_preds1 = local_model1.predict(train_val[col])
local_preds2 = local_model2.predict(train_val[col])

local_preds3 = local_model3.predict(train_val[col])
'''
local_preds4 = local_model4.predict(train_val[col])
local_preds5 = local_model5.predict(train_val[col])
'''
# .clip(0, 6.8) used to avoid random high values that might occur
local_preds6 = pd.Series(local_model6.predict(local_nn_test[0]).reshape(-1)).clip(0, 6.8).values

#print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train_val['visitors'].values), local_preds1)) #0.5771371713831291，训练时间很长，我觉得可以把这个模型去掉
print('RMSE XGBRegressor: ', RMSLE(np.log1p(train_val['visitors'].values), local_preds2)) #0.5102906350772146

print('RMSE RFRegressor: ', RMSLE(np.log1p(train_val['visitors'].values), local_preds3)) #0.5653453188639355
'''
print('RMSE LinearRegression: ', RMSLE(np.log1p(train_val['visitors'].values), local_preds4)) #0.565152580038004
print('RMSE Lasso: ', RMSLE(np.log1p(train_val['visitors'].values), local_preds5)) #0.571733131530236
'''
print('RMSE NeuralNetwork: ', RMSLE(np.log1p(train_val['visitors'].values), local_preds6)) #0.503899939431929


#local_preds = 0.1*local_preds1+0.3*local_preds2+0.1*local_preds3+0.1*local_preds4+0.1*local_preds5+0.3*local_preds6
local_preds = 0.5*local_preds2+0.5*local_preds3
print('综合加权：', RMSLE(np.log1p(train_val['visitors'].values), local_preds)) # 0.4996173134624456


##################feature_importance_picture########################
#XGB
importance = local_model2.booster().get_score(importance_type='weight')
importance = sorted(importance.items(), key=operator.itemgetter(1))  
  
fea_importance = pd.DataFrame(importance, columns=['feature', 'fscore'])  
fea_importance['fscore'] = fea_importance['fscore'] / fea_importance['fscore'].sum()  
fea_importance.to_csv("fea_importance.csv", index=False)  
   
  
plt.figure()
fea_importance.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')   
#fig.savefig(r'/Users/wangshaofei/Desktop/XGB_importance.png',dpi=300)

col = list(fea_importance.iloc[12:,0]) #一共38个特征
col = list(fea_importance.iloc[27:,0]) #一共23个特征




################### 线上评测 -- online_test ###################

slct = SelectKBest(k="all")  
slct.fit(train[col], train['visitors'])
scores = slct.scores_ 
KBest_fea = pd.DataFrame()
KBest_fea['feature'] = col
KBest_fea['score'] = scores
KBest_fea.sort_values(['score'],inplace=True)
plt.figure()
KBest_fea.plot(kind='barh', x='feature', y='score', legend=False, figsize=(6, 10))  
plt.title('KBest Feature Importance')  
plt.xlabel('relative importance')   



#model1 = ensemble.GradientBoostingRegressor(learning_rate=0.2, random_state=3,
                    #n_estimators=200, subsample=0.8, max_depth =10)
model2 = XGBRegressor(learning_rate=0.1, n_estimators=200, subsample=0.8, 
                      colsample_bytree=0.8, max_depth =8) #10
'''
model3 = RandomForestRegressor(oob_score=True, random_state=10)
model4 = linear_model.LinearRegression()
model5 = linear_model.Lasso(alpha=0.1)
'''
model6 = get_nn_complete_model(train, hidden2_neurons=12) #15

 
#model1.fit(train[col], np.log1p(train['visitors'].values))
#print("Model1 trained")

model2.fit(train[col], np.log1p(train['visitors'].values), eval_metric=RMSLE)
print("Model2 trained")
'''
model3.fit(train[col], np.log1p(train['visitors'].values))
print("Model3 trained")
model4.fit(train[col], np.log1p(train['visitors'].values))
print("Model4 trained")
model5.fit(train[col], np.log1p(train['visitors'].values))
print("Model5 trained")
'''
for i in range(5):
    model6.fit(nn_train[0], nn_train[1], epochs=3, verbose=1,
        batch_size=256, shuffle=True, validation_split=0.15)
    model6.fit(nn_train[0], nn_train[1], epochs=8, verbose=0,
        batch_size=256, shuffle=True)
    model6.fit(nn_train[0], nn_train[1], epochs=2, verbose=0,
        batch_size=512, shuffle=True)
print("Model6 trained")



######## 用全部训练集当做测试集预测 ########
preds1 = model1.predict(train[col])

preds2 = model2.predict(train[col])
preds3 = model3.predict(train[col])
preds4 = model4.predict(train[col])
preds5 = model5.predict(train[col])

# .clip(0, 6.8) used to avoid random high values that might occur
preds6 = pd.Series(model6.predict(nn_train[0]).reshape(-1)).clip(0, 6.8).values

print('RMSE GradientBoostingRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds1)) #0.3525828248968016，是不是过拟合了？？

print('RMSE XGBRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds2)) #0.4042822982431063

print('RMSE RFRegressor: ', RMSLE(np.log1p(train['visitors'].values), preds3)) #0.22503146839709723
print('RMSE LinearRegression: ', RMSLE(np.log1p(train['visitors'].values), preds4)) #0.551629590320174

print('RMSE Lasso: ', RMSLE(np.log1p(train['visitors'].values), preds5)) #0.5562617563069968
print('RMSE NeuralNetwork: ', RMSLE(np.log1p(train['visitors'].values), preds6)) #0.4935254035648056

preds = 0.1*preds1+0.3*preds2+0.1*preds3+0.1*preds4+0.1*preds5+0.3*preds6 #0.4149233406078539
preds = 0.3*preds2+0.2*preds3+0.1*preds4+0.1*preds5+0.3*preds6 #0.4006411704976203
preds = 0.3*preds2+0.3*preds3+0.1*preds5+0.3*preds6 #0.3684834010002285
preds = 0.5*preds2+0.5*preds6
print('综合加权：', RMSLE(np.log1p(train['visitors'].values), preds)) 


########## online测试集，预测visitors的值 ##########
#online_preds1 = model1.predict(test[col])

online_preds2 = model2.predict(test[col])
'''
online_preds3 = model3.predict(test[col])
online_preds4 = model4.predict(test[col])
online_preds5 = model5.predict(test[col])
'''
online_preds6 = pd.Series(model6.predict(nn_test[0]).reshape(-1)).clip(0, 6.8).values

#test['visitors'] = 0.3*preds1+0.3*preds2+0.4*preds3 #这个权重很迷啊，怎么得来的
#test['visitors'] = 0.3*online_preds2+0.1*online_preds3+0.2*online_preds4+0.1*online_preds5+0.3*online_preds6 #0.491, 0.486

test['visitors'] = 0.5*online_preds2+0.5*online_preds6  #0.482, 0.480
test['visitors'] = np.expm1(test['visitors']).clip(lower=0.)
sub1 = test[['id','visitors']].copy()
sub1.to_csv('sub1.csv', index=False) #sub1.csv提交上去是
print("Model predictions done.")



#################### 模型优化 ####################
# from hklee
# https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st/code
''' In the following, a small modification from the original kernal is made, 
    we only get the visitors values from history only if 
    they match ['air_store_id', 'day_of_week', 'holiday_flg'] or ['air_store_id', 'day_of_week'] columns match
'''

date_info = pd.read_csv(PATH+'date_info.csv')
air_visit_data = pd.read_csv(PATH+'air_visit_data.csv')
sample_submission = pd.read_csv(PATH+'sample_submission.csv')


date_info = data['hol'].copy()
air_visit_data = tra.copy()


#在周末的holidays应该不应该特殊处理？holiday_flg应该等于0
wkend_holidays = date_info.apply(
    (lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') and x.holiday_flg==1), axis=1)
date_info.loc[wkend_holidays, 'holiday_flg'] = 0

#这个权重有何用？add decreasing weights from now —— 时间越近，权重越高！厉害啊
date_info['weight'] = ((date_info.index + 1) / len(date_info)) ** 5 #为什么要用结果的5次方，是为了拉开权重差距么？
 
# I've consumed all my submissions for day. I suggest to try weight 1/x, exp(-x).
# It seems to me that Recruit sales may be stochastic and depends only on recent means.

#weighted mean visitors for each (air_store_id, day_of_week, holiday_flag) or (air_store_id, day_of_week)
visit_data = air_visit_data.merge(date_info, left_on='visit_date', right_on='visit_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True) #visit_date和calendar_date是一样的
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p) #顾客数取log

wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() ) #求加权平均数
#每个餐厅每周一 ~ 周日的每天加权顾客数
visitors = visit_data.groupby(['air_store_id', 'month', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) # 一共4列'air_store_id','day_of_week','holiday_flg','visitors'

sample_submission['air_store_id'] = sample_submission.id.map(lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True) #去掉submission.csv中原来的'visitors'这一列

sample_submission = sample_submission.merge(date_info, left_on='calendar_date',right_on='visit_date', how='left')
sample_submission.drop('calendar_date', axis=1, inplace=True)
#合并visitors，求得人数
sample_submission = sample_submission.merge(visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')
sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1) #把之前的对数还原

sub2 = sample_submission[['id', 'visitors']].copy()
sub2 = sub2.fillna(-1) #为空值，一共8878项为nan,为何会有

'''
sub3 = sub1.copy()
sub3['visitors'] = sub1['visitors']-sub2['visitors'] #32019个
n = sub3[sub3.visitors > 0] #14920个，意味着sub2预测偏大
'''

'''
col = ['air_store_id','calendar_date','day_of_week','holiday_flg','visitors']
hol_visitors = sample_submission[col].fillna(-1)
#筛选出哪些餐厅在哪些日期顾客是空值
nan_visitors = hol_visitors[hol_visitors.visitors == -1] #一共668条记录，空值如何处理？

tes_stores = sample_submission['air_store_id'].unique() #测试集一共821家餐厅
tra_stores = visitors['air_store_id'].unique() #训练集一共829家餐厅
'''


############ 综合sub1, sub2求得最后预测人数 ############
def final_visitors(x, alt=False):
    visitors_x, visitors_y = x['visitors_x'], x['visitors_y'] #在两个相同的列名加_x, _y是自动的么，神奇？
    if x['visitors_y'] == -1:
        return visitors_x
    else:
        return 0.7*visitors_x + 0.3*visitors_y* 1.1 #为何要*1.1


sub_merge = pd.merge(sub1, sub2, on='id', how='inner')
sub_merge['visitors'] = sub_merge.apply(lambda x: final_visitors(x), axis=1)
print("Done")
sub_merge[['id', 'visitors']].to_csv('submission.csv', index=False) #0.481。。。























