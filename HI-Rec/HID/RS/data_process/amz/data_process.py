#-*- coding : utf-8-*-
# coding:unicode_escape

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import re
import pickle
import os

from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

"""
数据预处理
对原始电影数据，user数据进行处理
"""
def user_data_processing(path):
    '''
    对原始user数据进行处理
    UserID：保持不变
    JobID：保持不变
    Gender字段：需要将‘F’和‘M’转换成0和1。
    Age字段：要转成7个连续数字0~6。
    舍弃： zip-code
    '''
    
    print('user_data_processing....')
    user_title = ['UserID']
    print(path)
    users = pd.read_table(os.path.join(path, 'user.dat'), sep='::', header=None,
        names=user_title, engine='python')
    users = users.filter(regex='UserID')
    users_orig = users.values #a list

    # gender_to_int = {'F':0,'M':1}
    # users['Gender'] = users['Gender'].map(gender_to_int)
    # age2int = {val:ii for ii, val in enumerate(set(users['Age']))}
    # users['Age'] = users['Age'].map(age2int)

    return users, users_orig

def movie_data_processing(path, title_length = 16):
    '''
    对原始movie数据不作处理
    Genres字段：进行int映射，因为有些电影是多个Genres的组合,需要再将每个电影的Genres字段转成数字列表.
    Title字段：首先去除掉title中的year。然后将title映射成数字列表。（int映射粒度为单词而不是整个title）
    Genres和Title字段需要将长度统一，这样在神经网络中方便处理。
    空白部分用‘< PAD >’对应的数字填充。
    '''
    print('movie_data_processing....')
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table(os.path.join(path, 'item.dat'), sep='::', encoding='ISO-8859-1',
        header=None, names=movies_title, engine='python')
    movies_orig = movies.values#length:3883
    genres_set = set()
    for val in movies['Genres']:
        genres_set.add(val)
    genres2int = {val:ii for ii, val in enumerate(genres_set)}
    movies['Genres'] = movies['Genres'].map(genres2int)

    titel_set = set()
    for val in movies['Title']:
        titel_set.add(val)
    title2int = {val:ii for ii, val in enumerate(titel_set)}
    movies['Title'] = movies['Title'].map(title2int)
    
    
    return movies, movies_orig, genres2int, title2int


def rating_data_processing(path):
    '''
    rating数据处理，只需要将timestamps舍去，保留其他属性即可
    '''
    print('rating_data_processing....')
    ratings_title = ['UserID', 'MovieID', 'ratings']
    ratings = pd.read_table(os.path.join(path, 'rating.dat'), sep='::',
        header=None, names=ratings_title, engine='python')
    # 保留每个UserID的前10条评分记录
    ratings = ratings.sort_values(by=['MovieID']).groupby('MovieID').head(20)

    # 将同一电影的ratings进行求和平均并赋值给各个电影
    ratings_mean = ratings.groupby('MovieID')['ratings'].mean().astype('int')
    ratings_counts = ratings.groupby('MovieID')['ratings'].size()
    # print(ratings_counts)
    # print('-------------------------------------')
    # 将评论数据进行分桶, 分为5个等级
    ratings_counts_max = max(ratings_counts)
    # print(ratings_counts_max)
    cut_num = int(ratings_counts_max / 5) + 1
    cut_range = []
    for i in range(5 + 1):
        cut_range.append(i * cut_num)
    # print(cut_range)
    ratings_counts = pd.cut(ratings_counts, bins=cut_range, labels=False)
    # print(ratings_counts)

    if len(ratings_mean) != len(ratings_counts):
        print('total_ratings is not equal ratings_counts!')
    else:
        ratings = pd.merge(pd.merge(ratings, ratings_counts, on='MovieID'), ratings_mean, on='MovieID')
        # rename the columns
        # ratings_x: 原ratings
        # ratings_y: ratings_counts
        # ratings: ratings_mean
        ratings = ratings.rename(columns={'ratings': 'ratings_mean'}).rename(columns={'ratings_x': 'ratings'}).rename(columns={'ratings_y': 'ratings_count'})
        ratings = ratings.filter(regex='UserID|MovieID|ratings_mean|ratings_count|ratings')

    rating_datatype = []
    rating_datatype.append({'name': 'ratings_count', 'len': ratings['ratings_count'].max() + 1,
                           'ori_scaler': cut_range,
                           'type': 'LabelEncoder', 'nan_value': None})
    rating_datatype.append({'name': 'ratings_mean', 'len': ratings['ratings_mean'].max() + 1,
                            'ori_scaler': {i: i for i in range(ratings['ratings_mean'].max() + 1)},
                            'type': 'LabelEncoder', 'nan_value': None})
    return ratings, rating_datatype

def get_feature():
    """
    将多个方法整合在一起，得到movie数据，user数据，rating数据。
    然后将三个table合并到一起，组成一个大table。
    最后将table切割，分别得到features 和 target（rating）
    """
    title_length = 16
    path = os.path.abspath(os.path.join('../../', 'dataset/amz-book'))
    users, users_orig = user_data_processing(path)
    movies, movies_orig, genres2int,title_set = movie_data_processing(path)
    ratings, rating_datatype = rating_data_processing(path)

    #merge three tables
    data = pd.merge(pd.merge(ratings, users), movies)

    # split data to feature set:X and lable set:y
    target_fields = ['ratings']
    features, tragets_pd = data.drop(target_fields, axis=1), data[target_fields]
    # features = feature_pd.values


    # 针对ratings进行数据的分割，将ratings大于等于3的作为用户click的数据，反之为不会click的数据
    tragets_pd.ratings[tragets_pd['ratings'] <= 4] = 0
    tragets_pd.ratings[tragets_pd['ratings'] > 4] = 1

    targets = tragets_pd.values

    # 将处理后的数据保存到本地
    if not os.path.exists(os.path.join(path, 'feature')):
        os.makedirs(os.path.join(path, 'feature'))
    f = open(os.path.join(path, 'feature/ctr_features.p'), 'wb')
    # ['UserID' 'MovieID' 'Gender' 'Age' 'JobID' 'Title' 'Genres' 'ratings_counts' 'ratings_mean']
    pickle.dump(features, f)

    f = open(os.path.join(path, 'feature/ctr_target.p'), 'wb')
    pickle.dump(targets, f)

    f = open(os.path.join(path, 'feature/ctr_params.p'), 'wb')
    pickle.dump((title_length, title_set, genres2int, features, targets, \
                 ratings, users, movies, data, movies_orig, users_orig), f)

    f = open(os.path.join(path, 'feature/ctr_data.p'), 'wb')
    pickle.dump(data, f)

    return features, targets, data, users, movies


def split_train_test(feature, targets, data, users, movies):
    """
    将feature和targets分割成train, val, test。
    并将数据处理成两类，一种为onehot形式，一种为数据流形式
    :param feature:
    :param targets:
    :return:
    """
    x_train, y_train = feature, targets
    _, x_test, _, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=2022)
    _, x_val, _, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2022)

    # x_train, x_test, y_train, y_test = train_test_split(feature, targets, test_size=0.1, random_state=2022)#x是输入特征，y是label
    # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=2022)
    # item_data = data.loc[data['MovieID'] == 10]
    
    # data = data.sort_values(by='UserID')
    # # 初始化空列表，用于存储训练、验证和测试集
    # # x_train, x_val, x_test = [], [], []
    # # y_train, y_val, y_test = [], [], []
    # x_train = pd.DataFrame()
    # y_train = pd.DataFrame()
    # x_val = pd.DataFrame()
    # y_val = pd.DataFrame()
    # x_test = pd.DataFrame()
    # y_test = pd.DataFrame()
    
    # # 定义测试和验证集的比例
    # test_ratio = 0.05
    # val_ratio = 0.05
    # i=0
    # for user_id, group in data.groupby('UserID'):
    #     i+=1
    #     if i%1000==0:
    #         print(i)
    #     # 计算测试和验证集的大小
    #     test_size = int(len(group) * test_ratio)
    #     val_size = int(len(group) * val_ratio)
        
    #     # 分割数据
    #     test = group.iloc[-test_size:]
    #     val = group.iloc[-(test_size + val_size):-test_size]
    #     train = group.iloc[:-test_size - val_size]
    #     # print(train.head())
    #     # 将数据追加到相应的DataFrame和Series中
    #     x_train = pd.concat([x_train, train.drop('ratings', axis=1)])
    #     y_train = pd.concat([y_train, train['ratings']])
    #     x_val = pd.concat([x_val, val.drop('ratings', axis=1)])
    #     y_val = pd.concat([y_val, val['ratings']])
    #     x_test = pd.concat([x_test, test.drop('ratings', axis=1)])
    #     y_test = pd.concat([y_test, test['ratings']])
    # y_train.ratings[y_train['ratings'] <= 4] = 0
    # y_train.ratings[y_train['ratings'] > 4] = 1
    # y_val.ratings[y_val['ratings'] <= 4] = 0
    # y_val.ratings[y_val['ratings'] > 4] = 1
    # y_test.ratings[y_test['ratings'] <= 4] = 0
    # y_test.ratings[y_test['ratings'] > 4] = 1
    # print(x_train.shape, y_train.shape)
    # print(x_val.shape, y_val.shape)
    # print(x_test.shape, y_test.shape)
    # print(x_train.head())
    # print(y_train.head())
    # print(x_val.head())
    # print(y_val.head())
    # print(x_test.head())
    # print(y_test.head())
        
    # 获取所有的userid和itemid
    unique_userids = data['UserID'].unique()
    unique_itemids = data['MovieID'].unique()
    print(len(feature['MovieID'].unique()))
    print(data.shape)
    print(len(unique_userids), len(unique_itemids))
    print(data.head(10))
    # 创建userid-itemid配对的DataFrame
    u_special_pairs = []
    i_special_pairs = []
    for userid in unique_userids:
        # 选择一个固定itemid
        # fixed_itemid = unique_itemids[0]  # 这里假设我们总是选择第一个itemid
        user_data = data.loc[data['UserID'] == userid]
        rating = 1 if user_data['ratings'].values[0]>4 else 0
        new_entry = {'UserID': userid, 'MovieID': user_data['MovieID'].values[0], 'ratings_count': user_data['ratings_count'].values[0],
        'ratings_mean': user_data['ratings_mean'].values[0],
        'Title': user_data['Title'].values[0],
        'Genres': user_data['Genres'].values[0], 'ratings': rating}
        u_special_pairs.append(new_entry)
    print(len(u_special_pairs))
    for itemid in unique_itemids:
        # 选择一个固定userid
        fixed_userid = unique_userids[0]  # 同样，这里假设我们总是选择第一个userid
        item_data = data.loc[data['MovieID'] == itemid]
        rating = 1 if item_data['ratings'].values[0] > 4 else 0
        new_entry = {'UserID': item_data['UserID'].values[0], 'MovieID': itemid, 'ratings_count': item_data['ratings_count'].values[0],
        'ratings_mean': item_data['ratings_mean'].values[0],
        'Title': item_data['Title'].values[0],
        'Genres': item_data['Genres'].values[0], 'ratings': rating}
        i_special_pairs.append(new_entry)
    print(len(i_special_pairs))
    # u_special_pairs_df = pd.concat(u_special_pairs)
    # i_special_pairs_df = pd.concat(i_special_pairs)
    u_special_pairs_df = pd.DataFrame(u_special_pairs)
    i_special_pairs_df = pd.DataFrame(i_special_pairs)
    # special_pairs_df = pd.concat(special_pairs)
    # special_pairs_df = pd.DataFrame(special_pairs)
    print("=============================")
    print(x_train.shape, x_test.shape, x_val.shape)
    print(u_special_pairs_df.shape)
    print(i_special_pairs_df.shape)
    print(u_special_pairs_df.head(20))
    print(i_special_pairs_df.head(20))
    u_infer_x = u_special_pairs_df.drop(['ratings'], axis=1)
    i_infer_x = i_special_pairs_df.drop(['ratings'], axis=1)
    u_infer_y = u_special_pairs_df['ratings']
    i_infer_y = i_special_pairs_df['ratings']

    # 设置label为1
    # special_pairs_df['ratings'] = 1
    # print(special_pairs_df.head(20))
    # x_test = special_pairs_df.drop(['ratings'], axis=1)
    # print(x_test.shape)
    # print(x_test.head())
    # y_test = special_pairs_df['ratings']


    x_train.reset_index(drop=True, inplace=True)
    # y_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)
    # y_test.reset_index(drop=True, inplace=True)
    x_val.reset_index(drop=True, inplace=True)
    # y_val.reset_index(drop=True, inplace=True)

    # 保存为数据流格式, 对于seq和multi的数据，暂不处理
    path = os.path.join('../../', 'dataset/amz-book')
    f = open(os.path.join(path, 'feature/x_train.p'), 'wb')
    pickle.dump(x_train, f)

    f = open(os.path.join(path, 'feature/y_train.p'), 'wb')
    pickle.dump(y_train, f)

    f = open(os.path.join(path, 'feature/x_test.p'), 'wb')
    pickle.dump(x_test, f)

    f = open(os.path.join(path, 'feature/y_test.p'), 'wb')
    pickle.dump(y_test, f)

    f = open(os.path.join(path, 'feature/x_val.p'), 'wb')
    pickle.dump(x_val, f)

    f = open(os.path.join(path, 'feature/y_val.p'), 'wb')
    pickle.dump(y_val, f)

    f = open(os.path.join(path, 'feature/u_infer_x.p'), 'wb')
    pickle.dump(u_infer_x, f)
    f = open(os.path.join(path, 'feature/u_infer_y.p'), 'wb')
    pickle.dump(u_infer_y, f)
    f = open(os.path.join(path, 'feature/i_infer_x.p'), 'wb')
    pickle.dump(i_infer_x, f)
    f = open(os.path.join(path, 'feature/i_infer_y.p'), 'wb')
    pickle.dump(i_infer_y, f)



def onehot_format(feature, datatype, cols):
    res = []
    for _, r in tqdm(datatype[datatype['type'] == "LabelEncoder"].iterrows()):
        if r['name'] not in cols:
            continue
        # sc = pickle.loads(r.scaler)
        d = feature[r['name']]
        assert r['len'] > d.max()
        onehot = np.zeros((len(d), r['len']), dtype=np.int)
        onehot[np.arange(len(d)), d.astype(int)] = 1
        res.append(onehot)
    for _, r in tqdm(datatype[datatype['type'] == "MinMaxScaler"].iterrows()):
        if r['name'] not in cols:
            continue
        sc = pickle.loads(r['scaler'])
        # sc = MinMaxScaler()
        v = feature[r['name']].reshape((-1, 1))
        mask = np.isnan(v)
        v[~mask] = sc.transform(v[~mask].reshape((-1, 1))).reshape(-1)
        v[mask] = r['nan_value']
        res.append(v)
    for _, r in tqdm(datatype[datatype['type'] == "MultiLabelEncoder"].iterrows()):
        if r['name'] not in cols:
            continue
        d = feature[r['name']]
        onehot = np.zeros((len(d), r['len']), dtype=np.int)
        for index in range(len(d)):
            for item in d[index]:
                onehot[index, item] = 1
            # for item in d[index].split(','):
            #     onehot[index, item] = 1
        res.append(onehot)
    res = np.concatenate(res, axis=1)
    return res


def datanorm_xlearn(df, datatypes, tqdm=lambda x, *args, **kargs: x):
    """
    老版本 xlearn输入格式化
    :param df: 喂入过DeepFM后出来的数据
    :param datatypes:
    :param tqdm:
    :return:
    """
    features = []
    for _, r in tqdm(datatypes[datatypes.type == "LabelEncoder"].iterrows()):
        sc = pickle.loads(r.scaler)
        d = df[r.name]
        assert len(sc) > d.max()
        onehot = np.zeros((len(d), len(sc)))
        onehot[np.arange(len(d)), d.astype(int)] = 1
        features.append(onehot)
    for _, r in tqdm(datatypes[datatypes.type == "MinMaxScaler"].iterrows()):
        sc = pickle.loads(r.scaler)
        v = df[r.name].reshape((-1, 1))
        mask = np.isnan(v)
        v[~mask] = sc.transform(v[~mask].reshape((-1, 1))).reshape(-1)
        v[mask] = r.nan_value
        features.append(v)
    for _, r in tqdm(datatypes[datatypes.type == "keep"].iterrows()):
        v = df[r.name].reshape((-1, 1))
        features.append(v)
    features = np.concatenate(features, axis=1)
    return features

def main():
    features, targets, data, users, movies = get_feature()
    split_train_test(features, targets, data, users, movies)

if __name__ == '__main__':
    main()
