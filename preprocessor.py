"""
Ziwei Zhu
Computer Science and Engineering Department, Texas A&M University
zhuziwei@tamu.edu
"""
import numpy as np
import pandas as pd


class ml1m:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/ml-1m/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/ml-1m/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        return train_R, vali_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/ml-1m/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['movieId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/ml-1m/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R

class Pinterest:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/p/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/p/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        return train_R, vali_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/p/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['movieId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/p/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['movieId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R




class yelp:
    def __init__(self):
        return

    @staticmethod
    def train(n):
        train_df = pd.read_csv('./data/yelp/train_%d.csv' % n)
        vali_df = pd.read_csv('./data/yelp/vali_%d.csv' % n)
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['itemId'])

        train_R = np.zeros((num_users, num_items))  # training rating matrix
        vali_R = np.zeros((num_users, num_items))  # validation rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1

        vali_mat = vali_df.values
        for i in range(len(vali_df)):
            user_idx = int(vali_mat[i, 0]) - 1
            item_idx = int(vali_mat[i, 1]) - 1
            vali_R[user_idx, item_idx] = 1
        return train_R, vali_R

    @staticmethod
    def test():
        test_df = pd.read_csv('./data/yelp/test.csv')
        num_users = np.max(test_df['userId'])
        num_items = np.max(test_df['itemId'])

        test_R = np.zeros((num_users, num_items))  # testing rating matrix

        test_mat = test_df.values
        for i in range(len(test_df)):
            user_idx = int(test_mat[i, 0]) - 1
            item_idx = int(test_mat[i, 1]) - 1
            test_R[user_idx, item_idx] = 1

        train_df = pd.read_csv('./data/yelp/train.csv')
        num_users = np.max(train_df['userId'])
        num_items = np.max(train_df['itemId'])

        train_R = np.zeros((num_users, num_items))  # testing rating matrix

        train_mat = train_df.values
        for i in range(len(train_df)):
            user_idx = int(train_mat[i, 0]) - 1
            item_idx = int(train_mat[i, 1]) - 1
            train_R[user_idx, item_idx] = 1
            train_R[user_idx, item_idx] = 1

        return train_R, test_R

