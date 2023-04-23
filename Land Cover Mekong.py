# -*- coding: utf-8 -*-
"""
Created on Wed May 11 20:50:33 2022

@author: neptuner
"""

import os
from argparse import ArgumentParser

from osgeo import gdal
import yaml
import json
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import joblib

import numpy as np
import sklearn.metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize

import data_preprocessing as DP
import evaluation_metrics as EM
import spectual_constants as SC


def pca_top(pixel_array, n_components, mode, pca_path):  # PCA计算
    """
    # 重整数据成二维
    shape = image_array.shape
    pixel_array = np.reshape(image_array, (shape[0] * shape[1], shape[2]))
    """
    # If 0 < n_components < 1 and svd_solver == 'full', select the number of components
    # such that the amount of variance that needs to be explained is
    # greater than the percentage specified by n_components.
    if mode == "train":
        if os.path.isfile(pca_path):
            pca = joblib.load(pca_path)
        else:
            pca = PCA(n_components)
        pca.fit(pixel_array)
        pixel_array_pca = pca.transform(pixel_array)
        joblib.dump(pca, pca_path)
    elif mode == "produce":
        pca = joblib.load(pca_path)
        pixel_array_pca = pca.transform(pixel_array)
    else:
        raise AttributeError
        print("mode input is in wrong format. please check again.")

    return pixel_array_pca


# 数据整合和归一化处理
def data_fusion(channels,
                mode,
                pca_path):
    # 获得各个通道数据
    S1 = channels[:, 0: 2]
    S2 = channels[:, 2: 8]
    ti = channels[:, 8:16]
    # DSM = channels[:, 8]
    DSM = channels[:, 16]
    GEZ = channels[:, 17]

    # 计算遥感指数
    ndvi = SC.getNDVI(S2)
    mndwi = SC.getMNDWI(S2)
    ndbi = SC.getNDBI(S2)
    ndgi = SC.getNDGI(S2)

    rs_consts = np.stack((ndvi, mndwi, ndbi, ndgi), axis=1)
    # 计算和保存pca
    pca = pca_top(pixel_array=S2,
                  n_components=6,
                  mode=mode,
                  pca_path=pca_path)

    # 将所有参数整合在一起
    S1_name = ["S1_1", "S1_2"]
    S2_name = ["S2_1", "S2_2", "S2_3", "S2_4", "S2_5", "S2_6"]
    PCA_name = ["PCA_1", "PCA_2", "PCA_3", "PCA_4", "PCA_5", "PCA_6"]
    DSM_name = ["DEM"]
    TI_name = ["TI_1", "TI_2", "TI_3", "TI_4", "TI_5", "TI_6", "TI_7", "TI_8"]
    RSI_name = ["RSI_1", "RSI_2", "RSI_3", "RSI_4"]
    GEZ_name = ["GEZ"]

    channels = np.concatenate((S1,
                               # S2,
                               pca,
                               DSM[:, np.newaxis],
                               ti,
                               rs_consts
                               ), axis=1)
    features_name = S1_name + \
                    PCA_name + \
                    DSM_name + \
                    TI_name + \
                    RSI_name
                    # GEZ_name + \
                    # S2_name + \

    # 输入数据归一化
    channels[np.isnan(channels)] = 0
    # channels = normalize(channels, axis=0, norm='max')
    # channels = np.concatenate((channels,
    #                            GEZ[:, np.newaxis]
    #                            ), axis=1)
    return channels, features_name


def test_data_fusion(S1img_data,
                     S2img_data,
                     TI_data,
                     DEM_data,
                     mode,
                     pca_path):

    # 计算遥感指数
    ndvi = SC.getNDVI(S2img_data)
    mndwi = SC.getMNDWI(S2img_data)
    ndbi = SC.getNDBI(S2img_data)
    ndgi = SC.getNDGI(S2img_data)

    rs_consts = np.stack((ndvi, mndwi, ndbi, ndgi), axis=1)
    # 计算和保存pca
    pca = pca_top(pixel_array=S2img_data,
                  n_components=6,
                  mode=mode,
                  pca_path=pca_path)

    # 将所有参数整合在一起
    S1_name = ["S1_1", "S1_2"]
    S2_name = ["S2_1", "S2_2", "S2_3", "S2_4", "S2_5", "S2_6"]
    PCA_name = ["PCA_1", "PCA_2", "PCA_3", "PCA_4", "PCA_5", "PCA_6"]
    DSM_name = ["DEM"]
    TI_name = ["TI_1", "TI_2", "TI_3", "TI_4", "TI_5", "TI_6", "TI_7", "TI_8"]
    RSI_name = ["RSI_1", "RSI_2", "RSI_3", "RSI_4"]
    GEZ_name = ["GEZ"]

    channels = np.concatenate((S1img_data,
                               # S2img_data,
                               pca,
                               DEM_data,
                               TI_data,
                               rs_consts
                               ), axis=1)
    features_name = S1_name + \
                    PCA_name + \
                    DSM_name + \
                    TI_name + \
                    RSI_name

                    # GEZ_name + \
                    # S2_name + \

    # 输入数据归一化
    channels[np.isnan(channels)] = 0
    # channels = normalize(channels, axis=0, norm='max')
    return channels, features_name


def select_combine(X_train,
                   y_train,
                   seq_feat_imp,
                   rf):
    oob_result = []
    feat_n = len(seq_feat_imp)
    for i in range(feat_n):
        train_index = seq_feat_imp[: i + 1]
        train_data = X_train[:, train_index]
        rf.fit(train_data, y_train)
        acc = rf.oob_score_
        print(acc)
        oob_result.append(acc)
    return oob_result


def plotDiagram(x_data,
                y1_data,
                y2_data,
                save_path):
    # 遇到数据中有中文的时候，一定要先设置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial']

    x = plt.figure(figsize=(8, 6))
    a = x.add_subplot(111)   # 一行一列一个
    a.bar(x_data,
          y1_data,
          label='feature importance')
    a.legend(loc='upper left')
    bfb = mtick.FormatStrFormatter('%.2f')
    a.set_xticklabels(x_data, rotation=45)
    # a.set_ylim(0, 0.08)
    a.yaxis.set_major_formatter(bfb)

    b = a.twinx()   # 共用x轴
    b.plot(x_data,
           y2_data,
           color='orange',
           marker='o',
           label='OOB accuracy')
    b.legend(loc='upper right')
    bfb = mtick.FormatStrFormatter('%.2f')
    b.yaxis.set_major_formatter(bfb)

    # for i, j in zip(x_data, y2_data):
    #     plt.text(i, j, f"{j:%.1f%%}")
    plt.show()
    x.savefig(save_path, dpi=600)


def train(channels_data=None,
          labels_data=None,
          training_ratio=0.8,
          criterion='entropy',
          max_depth=None,
          output_folder='',
          experi_feat=''):
    # 数据处理，加入指数维度和pca维度
    channels_data, features_name = data_fusion(channels_data,
                                               mode="train",
                                               pca_path=os.path.join(output_folder, 'pca.joblib'))

    # DEM_data = channels_data[:, features_name.index('DEM')]
    # GEZ_data = channels_data[:, features_name.index('GEZ')]
    # data_n = len(channels_data)
    # seq_dem = np.argsort(DEM_data)
    # seq_gez = np.argsort(GEZ_data)
    # gez_idx = list(set(GEZ_data))
    # parts = 5
    # gez_count = [np.sum(GEZ_data == idx) for idx in gez_idx]

    # for i, gez_type in enumerate(gez_idx):
        # subseq = seq_dem[i * int(data_n / parts): (i + 1) * int(data_n / parts)]
        # if gez_type < 0:
        #     continue

        # subseq = seq_gez[np.sum(gez_count[:i]): np.sum(gez_count[:i + 1])]
        # sub_channels_data = np.delete(channels_data[subseq],
        #                               (# features_name.index('DEM'),
        #                                features_name.index('GEZ')
        #                                ),
        #                               axis = 1)
        # sub_labels_data = labels_data[subseq]
        #
        # print("gez type is ", int(gez_type / 35),
        #       np.sum(sub_labels_data == 1),
        #       np.sum(sub_labels_data == 2),
        #       np.sum(sub_labels_data == 3),
        #       np.sum(sub_labels_data == 4),
        #       np.sum(sub_labels_data == 5))
        #
        # cls_stat = np.array([np.sum(sub_labels_data == 1),
        #                      np.sum(sub_labels_data == 2),
        #                      np.sum(sub_labels_data == 3),
        #                      np.sum(sub_labels_data == 4),
        #                      np.sum(sub_labels_data == 5)])

        # 将数据分为训练集和验证集，训练集占比：training_ratio
        # data_num = np.shape(sub_channels_data)[0]
    data_num = np.shape(channels_data)[0]
    # 数据随机化
    perm = np.random.permutation(data_num)
    # rand_channels_data = sub_channels_data[perm]
    # rand_labels_data = sub_labels_data[perm]
    rand_channels_data = channels_data[perm]
    rand_labels_data = labels_data[perm]

    training_channels = rand_channels_data[0: int(data_num * training_ratio)]
    training_labels = rand_labels_data[0: int(data_num * training_ratio)]

    test_channels = rand_channels_data[int(data_num * training_ratio):]
    test_labels = rand_labels_data[int(data_num * training_ratio):]

    # 分类器
    for n_estimators in range(40, 150, 10):  # 对RF的决策树数量参数进行遍历
        # 构建随机森林
        if max_depth is not None:
            RF = RandomForestClassifier(n_estimators=n_estimators,
                                        criterion=criterion,
                                        max_depth=max_depth,
                                        class_weight='balanced',
                                        oob_score=True)
        else:
            RF = RandomForestClassifier(n_estimators=n_estimators,
                                        criterion=criterion,
                                        class_weight='balanced',
                                        oob_score=True)
        # 训练随机森林
        RF.fit(training_channels, training_labels)

        # 保存rf模型
        with open(os.path.join(output_folder, experi_feat, 'RF_' + str(n_estimators) + \
                                                           # '_' + str(int(gez_type / 35)) + \
                                                           '_' + criterion + '.pickle'), 'wb') as f:
            pickle.dump(RF, f)

        # 特征重要性排序
        features_importance = RF.feature_importances_

        seq_features = list(features_importance)  # 生成一个排序特征，进行筛选
        seq_features = np.argsort(seq_features)[::-1]
        seq_features_name = [features_name[k] for k in seq_features]
        seq_features_imp = [features_importance[k] for k in seq_features]

        # 预测和生产精度指标
        output = {'seq_feature_name': seq_features_name,
                  'seq_features': seq_features.tolist(),
                  'seq_feature_imp': seq_features_imp,
                  }
        predict_labels = RF.predict(test_channels)
        # accuracy = calculate_accuracy(predict_labels, test_labels)
        confusion_matrix = sklearn.metrics.confusion_matrix(test_labels,
                                                            predict_labels,
                                                            labels=[1, 2, 3, 4, 5])
        accuracy = sklearn.metrics.accuracy_score(test_labels,
                                                  predict_labels)
        kappa = sklearn.metrics.cohen_kappa_score(test_labels,
                                                  predict_labels)
        f1_score = sklearn.metrics.f1_score(test_labels,
                                            predict_labels,
                                            average='weighted')
        precision = sklearn.metrics.precision_score(test_labels,
                                                    predict_labels,
                                                    average='weighted')
        recall = sklearn.metrics.recall_score(test_labels,
                                              predict_labels,
                                              average='weighted')

        output['confusion_matrix'] = confusion_matrix.tolist()
        output['accuracy'] = accuracy
        output['kappa'] = kappa
        output['f1_score'] = f1_score
        output['precision'] = precision
        output['recall'] = recall
        # output['cls_stat'] = cls_stat.tolist()

        print('confusion matrix:', confusion_matrix)
        print('accuracy: ', accuracy)
        print('kappa: ', kappa)
        print('f1_score: ', f1_score)
        print('precision: ', precision)
        print('recall: ', recall)

        oob_result = select_combine(training_channels,
                                    training_labels,
                                    seq_features,
                                    RF)
        output['oob_result'] = oob_result

        # 保存output
        plotDiagram(seq_features_name,
                    seq_features_imp,
                    oob_result,
                    os.path.join(output_folder, experi_feat, str(n_estimators) \
                                 # + '_' + str(int(gez_type / 35))\
                                 + '_featimp.png'))

        with open(os.path.join(output_folder, experi_feat, 'RF_' + str(n_estimators) + \
                                                           # '_' + str(int(gez_type / 35)) + \
                                                           '_' + criterion + '.json'), 'w') as out:
            json.dump(output, out)


def test(RF_path,
         pca_path,
         test_folder,
         output_folder):
    with open(RF_path, 'rb') as F:
        rf = pickle.load(F)

    for test_img in os.listdir(os.path.join(test_folder, 'S2')):
        # if test_img.split('.')[-1] == 'tif':
        if test_img == 'lanmei_254.tif':
            if not os.path.isfile(os.path.join(output_folder, test_img)):
                S1image_dataset = gdal.Open(os.path.join(test_folder, 'S1', 'warppedS1', test_img), gdal.GA_ReadOnly)
                S2image_dataset = gdal.Open(os.path.join(test_folder, 'S2', test_img), gdal.GA_ReadOnly)
                TIimage_dataset = gdal.Open(os.path.join(test_folder, r'transTI', test_img), gdal.GA_ReadOnly)
                DEM_dataset = gdal.Open(os.path.join(test_folder, r'DEM', 'warppedDEM', test_img), gdal.GA_ReadOnly)
                geo_transform = S2image_dataset.GetGeoTransform()
                projection = S2image_dataset.GetProjectionRef()
                img_columns = S2image_dataset.RasterXSize
                img_rows = S2image_dataset.RasterYSize

                unit = 500
                pred = np.zeros((img_rows, img_columns))
                for i in range(int(img_rows / unit)):
                    for j in range(int(img_columns / unit)):
                        S1img_data = np.array(S1image_dataset.ReadAsArray(j * unit, i * unit, unit, unit), dtype=float)
                        S1img_data = np.reshape(np.transpose(S1img_data, (1, 2, 0)), (unit * unit, S1img_data.shape[0]))
                        S2img_data = np.array(S2image_dataset.ReadAsArray(j * unit, i * unit, unit, unit), dtype=float)
                        S2img_data = np.reshape(np.transpose(S2img_data, (1, 2, 0)), (unit * unit, S2img_data.shape[0]))
                        TIimg_data = np.array(TIimage_dataset.ReadAsArray(j * unit, i * unit, unit, unit), dtype=float)
                        TIimg_data = np.reshape(np.transpose(TIimg_data, (1, 2, 0)), (unit * unit, TIimg_data.shape[0]))
                        DEMimg_data = np.array(DEM_dataset.ReadAsArray(j * unit, i * unit, unit, unit), dtype=float)
                        DEMimg_data = DEMimg_data.ravel()[:, np.newaxis]

                        pred_data,_ = test_data_fusion(S1img_data,
                                                       S2img_data,
                                                       TIimg_data,
                                                       DEMimg_data,
                                                       mode='produce',
                                                       pca_path=pca_path)

                        output = rf.predict(pred_data)
                        pred[i * unit: (i + 1) * unit, j * unit: (j + 1) * unit] = np.reshape(output, (unit, unit))
                        print(i, j)

                DP.saveImage(os.path.join(output_folder, test_img), pred, geo_transform, projection)


if __name__ == '__main__':
    # 输入准备
    parser = ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r"a.yaml", help="...")
    args = parser.parse_args()
    filepath = os.path.join(os.getcwd(), args.cfg)

    with open(filepath, 'r', encoding='utf-8') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)

    # 训练
    if args['mode'] == 'train':
        # 分类别读取feature和label
        feature_data_paths = os.listdir(args['train_data_folder'])
        label_names = list(set([f.split('-')[-1].split('.')[0] for f in feature_data_paths]))
        label_names.sort()
        features = []
        labels = []
        for feature_data_path in feature_data_paths:
            feature = np.load(os.path.join(args['train_data_folder'], feature_data_path))
            shape = np.shape(feature)
            if shape[0] != 0:
                features.append(feature)
                labels.append(np.ones(shape[0]) * (label_names.index(feature_data_path.split('-')[-1].split('.')[0]) + 1))
        # 整合读取结果
        channels_data = np.concatenate(features)
        labels_data = np.concatenate(labels)

        # 读取训练参数
        training_ratio = 0.8
        # n_estimators = args['n_estimators']
        criterion = args['criterion']
        max_depth = args['max_depth']

        output_folder = args['output_model_folder']
        experi_feat = args['experi_feat']
        # 创建结果存放文件夹
        if os.path.isdir(os.path.join(output_folder, experi_feat)) is False:
            os.mkdir(os.path.join(output_folder, experi_feat))

        # 训练
        train(channels_data, labels_data, training_ratio,
              criterion=criterion,
              max_depth=max_depth,
              output_folder=output_folder,
              experi_feat=experi_feat)

    # 测试
    elif args['mode'] == 'produce':
        RF_path = args['input_random_forest_model']
        pca_path = args['input_pca_model']
        test_folder = args['input_image_folder']
        output_folder = args['output_image_folder']
        test(RF_path, pca_path, test_folder, output_folder)
