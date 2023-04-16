import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import matplotlib.pylab as plb
import shap
import seaborn as sns
import japanize_matplotlib
import IPython.display as display
from preprocessing import Preprocessing
from analyzing import Analizing

def main():
    # データの前処理
    pp = Preprocessing()
    # csvデータ読み込み
    xgbData = pp.DataRead()

    # データの分析
    an = Analizing()
    # 説明変数の設定
    useColumns = [
        'column1', 'column2', 'column3', 'column4', 'column5',
        'column6', 'column7', 'column8', 'column9', 'column10',
        'column11', 'column12', 'column13', 'column14', 'column15',
        'column16', 'column17', 'column18', 'column19', 'column20',
        'column21', 'column22', 'column23', 'column24', 'column25',
        'column26', 'column27', 'column28', 'column29', 'column30',
        'column31', 'column32', 'column33', 'column34', 'column35',
        'column36', 'column37', 'column38', 'column39', 'column40',
        'column41', 'column42', 'column43', 'column44', 'column45',
        'column46', 'column47', 'column48'
        ]
    expAll = an.MakeExp(xgbData, useColumns)
    # print(expAll)

    # useColumns = [
    #     'column34'
    # ]
    # expCheck = an.MakeExp(xgbData, useColumns)
    # print(expCheck)

    # 目的変数の設定
    useColumns = [
        'target'
    ]
    tar = an.MakeTar(xgbData, useColumns)
    # print(tar)

    # データの切り分け
    trainX, testX, trainY, testY = an.MakeTTData(expAll, tar)
    # print(trainX)

    # XGBoostを用いた分析
    xgbModel, trainXGB, testXGB = an.GenerateXGBoost(trainX, testX, trainY, testY, expAll)
    # print(xgbModel)

    # 分析結果を図示(Heatmap)
    # an.ShowHeatmap(xgbModel, testXGB, testY)

    # 正解率の表示
    # an.ShowAccuracies(xgbModel, testXGB, testY)

    # 特徴量の重要度を図示
    # an.ShowFeatureImportances(xgbModel)

    # SHAP値を図示
    # an.ShowSHAP(xgbModel, trainX, expAll)

    # 決定木の表示(改良中)
    # graph = an.ShowDecisionTree(xgbModel)
    # display.display(graph)



    # 10000回繰り返して正解率の平均をとる(全体)
    # AveAc, AvePr, AveRc, AveF1 = 0.0, 0.0, 0.0, 0.0
    # for i in range(10000):
    #     trainX, testX, trainY, testY = an.MakeTTData_ChangeRS(expAll, tar, i)
    #     xgbModel, trainXGB, testXGB = an.GenerateXGBoost(trainX, testX, trainY, testY, expAll)
    #     scoreAc, scorePr, scoreRc, scoreF1 = an.ShowAccuracies_ToRecord(xgbModel, testXGB, testY)
    #
    #     AveAc = AveAc + float(scoreAc)
    #     AvePr = AvePr + float(scorePr)
    #     AveRc = AveRc + float(scoreRc)
    #     AveF1 = AveF1 + float(scoreF1)
    #
    # print(AveAc / 10000)
    # print(AvePr / 10000)
    # print(AveRc / 10000)
    # print(AveF1 / 10000)


    # # 説明変数を絞って10000回
    # useColumns = [
    #     'column26', 'column27', 'column28', 'column34', 'column15'
    # ]
    # expFiltered = an.MakeExp(xgbData, useColumns)
    # AveAc, AvePr, AveRc, AveF1 = 0.0, 0.0, 0.0, 0.0
    # for i in range(10000):
    #     trainX, testX, trainY, testY = an.MakeTTData_ChangeRS(expFiltered, tar, i)
    #     xgbModel, trainXGB, testXGB = an.GenerateXGBoost(trainX, testX, trainY, testY, expFiltered)
    #     scoreAc, scorePr, scoreRc, scoreF1 = an.ShowAccuracies_ToRecord(xgbModel, testXGB, testY)
    #
    #     AveAc = AveAc + float(scoreAc)
    #     AvePr = AvePr + float(scorePr)
    #     AveRc = AveRc + float(scoreRc)
    #     AveF1 = AveF1 + float(scoreF1)
    #
    # print(AveAc / 10000)
    # print(AvePr / 10000)
    # print(AveRc / 10000)
    # print(AveF1 / 10000)


if __name__ == '__main__':
    main()

