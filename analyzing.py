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
from sklearn.preprocessing import scale
import japanize_matplotlib
import graphviz
from graphviz import Source



class Analizing:



    def MakeExp(self, DataFrame, columns):

        exp = DataFrame[columns]

        # 正規化が必要な列は正規化
        if 'column11' in columns:
            exp['column11'] = scale(exp['column11'])
        if 'column28' in columns:
            exp['column28'] = scale(exp['column28'])

        # 体温等、正規化に工夫が必要なものは個別に対応
        # 平均体温をマイナスしてから正規化
        if 'column30' in columns:
            exp['column30'] = exp['column30'] - 36.89
            exp['column30'] = scale(exp['column30'])
        # 141.5をマイナスしてから正規化
        if 'column34' in columns:
            exp['column34'] = exp['column34'] - 141.5
            exp['column34'] = scale(exp['column34'])

        return exp


    def MakeTar(self, DataFrame, columns):

        tar = DataFrame[columns]

        return tar


    def MakeTTData(self, exp, tar):

        exp_X = exp.to_numpy(dtype=float)
        tar_Y = tar.to_numpy(dtype=float)
        # return exp_X

        trainX, testX, trainY, testY = train_test_split(
            exp_X,
            tar_Y,
            test_size=0.25,
            shuffle=True,
            random_state=0,
            stratify=tar_Y
        )

        return trainX, testX, trainY, testY

    def MakeTTData_ChangeRS(self, exp, tar, RS):

        exp_X = exp.to_numpy(dtype=float)
        tar_Y = tar.to_numpy(dtype=float)
        # return exp_X

        trainX, testX, trainY, testY = train_test_split(
            exp_X,
            tar_Y,
            test_size=0.25,
            shuffle=True,
            random_state=RS,
            stratify=tar_Y
        )

        return trainX, testX, trainY, testY


    def GenerateXGBoost(self, trainX, testX, trainY, testY, exp):

        trainXGB = xgb.DMatrix(trainX,
                               label=trainY,
                               feature_names=exp,
                               )
        testXGB = xgb.DMatrix(testX,
                               label=testY,
                               feature_names=exp,
                               )
        # return trainXGB

        param = {
            # 2値分類問題
            'objective':'binary:logistic',
            # 評価指標
            'eval_metric':'logloss',
            "learning_rate": 0.25,
            "max_depth": 3,
            "min_child_weight": 1,
            "colsample_bytree": 0.9,
            "lambda": 1,
        }

        xgb_model = xgb.train(
            param,
            trainXGB,
        )

        return xgb_model, trainXGB, testXGB


    def ShowHeatmap(self, xgb_model, testXGB, testY):

        predY_proba = xgb_model.predict(testXGB)
        predY_proba_over80 = np.where(predY_proba > 0.8, 1, 0)
        # print(predY_proba_over80)

        acc = accuracy_score(testY, predY_proba_over80)
        cmt = confusion_matrix(testY, predY_proba_over80)

        sns.heatmap(cmt, annot=True, cmap='Blues')
        plt.xlabel("予測結果(1: せん妄である、0: せん妄ではない)")
        plt.ylabel("テストデータ(1: せん妄である、0: せん妄ではない)")
        plt.show()


    def ShowAccuracies(self, xgb_model, testXGB, testY):

        predY_proba = xgb_model.predict(testXGB)
        predY_proba_over80 = np.where(predY_proba > 0.8, 1, 0)

        scoreAc = "accuracy_score:  " + str(accuracy_score(testY, predY_proba_over80))
        scorePr = "precision_score: " + str(precision_score(testY, predY_proba_over80))
        scoreRc = "recall_score:    " + str(recall_score(testY, predY_proba_over80))
        scoreF1 = "f1_score:        " + str(f1_score(testY, predY_proba_over80))

        print(scoreAc)
        print(scorePr)
        print(scoreRc)
        print(scoreF1)

    def ShowAccuracies_ToRecord(self, xgb_model, testXGB, testY):

        predY_proba = xgb_model.predict(testXGB)
        predY_proba_over80 = np.where(predY_proba > 0.8, 1, 0)

        scoreAc = accuracy_score(testY, predY_proba_over80)
        scorePr = precision_score(testY, predY_proba_over80)
        scoreRc = recall_score(testY, predY_proba_over80)
        scoreF1 = f1_score(testY, predY_proba_over80)

        return scoreAc, scorePr, scoreRc, scoreF1


    def ShowFeatureImportances(self, xgb_model):

        fig, ax1 = plt.subplots(figsize=(7, 10))
        xgb.plot_importance(xgb_model, ax=ax1, color='black')

        plb.tight_layout()
        plt.show()
        # column26,27,28,34,36,15 = 場所の認識、固執、JCS、Na、鎮静・麻酔、救急


    def ShowSHAP(self, xgb_model, trainX, exp):

        xgb_explainer = shap.TreeExplainer(xgb_model, data=trainX)
        xgb_shap_values = xgb_explainer.shap_values(trainX)

        shap.summary_plot(xgb_shap_values, trainX, feature_names=exp.columns)
        shap.summary_plot(xgb_shap_values, trainX, feature_names=exp.columns, plot_type='bar')


    # ↓要修正
    def ShowDecisionTree(self, xgb_model):

        xgb_DT = xgb.to_graphviz(xgb_model, num_trees=3)
        graph = graphviz.Source(xgb_DT)
        graph.render(filename='xgb_tree.png', format='png', cleanup=True)

        return graph

