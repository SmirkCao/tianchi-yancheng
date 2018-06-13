# -*- coding:utf-8 -*-
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
import utils
import xgboost as xgb
import chinese_calendar as cc
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def data_process():
    # 数据处理

    # 训练数据
    train_by_date_df = train_df.groupby(by='date', as_index=False).first()
    train_by_date_df["cnt"] = train_df.drop("brand", axis=1).groupby(by='date', as_index=False).sum()["cnt"]
    # 求DOW DIFF，NAN填充更改为-7
    train_by_date_df["dow_diff"] = train_by_date_df.day_of_week.diff().fillna(-7)
    # DOW DIFF 映射到1-7
    train_by_date_df.loc[train_by_date_df[train_by_date_df["dow_diff"] <= 0].index, 'dow_diff'] += 7

    print(train_by_date_df.tail())
    filename = BC.outputFolder + "train_by_date_df.csv"
    train_by_date_df.to_csv(filename)
    BC.log("Group train data by date, output file : " + filename)

    # 测试数据
    test_by_date_df = test_df.groupby(by='date', as_index=False).first()
    test_by_date_df["cnt"] = 0
    # 求DOW DIFF，NAN填充更改为-7
    test_by_date_df["dow_diff"] = test_by_date_df.day_of_week.diff().fillna(-7)
    print(test_by_date_df.head())
    # DOW DIFF 映射到1-7
    test_by_date_df.loc[test_by_date_df[test_by_date_df["dow_diff"] <= 0].index, 'dow_diff'] += 7
    # 去重
    # test_by_date_df = test_by_date_df[1:] # 有重叠
    test_by_date_df.loc[test_by_date_df[test_by_date_df["dow_diff"] == 0].index, 'dow_diff'] = 1  # 无重叠

    filename = BC.outputFolder + "test_by_date_df.csv"
    test_by_date_df.to_csv(filename)
    print(test_by_date_df.head())
    BC.log("Group test data by date, output file : " + filename)

    # 训练数据+测试数据 求date用
    all_data_df = pd.concat([train_by_date_df, test_by_date_df], axis=0)
    all_data_df.reset_index(inplace=True)
    print(all_data_df.describe())
    BC.log("Concate train data and test data")

    all_data_df["date_t"] = BC.gen_date_by_dows(startDate, all_data_df["dow_diff"])
    date2datet_df = all_data_df[["date", "date_t", "dow_diff"]]
    filename = BC.outputFolder + "date2datet.txt"
    date2datet_df.to_csv(filename)
    BC.log("Write date2datet map to file : " + filename)

    # 对日期进行补齐
    date_df = date2datet_df.set_index("date_t")
    date_df = date_df.resample('D').asfreq()
    date_df = date_df.fillna(0)
    date_df.reset_index(inplace=True)
    # print(date_df.info())
    filename = BC.outputFolder + "date2datet_intep.txt"
    date_df.to_csv(filename)
    BC.log("Write date_df map to file : " + filename)

    # 匹配真实日期
    all_data_df = pd.concat([train_df, test_df], axis=0)
    # all_data_df = pd.merge(all_data_df,date2datet_df,on="date")

    all_data_df.tail(20)
    # 生成date brand 模板
    datet_arr = date_df[["date_t"]].values.reshape(len(date_df))
    datet_brand_df = pd.DataFrame([[x, y] for x in datet_arr for y in np.arange(1, 11)], columns=["date_t", "brand"])
    datet_brand_date_dowdiff_df = pd.merge(datet_brand_df, date_df, on=['date_t'], how="outer")
    print(datet_brand_date_dowdiff_df[380:420])

    # 补全训练集品牌
    all_data_df = pd.merge(datet_brand_date_dowdiff_df, all_data_df, on=['date', 'brand'], how="outer")
    all_data_df["cnt"] = all_data_df["cnt"].fillna(0)
    BC.log("Fill all_data_df with brand")
    print(all_data_df.tail())
    BC.log("Merge all_data_df with date,dow_diff")

    all_data_df.info()
    return all_data_df


def gen_feas(all_data_df):
    # 添加特征
    all_data_df["year"] = all_data_df["date_t"].dt.year
    all_data_df["month"] = all_data_df["date_t"].dt.month
    all_data_df["day"] = all_data_df["date_t"].dt.day
    # mon -> 0, sun->6
    all_data_df["dow"] = all_data_df["date_t"].dt.dayofweek
    all_data_df["doy"] = all_data_df["date_t"].dt.dayofyear
    all_data_df["is_sa"] = all_data_df["dow"] == 5
    all_data_df["is_su"] = all_data_df["dow"] == 6
    BC.log("Add more features : year, month, day, dow, doy, is_sa, is_su")

    # 处理节假日
    all_data_df["is_holiday"] = [1 if cc.is_holiday(x.date()) else 0 for x in all_data_df.date_t]
    all_data_df["is_holiday"] = (all_data_df["is_sa"] != 1) & (all_data_df["is_su"] != 1) & (all_data_df["is_holiday"] == 1)
    all_data_df["is_holiday"] = [1 if x else 0 for x in all_data_df.is_holiday]
    all_data_df["is_WDA"] = [1 if cc.is_workday(x.date()) else 0 for x in all_data_df.date_t]
    all_data_df["is_WDA"] = ((all_data_df["is_sa"] == 1) | (all_data_df["is_su"] == 1)) & (all_data_df["is_WDA"] == 1)
    BC.log("Add more features : is_holiday, is_WDA")

    # 对比了下，单独划分没有明显效果，根据数据特征分组
    all_data_df["is_B12367"] = all_data_df["brand"].isin([1, 2, 3, 6, 7])
    all_data_df["is_B410"] = all_data_df["brand"].isin([4, 10])
    all_data_df["is_B5"] = all_data_df["brand"].isin([5])
    all_data_df["is_B8"] = all_data_df["brand"].isin([8])
    all_data_df["is_B9"] = all_data_df["brand"].isin([9])
    BC.log("Add more features : is_B12367, is_B410, is_B5, is_B8，is_B9")

    # 外部数据
    # 增加车牌量
    # 处理最后一个月的值
    filename = "../Input/car_sale_volume.csv"
    car_sales_df = pd.read_csv(filename)
    car_sales_df = car_sales_df[["sale_quantity", "year", "month"]]
    BC.log("Read cas sales : " + filename)
    # 最后一个月的数据处理
    df_ = pd.Series([car_sales_df[-3:]["sale_quantity"].mean(), 2017, 11],
                    index=["sale_quantity", "year", "month"]).to_frame()
    car_sales_df = pd.concat([car_sales_df, df_.T])
    # car_sales_df.reset_index(inplace=True)
    all_data_df = pd.merge(all_data_df, car_sales_df, on=["year", "month"])
    all_data_df.drop("day_of_week", inplace=True, axis=1)

    all_data_df["is_Jan"] = all_data_df["month"] == 1
    all_data_df["is_Feb"] = all_data_df["month"] == 2

    BC.log("Add more features : is_Jan, is_Feb")

    filename = BC.outputFolder + "all_data.csv"
    all_data_df.to_csv(filename)
    BC.log("Write all_data_df to " + filename)
    return all_data_df


def find_params(train_X, train_y, param_grid):
    xgb_model = xgb.XGBRegressor(random_state=2018)
    rgs = GridSearchCV(xgb_model, param_grid, n_jobs=8)
    rgs.fit(train_X, train_y, eval_metric='rmse')
    print(rgs.best_score_)
    print(rgs.best_params_)
    return rgs


def sub(result_in):
    result_df = all_data_df[13810:]
    result_df["cnt"] = result_in
    BC.log("Result info:" + str(result_df.info()))

    # 最后输出的结果也与测试集和训练集之间的接口有点关系
    BC.log("Generate result")
    filename = "../Input/fusai_sample_B_20180227.txt"
    result_sample_df = pd.read_table(filename, names=["date", "brand", "cnt"])
    BC.log("Read sample file." + filename)
    # TODO: 效率
    cnt_arr = []
    for x in result_sample_df.values:
        cnt_arr.append(int(result_df[(result_df["date"] == x[0]) & (result_df["brand"] == x[1])]["cnt"].values[0]))

    # 后处理
    result_sample_df["cnt"] = cnt_arr
    result_sample_df["cnt"][result_sample_df["cnt"] < 0] = 12
    result_sample_df.head()

    filename = pd.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    filename = BC.outputFolder + "Result_" + filename + ".txt"
    result_sample_df.loc[:, ["date", "brand", "cnt"]].to_csv(filename, sep="\t", header=False, index=False)

    BC.log("Write final result to " + filename)
    BC.log("----The End----")
    BC.endTime = pd.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    filename = BC.outputFolder + "OperationRecord.txt"
    with open(filename, "w") as f:
        f.write(str(BC))


def for_other():
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.title("Feature importances")
    plt.bar(range(len(feature_importances)), feature_importances[indices], color="b", align="center")
    plt.xticks(range(len(feature_importances)), np.array(feature_names)[indices])
    plt.xlim([-1, train_X.shape[1]])
    plt.show()


if __name__ == "__main__":
    print("Talk is cheap. Show me the code.")
    BC = utils.Briefcase()
    # Load Data start
    filename = "../Input/fusai_train_20180227.txt"
    train_df = pd.read_table(filename, dtype='int')
    print(train_df.tail())
    BC.log("Read train file : " + filename)

    filename = "../Input/fusai_test_A_20180227.txt"
    test_a_df = pd.read_table(filename, dtype="int")
    print(test_a_df.tail())
    BC.log("Read test A file to test_a_df: " + filename)

    filename = "../Input/fusai_answer_a_20180307.txt"
    answer_a_df = pd.read_table(filename, dtype="int", names=["date", "brand", "cnt"])
    BC.log("Read test A answer file to answer_a_df: " + filename)
    answer_a_df.head()

    test_a_df = pd.merge(test_a_df, answer_a_df, on=["date", "brand"])
    BC.log("Merge test A cnt")
    train_df = pd.concat([train_df, test_a_df[1:]])
    BC.log("Concat train_df and test_a_df")

    filename = "../Input/fusai_test_B_20180227.txt"
    test_df = pd.read_table(filename, dtype="int")
    # print(test_df.tail())
    BC.log("Read test B file to test_df: " + filename)
    del test_a_df
    del answer_a_df
    # Load Data End

    startDate = pd.to_datetime("2013-01-01")  # 初始日期

    all_data_df = data_process()

    # Feature Engineering
    all_data_df = gen_feas(all_data_df)

    # 保留必须特征，添加不同特征，看效果进行特征选择
    train_feas = ["brand", "dow_diff", "year", "month", "day", "dow", "doy",
                  "is_sa", "is_su", "is_holiday", "is_WDA",
                  "is_B12367", "is_B410", "is_B5", "is_B8", "is_B9",
                  "is_Jan", "is_Feb", "sale_quantity"
                  ]
    BC.log("Train features : " + str(train_feas))
    # TODO: 贪心个自动特征选择的方案

    train_X = all_data_df[:13810][train_feas].values
    train_y = all_data_df[:13810]['cnt'].values
    train_y = train_y.reshape(train_y.shape[0], 1)

    param_grid = {
        'max_depth':[3],
        # 'max_depth': [3, 4, 5],
        # 'n_estimators': [20, 40, 50, 60, 80, 100, 200, 400],
        # 'learning_rate': [0.01, 0.02, 0.05, 0.1, 0.2],
        # 'subsample': [0.65, 0.7, 0.8],
        # 'colsample_bylevel': [0.65, 0.7, 0.8],
    }

    BC.log("Start to search params : \n" + str(param_grid))
    rgs = find_params(train_X, train_y, param_grid)
    BC.log("End to search params.")
    BC.log("Best Score is " + str(rgs.best_score_))
    BC.log("Best Params are " + str(rgs.best_params_))
    BC.log(str(rgs.best_estimator_.feature_importances_))
    print("Feature ranking:")

    feature_importances = rgs.best_estimator_.feature_importances_
    BC.log(str(train_feas) + str(feature_importances))
    indices = np.argsort(feature_importances)[::-1]
    # BC.log("Plot feature ranking.")

    for f in indices:
        print("feature %s (%f)" % (train_feas[f], feature_importances[f]))

    predict_y = rgs.best_estimator_.predict(train_X)
    train_data_df = all_data_df[:13810]
    train_data_df = train_data_df[["date", "date_t", "brand", "cnt"]]
    train_data_df["cnt_p"] = predict_y

    filename = BC.outputFolder + BC.startTime + "train_data_df.csv"
    train_data_df.to_csv(filename)

    # 预测
    test_X = all_data_df[13810:][train_feas].values

    BC.log("Start to predict")
    result = rgs.best_estimator_.predict(test_X)
    sub(result)
