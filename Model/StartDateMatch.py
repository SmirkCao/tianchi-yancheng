#-*-coding:utf-8-*-
# Project: tianchi-yancheng  
# Filename: StartDateMatch
# Author: Smirk <smirk dot cao at gmail dot com>
# 日期匹配方法
# 最粗暴方法是求和最小的来匹配，这样计算量有点大，但是依然是可以实现的。
import chinese_calendar as cc
import pandas as pd
import utils


class BC_Tinachi_Yancheng(utils.Briefcase):
    @staticmethod
    def start_date_match(start_date, end_date, **kw):
        # 生成初始日期序列
        rst_date = pd.DataFrame({"date": [], "count": []})

        last_count = 0

        for tryDate in pd.date_range(start=startDatein, end=endDatein):
            #     print(x.dayofweek) #0-6 -> 1-7
            train_by_date_zeromatch_df = kw["train_by_date_df"]

            if tryDate.dayofweek == 2:
                train_by_date_zeromatch_df["date_t"] = utils.Briefcase.gen_date_by_dows(tryDate,
                                                                                        train_by_date_zeromatch_df["dow_diff"])
                train_by_date_zeromatch_df["is_holiday"] = [1 if cc.is_holiday(x.date()) else 0 for x in
                                                            train_by_date_zeromatch_df["date_t"]]
                zeromatch_idx = train_by_date_zeromatch_df.loc[:, "day_of_week"].isin([6, 7])
                for x in range(len(zeromatch_idx)):
                    if zeromatch_idx[x] == True:
                        train_by_date_zeromatch_df.drop(x, inplace=True)
                curr_count = train_by_date_zeromatch_df["is_holiday"].sum()
                if last_count == 0:
                    last_count = curr_count
                    min_count = curr_count
                min_count = min(min_count, curr_count)
                last_count = curr_count
                print(tryDate.date(), last_count, min_count)

                df = pd.DataFrame({"date": [tryDate], "count": [train_by_date_zeromatch_df["is_holiday"].sum()]})
                rst_date = pd.concat([rst_date, df], ignore_index=True)
        rst_date.describe()
        return rst_date[rst_date["count"] == rst_date["count"].min()]


# 肯定是历史数据，所以三年的历史数据的话，这个数值不会超过20171225之前三年20141225
startDatein = "2008-1-1"
endDatein = "2014-2-7"

# 读取训练数据
train_df = pd.read_table("../Input/fusai_train_20180227.txt", dtype='int')
# 不考虑品牌因素，按照日期求和
train_by_date_df = train_df.drop("brand", axis=1).groupby(by='date', as_index=False).first()
train_by_date_df["cnt"] = train_df.drop("brand", axis=1).groupby(by='date', as_index=False).sum()["cnt"]
# 求DOW DIFF，NAN填充更改为-7
train_by_date_df["dow_diff"] = train_by_date_df.day_of_week.diff().fillna(-7)
train_by_date_df.loc[train_by_date_df[train_by_date_df["dow_diff"] <= 0].index, 'dow_diff'] += 7

bc = BC_Tinachi_Yancheng()
bc.start_date_match(startDatein, endDatein, **{"train_by_date_df": train_by_date_df})
