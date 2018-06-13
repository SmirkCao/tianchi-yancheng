from datetime import timedelta
import os
import pandas as pd


class Briefcase:
    startTime = ""
    endTime = ""
    outputFolder = ""
    oprLog = ""

    def __init__(self, ):
        path = os.getcwd()
        self.startTime = pd.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.outputFolder = path + "/../Output/" + self.startTime + "/"
        self.relativeOutputFolder = "/../Output/" + self.startTime + "/"
        os.mkdir(self.outputFolder)

    def __str__(self, ):
        str_return = "----Start Time----\n"
        str_return += self.startTime + "\n"
        str_return += "----End Time----\n"
        str_return += self.endTime + "\n"
        str_return += "----Output Folder----\n"
        str_return += self.outputFolder + "\n"
        str_return += "----Operation Log----\n"
        str_return += self.oprLog + "\n"
        return str_return

    # print output string with timestamp
    def log(self, str_in):
        str_out = pd.datetime.now().strftime('%Y-%m-%d %H-%M-%S :')
        str_out += str_in.replace(self.outputFolder, self.relativeOutputFolder)
        self.oprLog += str_out + "\n"
        print(str_out)

    @staticmethod
    def gen_date_by_dows(start_date, dow_series):
        # 根据DOW和初始日期，创建日期序列函数
        # start_date：起始日期
        day_delta = []
        last_day = start_date

        for i in range(len(dow_series)):
            day_delta.append(timedelta(dow_series[i]) + last_day)
            last_day = day_delta[i]
        return day_delta

