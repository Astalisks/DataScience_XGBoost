import pandas as pd

class Preprocessing:

    def DataRead(self):

        dataPath = 'DataSets/SampleData.csv'
        dataFrame = pd.read_csv(dataPath)
        # print(dataFrame)
        # print(dataFrame.head(10))

        return dataFrame
