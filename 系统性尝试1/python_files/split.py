import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 加载数据集
data = pd.read_excel('Final Road Traffic Incidents.xlsx')

# 删除重复的列名
data = data.loc[:, ~data.columns.duplicated()]

# 创建目录以保存分割后的数据集
os.makedirs('./dataset', exist_ok=True)

# 划分数据集
train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
valid_data, predict_data = train_test_split(temp_data, test_size=1/6, random_state=42)
predict1_data, predict_temp = train_test_split(predict_data, test_size=0.5, random_state=42)
predict2_data, predict3_data = train_test_split(predict_temp, test_size=0.5, random_state=42)

# 保存数据集
train_data.to_excel('./dataset/train.xlsx', index=False)
valid_data.to_excel('./dataset/valid.xlsx', index=False)
predict1_data.to_excel('./dataset/predict1.xlsx', index=False)
predict2_data.to_excel('./dataset/predict2.xlsx', index=False)
predict3_data.to_excel('./dataset/predict3.xlsx', index=False)

print("数据集已成功划分并保存到./dataset目录中")