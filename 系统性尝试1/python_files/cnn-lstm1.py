import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, recall_score
import seaborn as sns
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 检查是否有可用的GPU并设置为默认设备
physical_devices = tf.config.list_physical_devices('GPU')
print("检测到GPU设备:", physical_devices)
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        tf.config.set_visible_devices(physical_devices[0], 'GPU')
        print("使用GPU进行训练")
    except Exception as e:
        print("配置GPU时出错:", e)
else:
    print("GPU不可用，使用CPU进行训练")

# 加载数据集
train_data = pd.read_excel('./dataset/train.xlsx')
valid_data = pd.read_excel('./dataset/valid.xlsx')
predict1_data = pd.read_excel('./dataset/predict1.xlsx')
predict2_data = pd.read_excel('./dataset/predict2.xlsx')
predict3_data = pd.read_excel('./dataset/predict3.xlsx')


# 数据预处理函数
def preprocess_data(data):
    # 删除重复的列名
    data = data.loc[:, ~data.columns.duplicated()]

    # 数据预处理
    data['Start_Time'] = pd.to_datetime(data['Start_Time'], dayfirst=True, format='%d-%m-%Y %H:%M')
    data['Last_Updated'] = pd.to_datetime(data['Last_Updated'], dayfirst=True, format='%d-%m-%Y %H:%M')
    data['Duration_in_Minutes'] = (data['Last_Updated'] - data['Start_Time']).dt.total_seconds() / 60  # 持续时间，以分钟为单位

    # 将严重性分为light, moderate, severe三个级别
    data['Severity'] = pd.cut(data['Duration_in_Minutes'], bins=[0, 10, 30, np.inf],
                              labels=['light', 'moderate', 'severe'])
    data['Severity'] = LabelEncoder().fit_transform(data['Severity'])

    # 编码分类特征
    label_encoders = {}
    categorical_features = ['Main_Category', 'Primary_Vehicle', 'Secondary_Vehicle', 'Attending_Groups', 'Display_Name',
                            'Direction', 'Main_Street', 'Suburb']
    for column in categorical_features:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column].astype(str))

    # 归一化数值特征
    scaler = MinMaxScaler()
    numerical_features = ['Longitude', 'Latitude', 'Duration_in_Minutes', 'Actual_Number_of_Lanes ']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # 定义特征和标签
    X = data[['Longitude', 'Latitude', 'Duration_in_Minutes', 'Main_Category', 'Primary_Vehicle', 'Secondary_Vehicle',
              'Attending_Groups', 'Direction', 'Main_Street', 'Actual_Number_of_Lanes ', 'Suburb']].values
    y_severity = data['Severity'].values
    y_time = data['Duration_in_Minutes'].values

    # 检查并处理超大值
    X = np.clip(X, -1e6, 1e6)
    # 打印数据统计信息
    print("特征数据统计信息:")
    print("最大值:", np.max(X))
    print("最小值:", np.min(X))
    print("平均值:", np.mean(X))

    return X, y_severity, y_time

    

# 预处理数据集
X_train, y_severity_train, y_time_train = preprocess_data(train_data)
X_valid, y_severity_valid, y_time_valid = preprocess_data(valid_data)
X_test1, y_severity_test1, y_time_test1 = preprocess_data(predict1_data)
X_test2, y_severity_test2, y_time_test2 = preprocess_data(predict2_data)
X_test3, y_severity_test3, y_time_test3 = preprocess_data(predict3_data)

# 重塑数据以适应CNN-LSTM模型
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_valid = X_valid.reshape(X_valid.shape[0], X_valid.shape[1], 1)
X_test1 = X_test1.reshape(X_test1.shape[0], X_test1.shape[1], 1)
X_test2 = X_test2.reshape(X_test2.shape[0], X_test2.shape[1], 1)
X_test3 = X_test3.reshape(X_test3.shape[0], X_test3.shape[1], 1)

# 定义用于预测严重性的CNN-LSTM模型
severity_model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(filters=64, kernel_size=2, activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01)),
    LSTM(100, kernel_regularizer=l2(0.01)),
    Flatten(),
    Dense(100, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(3, activation='softmax', kernel_regularizer=l2(0.01))  # 输出3个类别
])

# 定义带有学习率调度的SGD优化器
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=10000,
    decay_rate=0.9)
sgd = SGD(learning_rate=lr_schedule)

severity_model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 早停和模型检查点回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('./best_severity_model.h5', monitor='val_loss', save_best_only=True)

# 训练严重性模型
severity_history = severity_model.fit(X_train, y_severity_train, epochs=100, batch_size=64,
                                      validation_data=(X_valid, y_severity_valid),
                                      callbacks=[early_stopping, checkpoint])

# 预测严重性
y_severity_pred1 = np.argmax(severity_model.predict(X_test1), axis=1)
y_severity_pred2 = np.argmax(severity_model.predict(X_test2), axis=1)
y_severity_pred3 = np.argmax(severity_model.predict(X_test3), axis=1)


# 评估严重性模型
def evaluate_model(y_true, y_pred, title):
    print(f"{title} 评估")
    print("混淆矩阵:")
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)

    print("分类报告:")
    class_report = classification_report(y_true, y_pred, target_names=['light', 'moderate', 'severe'])
    print(class_report)

    print("准确率:", accuracy_score(y_true, y_pred))
    print("召回率:", recall_score(y_true, y_pred, average='weighted'))

    # 绘制混淆矩阵图
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', xticklabels=['light', 'moderate', 'severe'],
                yticklabels=['light', 'moderate', 'severe'])
    plt.title(f'{title} 混淆矩阵')
    plt.xlabel('预测')
    plt.ylabel('实际')
    plt.savefig(f'./img/{title}_confusion_matrix.png')
    plt.close()


# 绘制训练和验证的准确率和损失
def plot_metrics(history, title):
    plt.figure(figsize=(12, 5))

    # 绘制准确率图
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    plt.title(f'{title} 准确率')
    plt.savefig(f'./img/{title}_accuracy.png')

    # 绘制损失图
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    plt.title(f'{title} 损失')
    plt.savefig(f'./img/{title}_loss.png')
    plt.close()


# 在所有测试集上评估
evaluate_model(y_severity_test1, y_severity_pred1, 'Test_Set_1')
evaluate_model(y_severity_test2, y_severity_pred2, 'Test_Set_2')
evaluate_model(y_severity_test3, y_severity_pred3, 'Test_Set_3')

# 绘制严重性模型的准确率和损失
plot_metrics(severity_history, 'Severity_Model')

# 定义用于预测持续时间的CNN-LSTM模型
duration_model = Sequential([
    Input(shape=(X_train.shape[1], 1)),
    Conv1D(filters=64, kernel_size=2, activation='relu', kernel_regularizer=l2(0.01)),
    MaxPooling1D(pool_size=2),
    LSTM(100, return_sequences=True, kernel_regularizer=l2(0.01)),
    LSTM(100, kernel_regularizer=l2(0.01)),
    Flatten(),
    Dense(100, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.5),
    Dense(1, kernel_regularizer=l2(0.01))
])

duration_model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['mae'])

# 早停和模型检查点回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('./best_duration_model.h5', monitor='val_loss', save_best_only=True)

# 训练持续时间模型
duration_history = duration_model.fit(X_train, y_time_train, epochs=100, batch_size=64,
                                      validation_data=(X_valid, y_time_valid), callbacks=[early_stopping, checkpoint])

# 预测持续时间
y_time_pred1 = duration_model.predict(X_test1)
y_time_pred2 = duration_model.predict(X_test2)
y_time_pred3 = duration_model.predict(X_test3)


# 评估持续时间模型
def evaluate_duration_model(y_true, y_pred, title):
    mae = np.mean(np.abs(y_true - y_pred))
    print(f"{title} 持续时间预测的平均绝对误差:", mae)

    # 绘制实际值和预测值的散点图
    plt.figure(figsize=(10, 7))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('实际值')
    plt.ylabel('预测值')
    plt.title(f'{title} 实际值 vs 预测值')
    plt.savefig(f'./img/{title}_actual_vs_pred.png')
    plt.close()


# 在所有测试集上评估
evaluate_duration_model(y_time_test1, y_time_pred1, 'Test_Set_1')
evaluate_duration_model(y_time_test2, y_time_pred2, 'Test_Set_2')
evaluate_duration_model(y_time_test3, y_time_pred3, 'Test_Set_3')

# 绘制持续时间模型的MAE和损失
plot_metrics(duration_history, 'Duration_Model')