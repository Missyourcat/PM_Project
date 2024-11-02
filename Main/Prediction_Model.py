import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import FactorAnalysis
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity, FactorAnalyzer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 读取数据
SQ_TenYears_Data = pd.read_excel('../Data/商丘PM2.5研究-逐月(2013-2023)(完整数据).xlsx', engine='openpyxl')
Research_Target = SQ_TenYears_Data.iloc[:, 3:13]

# 提取目标值 Y 和输入特征 X
Y = Research_Target.iloc[:, 0].values
X = Research_Target.iloc[:, 1:]

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMO 和 Bartlett's 检验
kmo_all, kmo_model = calculate_kmo(X_scaled)
chi_square_value, p_value = calculate_bartlett_sphericity(X_scaled)

if kmo_model < 0.6 or p_value > 0.05:
    print("数据可能不适合进行因子分析。")
else:
    # 因子分析
    fa = FactorAnalyzer(rotation='varimax', n_factors=3)
    fa.fit(X_scaled)
    scores = fa.transform(X_scaled)  # 因子得分

    # 数据标准化
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_Y = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(scores)  # 使用因子得分作为新特征
    Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, 1))

    # 创建时间序列数据
    def create_sequences(data, target, time_steps=2):
        Xs, ys = [], []
        for i in range(len(data) - time_steps):
            v = data[i:(i + time_steps)]
            Xs.append(v)
            ys.append(target[i + time_steps])
        return np.array(Xs), np.array(ys)

    time_steps = 10
    X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, time_steps)

    # 划分训练集和测试集
    split = int(0.7 * len(X_seq))
    X_train, X_test = X_seq[:split], X_seq[split:]
    Y_train, Y_test = Y_seq[:split], Y_seq[split:]

    # 创建 LSTM 模型
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=150, return_sequences=True))  # 增加 LSTM 单元数
    model.add(BatchNormalization())
    model.add(Dropout(0.1))  # 调整 Dropout
    model.add(LSTM(units=150, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))  # 调整 Dropout
    model.add(LSTM(units=150))
    model.add(BatchNormalization())
    model.add(Dropout(0.1))  # 调整 Dropout
    model.add(Dense(1))

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')  # 调整学习率

    # 定义回调函数
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

    # 训练模型
    history = model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.2,  # 增加 epochs
                        callbacks=[early_stopping, reduce_lr], verbose=1)

    # 多步预测
    future_steps = 2
    last_sequence = X_test[-1]
    predictions = []

    for _ in range(future_steps):
        next_prediction = model.predict(last_sequence.reshape(1, time_steps, X_train.shape[2]))
        predictions.append(next_prediction[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, 0] = next_prediction  # 更新第一个特征为预测值

    # 反标准化预测结果
    predictions = scaler_Y.inverse_transform(np.array(predictions).reshape(-1, 1))

    # 计算均方误差 (MSE)
    mse_value = np.mean((scaler_Y.inverse_transform(Y_scaled[-future_steps:]) - predictions) ** 2)

    # 显示结果
    print('预测结果:')
    print(predictions.flatten())
    print(f'均方误差 (MSE): {mse_value}')

    # 绘制预测结果与实际结果的对比图
    plt.figure(figsize=(12, 6))
    start_index = len(Y) - 100  # 获取最后100条数据的起始索引
    plt.plot(range(start_index, len(Y)), scaler_Y.inverse_transform(Y_scaled)[start_index:], label='实际值',
             color='blue')
    plt.plot(range(len(Y), len(Y) + future_steps), predictions, label='预测值', color='red', linestyle='--')
    plt.legend()
    plt.xlabel('时间步')
    plt.ylabel('值')
    plt.title('实际值与预测值对比（最后100条）')
    plt.grid(True)
    plt.show()

    # 绘制训练和验证损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='训练损失', color='blue')
    plt.plot(history.history['val_loss'], label='验证损失', color='red')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.grid(True)
    plt.show()
