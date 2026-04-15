import pandas as pd
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
import os
import yaml

# 读取配置文件
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config['data']['raw_data_file'])

if not os.path.exists('img'):     
    os.makedirs('img') 

# 使用 with 语句打开 CSV 文件
with open('lightcurve_data.csv', 'w') as csv_file:
    # 写入 CSV 文件的列名
    csv_file.write('#oid,period,num_peaks,classALeRCE\n')

    for i in range(len(df)):
        time = np.array(df['hmjd'][i].strip('[]').split(','), dtype=float)
        flux = np.array(df['mag'][i].strip('[]').split(','), dtype=float)

        # 进行 Lomb-Scargle 周期搜索，限制周期范围 
        lc = lk.LightCurve(time=time, flux=flux)
        pg = lc.to_periodogram(method='lombscargle', minimum_frequency=0.1, maximum_frequency=24)  
        peri_arr = pg.period 
        # freq_arr = pg.frequency 
        # power_arr = pg.power 
        period_at_max_power = pg.period_at_max_power 
        period = period_at_max_power.value  
        # 折叠光变曲线
        folded_lc = lc.fold(period) 
        tim, flx = folded_lc.time.value, folded_lc.flux

        # 直接计算相位，避免多次复制数组
        phase = tim / period
        phase_extended = np.concatenate([phase, phase + 1, phase + 2])
        flux_extended = np.tile(flx, 3)

        # 创建图形和坐标轴
        plt.figure(figsize=(10, 6))

        # 绘制光变曲线，点线结合
        plt.plot(phase_extended, flux_extended, marker='o', linestyle='-', color='#4169E1', markersize=5, label='Light Curve')

        # 设置坐标轴标签和标题
        plt.ylabel('R-Magnitude', fontsize=12)
        plt.xlabel('MJD', fontsize=12)
        plt.title(f"Folded Light Curve for {df['#oid'][i]}", fontsize=14)  # 修改此处

        # 设置坐标轴刻度字体大小
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)

        # 显示图例
        plt.legend(fontsize=10)

        # 显示网格线，调整网格线样式
        plt.grid(True, linestyle='--', alpha=0.7)

        # 调整图形边框
        ax = plt.gca()
        ax.spines['top'].set_linewidth(1.2)
        ax.spines['right'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.spines['left'].set_linewidth(1.2)

        # 保存图片
        filename = os.path.join('img', f"{df['#oid'][i]}.png")
        plt.savefig(filename)

        plt.close()

        # 峰值数量特征
        peaks = pg.power > (0.5 * pg.max_power)
        num_peaks = np.sum(peaks)

        # 逐行写入 CSV 文件
        csv_file.write(f"{df['#oid'][i]},{period},{num_peaks},{df['classALeRCE'][i]}\n")