#!/usr/bin/env python3
"""
重构后的 MIDI benchmark 结果可视化代码，
修复了 DPI 设置、Log 轴下 0 值问题以及图例遮挡等 bug。
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# 统一输出图片 DPI 设置
OUTPUT_DPI = 600

# 库的顺序（用于颜色映射和排序）
LIB_ORDER = ['symusic', 'midifile_cpp', 'midi_jl', 'tonejs', 'miditoolkit', 'pretty_midi', 'partitura', 'music21']


def load_results(root_dir):
    """
    从指定目录中读取各库的读/写 benchmark 数据，并计算速度（单位 MB/s）。
    返回两个 DataFrame 分别对应 read_data 和 write_data。
    """
    # 读取所有 READ 数据
    read_data_list = []
    for lib in LIB_ORDER:
        path = os.path.join(root_dir, f'{lib}_read.csv')
        if os.path.exists(path):
            data = pd.read_csv(path)
            # 计算速度：文件大小（KB）/ 读时间（s） 转换为 MB/s
            data['Speed (MB/s)'] = data['File Size (KB)'] / data['Read Time (s)'] / 1024
            data['Library'] = lib
            read_data_list.append(data)
    if len(read_data_list) == 0:
        raise ValueError("未找到任何 read 数据！")
    read_data = pd.concat(read_data_list, ignore_index=True)

    # 读取所有 WRITE 数据
    write_data_list = []
    for lib in LIB_ORDER:
        path = os.path.join(root_dir, f'{lib}_write.csv')
        if os.path.exists(path):
            data = pd.read_csv(path)
            data['Speed (MB/s)'] = data['File Size (KB)'] / data['Write Time (s)'] / 1024
            data['Library'] = lib
            write_data_list.append(data)
    if len(write_data_list) == 0:
        raise ValueError("未找到任何 write 数据！")
    write_data = pd.concat(write_data_list, ignore_index=True)

    return read_data, write_data


def compute_label_positions(avg_dict, min_gap_log=0.05):
    """
    根据每个库的平均速度（单位 MB/s）计算文本标签的 y 坐标，
    为避免标签在 log10 坐标下重叠，至少保持 min_gap_log 的间隔。
    如果平均速度值为 0 或负数，则以 1e-3 代替进行计算。
    
    参数：
      - avg_dict: {库名: 平均速度}
      - min_gap_log: 在 log10 空间下相邻标签的最小间隔
    返回：
      - label_positions: {库名: 调整后的 y 坐标}
    """
    # 构造 (lib, avg, log10(avg)) 列表（对于 <=0 的速度，使用 1e-3 代替）
    items = [(lib, avg, np.log10(avg if avg > 0 else 1e-3)) for lib, avg in avg_dict.items()]
    # 按 log10(avg) 升序排序
    items.sort(key=lambda x: x[2])
    adjusted = []
    for i, (lib, avg, log_val) in enumerate(items):
        if i == 0:
            adjusted.append((lib, avg, log_val))
        else:
            # 保证相邻标签在 log10 空间中至少相隔 min_gap_log
            prev_lib, prev_avg, prev_log = adjusted[-1]
            if log_val - prev_log < min_gap_log:
                log_val = prev_log + min_gap_log
            adjusted.append((lib, avg, log_val))
    # 转换回原数据坐标
    label_positions = {lib: 10 ** log_val for lib, avg, log_val in adjusted}
    return label_positions


def plot_line_chart(data, title, filename, x_min, x_max, y_min, y_max, output_dir):
    """
    绘制折线图（使用 LOWESS 平滑曲线），标注各库的平均速度，
    并在 log y 轴下设置合适的刻度和图例（图例放到图像外侧）。
    """
    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("colorblind", n_colors=len(LIB_ORDER))

    # 计算各库平均速度（如果平均速度为 0 或负，则后续计算中以 1e-3 代替）
    avg_speed_dict = {}
    for i, lib in enumerate(LIB_ORDER):
        lib_data = data[data['Library'] == lib]
        if lib_data.empty:
            continue
        avg_speed = lib_data['Speed (MB/s)'].mean()
        avg_speed_dict[lib] = avg_speed

        # 按 x 轴排序后绘制 LOWESS 平滑曲线
        lib_data = lib_data.sort_values(by='File Size (KB)')
        sns.regplot(
            x='File Size (KB)',
            y='Speed (MB/s)',
            data=lib_data,
            lowess=True,
            scatter=False,
            label=lib,
            line_kws={'alpha': 0.6, 'color': palette[i], 'linestyle': '-'}
        )

    # 根据平均速度计算标签在 y 轴上的显示位置（避免重叠）
    label_positions = compute_label_positions(avg_speed_dict, min_gap_log=0.15)
    for i, lib in enumerate(LIB_ORDER):
        if lib in avg_speed_dict:
            avg_speed = avg_speed_dict[lib]
            text_y = label_positions[lib]
            plt.text(
                x=x_max * 1.005 if avg_speed > 100 else x_max * 1.01,
                y=text_y,
                s=f"{avg_speed:.2f}" if avg_speed > 1 else f"{avg_speed:.3f}",
                va='center',
                ha='left',
                color=palette[i],
                fontsize=12
            )

    plt.yscale('log')
    # 为避免 y_min 为 0（log 轴不能显示 0），若小于等于 0 则使用 1e-3
    y_min = y_min if y_min > 0 else 1e-3

    plt.xlim(x_min, x_max * 1.1)  # 为右侧文本标签预留空间
    plt.ylim(y_min, y_max)
    plt.xlabel("File Size (KB)", fontsize=14)
    plt.ylabel("Speed (MB/s)", fontsize=14)
    plt.title(title, fontsize=16)

    ax = plt.gca()
    plt.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    # 设置 log 轴的主刻度和格式（显示所有刻度而非仅仅底数刻度）
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))  # 只显示 10 的整数次幂
    ax.yaxis.set_minor_locator(ticker.NullLocator())  # 取消次要刻度
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))  # 显示指数形式

    # 将图例放置在图像外侧（右上角）
    plt.legend(bbox_to_anchor=(1.025, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=OUTPUT_DPI)
    plt.close()


def plot_results(read_data, write_data, output_dir):
    """
    绘制读/写速度-文件大小折线图。
    计算两个图表共同的横轴和纵轴范围，并分别绘制图表。
    """
    # 计算横轴范围（File Size）
    x_min = min(read_data['File Size (KB)'].min(), write_data['File Size (KB)'].min())
    x_max = max(read_data['File Size (KB)'].max(), write_data['File Size (KB)'].max())
    # 计算纵轴范围（Speed），同时确保 y_min > 0（log scale 不能显示 0）
    y_min = min(read_data['Speed (MB/s)'].min(), write_data['Speed (MB/s)'].min())
    y_max = max(read_data['Speed (MB/s)'].max(), write_data['Speed (MB/s)'].max())
    y_min = y_min if y_min > 0 else 1e-3

    # 设置主题：paper 上下文、白色网格和 colorblind 调色板
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.3)

    # 绘制 READ Speed 折线图
    plot_line_chart(
        data=read_data,
        title="MIDI Parsing Benchmark (Mean Annotated)",
        filename="parse_speed_line.png",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        output_dir=output_dir,
    )

    # 绘制 WRITE Speed 折线图
    plot_line_chart(
        data=write_data,
        title="MIDI Dumping Benchmark (Mean Annotated)",
        filename="dump_speed_line.png",
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        output_dir=output_dir,
    )


def plot_violin_chart(data, title, filename, output_dir):
    """
    绘制小提琴图，显示各库的速度分布（过滤离群点后），并在图中标注中位数。
    为了保证同一库颜色一致，按 LIB_ORDER 固定顺序映射颜色。
    """
    # 筛选当前数据中实际存在的库，并按照 LIB_ORDER 排序
    present_libs = set(data['Library'])
    order = [lib for lib in LIB_ORDER if lib in present_libs]
    n_lib = len(order)
    # 根据库数量动态调整图形宽度（每个库约 1.5 英寸，最小 6 英寸）
    fig_width = max(6, n_lib * 1.5)
    plt.figure(figsize=(fig_width, 6))

    # 固定调色板映射：库 -> 颜色
    palette = sns.color_palette("colorblind", n_colors=len(LIB_ORDER))
    color_dict = {lib: palette[i] for i, lib in enumerate(LIB_ORDER)}

    ax = sns.violinplot(
        x="Library",
        y="Speed (MB/s)",
        hue="Library",
        data=data,
        order=order,
        palette=color_dict,
        dodge=False,
    )

    # 计算各库中位数并在图上标注（这里将标签置于中位数上方 5% 处）
    medians = data.groupby('Library')['Speed (MB/s)'].median().to_dict()
    for i, lib in enumerate(order):
        median_val = medians.get(lib)
        if median_val is not None:
            plt.text(
                x=i + 0.2,
                y=median_val * 1.15,
                s=f"{median_val:.2f}" if median_val > 1 else f"{median_val:.3f}",
                ha='center',
                va='bottom',
                fontsize=9,
                color=color_dict[lib]
            )

    # 设置每个小提琴的透明度
    for coll in ax.collections:
        coll.set_alpha(0.65)
    
    plt.minorticks_on()
    plt.yscale('log')
    # 设置 log 轴的主刻度和格式（显示所有刻度而非仅仅底数刻度）
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0))  # 只显示 10 的整数次幂
    ax.yaxis.set_minor_locator(ticker.NullLocator())  # 取消次要刻度
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))  # 显示指数形式
    plt.xlabel("Library", fontsize=14)
    plt.ylabel("Speed (MB/s)", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(output_dir, filename)
    plt.savefig(out_path, dpi=OUTPUT_DPI)
    plt.close()


def plot_speed_violin(read_data, write_data, output_dir):
    """
    绘制去除（或过滤）离群点后的速度分布小提琴图，
    分别针对 read 和 write 数据，每个库一个小提琴图，颜色与折线图保持一致。
    """
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.3)

    # 绘制 READ Speed 小提琴图
    plot_violin_chart(
        data=read_data,
        title="MIDI Parsing Speed Distribution (Filtered Outliers)\n(Median Annotated)",
        filename="parse_speed_violin.png",
        output_dir=output_dir,
    )

    # 绘制 WRITE Speed 小提琴图
    plot_violin_chart(
        data=write_data,
        title="MIDI Dumping Speed Distribution (Filtered Outliers)\n(Median Annotated)",
        filename="dump_speed_violin.png",
        output_dir=output_dir,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualize the results of the MIDI benchmark")
    parser.add_argument('--input', type=str, required=True,
                        help='Directory containing the benchmark CSV results')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory to save the generated figures')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 读取数据
    rd, wd = load_results(args.input)

    # 绘制文件大小-速度折线图
    plot_results(rd, wd, args.output)

    # 绘制速度分布小提琴图（过滤离群点，并标注中位数）
    plot_speed_violin(rd, wd, args.output)
