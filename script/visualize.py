import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
import matplotlib.ticker as ticker
import numpy as np

LIB_ORDER = ['symusic', 'midifile_cpp', 'miditoolkit', 'pretty_midi', 'partitura', 'music21']

def load_results(root_dir):
    # 读取所有 READ 数据
    read_data = []
    for lib in LIB_ORDER:
        path = os.path.join(root_dir, f'{lib}_read.csv')
        if os.path.exists(path):
            data = pd.read_csv(path)
            data['Speed (MB/s)'] = data['File Size (KB)'] / data['Read Time (s)'] / 1024
            data['Library'] = lib
            read_data.append(data)
    read_data = pd.concat(read_data)

    # 读取所有 WRITE 数据
    write_data = []
    for lib in LIB_ORDER:
        path = os.path.join(root_dir, f'{lib}_write.csv')
        if os.path.exists(path):
            data = pd.read_csv(path)
            data['Speed (MB/s)'] = data['File Size (KB)'] / data['Write Time (s)'] / 1024
            data['Library'] = lib
            write_data.append(data)
    write_data = pd.concat(write_data)
    return read_data, write_data


def compute_label_positions(avg_dict, min_gap_log=0.05):
    """
    根据每个库的平均速度（数据值）计算文本标签的 y 坐标，
    采用 log10 坐标下的最小间隔 min_gap_log 来避免重叠。
    返回一个字典，键为库名，值为调整后的 y 坐标（数据坐标）。
    """
    # 构造 (lib, avg_speed, log10(avg_speed)) 列表，并按 log10(avg_speed) 升序排序
    items = [(lib, avg, np.log10(avg)) for lib, avg in avg_dict.items()]
    items.sort(key=lambda x: x[2])
    adjusted = []
    for i, (lib, avg, log_val) in enumerate(items):
        if i == 0:
            adjusted.append((lib, avg, log_val))
        else:
            # 保证相邻标签在 log10 空间至少有 min_gap_log 的间隔
            prev_lib, prev_avg, prev_log = adjusted[-1]
            if log_val - prev_log < min_gap_log:
                log_val = prev_log + min_gap_log
            adjusted.append((lib, avg, log_val))
    # 转回原坐标（10**(log_val)）
    label_positions = {lib: 10 ** log_val for lib, avg, log_val in adjusted}
    return label_positions


def plot_results(read_data, write_data, output_dir):
    # 确定两个图表共同的横纵坐标范围
    x_min = min(read_data['File Size (KB)'].min(), write_data['File Size (KB)'].min())
    x_max = max(read_data['File Size (KB)'].max(), write_data['File Size (KB)'].max())
    y_min = min(read_data['Speed (MB/s)'].min(), write_data['Speed (MB/s)'].min())
    y_max = max(read_data['Speed (MB/s)'].max(), write_data['Speed (MB/s)'].max())

    os.makedirs(output_dir, exist_ok=True)

    # 设置主题：paper 上下文、白色网格和 colorblind 调色板（学术风格），并放大字体
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.3)

    ######################
    # 绘制 READ Speed 图表
    ######################
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("colorblind", n_colors=len(LIB_ORDER))

    # 计算各库平均速度（仅对有数据的库）
    avg_speed_dict = {}
    for i, lib in enumerate(LIB_ORDER):
        lib_data = read_data[read_data['Library'] == lib]
        if lib_data.empty:
            continue
        avg_speed = lib_data['Speed (MB/s)'].mean()
        avg_speed_dict[lib] = avg_speed

    # 根据平均速度计算文本标签的 y 坐标，避免重叠（使用 log10 坐标调整）
    label_positions = compute_label_positions(avg_speed_dict, min_gap_log=0.15)

    # 绘制各个库的 LOWESS 平滑曲线（全部采用实线）
    for i, lib in enumerate(LIB_ORDER):
        lib_data = read_data[read_data['Library'] == lib]
        if lib_data.empty:
            continue
        lib_data = lib_data.sort_values(by='File Size (KB)')
        sns.regplot(
            x='File Size (KB)',
            y='Speed (MB/s)',
            data=lib_data,
            lowess=True,
            scatter=False,
            label=lib,
            line_kws={
                'alpha': 0.6,
                'color': palette[i],
                'linestyle': '-'
            }
        )

    # 添加平均速度文本标签（无背景）
    for i, lib in enumerate(LIB_ORDER):
        if lib in avg_speed_dict:
            avg_speed = avg_speed_dict[lib]
            text_y = label_positions[lib]
            plt.text(
                x_max * 1.01,
                text_y,
                f"{avg_speed:.2f}",
                va='center',
                ha='left',
                color=palette[i],
                fontsize=12
            )

    plt.yscale('log')
    plt.xlim(x_min, x_max * 1.1)  # 为文本标签留出空间
    plt.ylim(y_min, y_max)
    plt.xlabel("File Size (KB)", fontsize=14)
    plt.ylabel("Speed (MB/s)", fontsize=14)
    plt.title("MIDI Parsing Benchmark (Mean Annotated)", fontsize=16)

    ax = plt.gca()
    plt.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    if ax.get_yscale() == 'log':
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10)))
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parse_speed_line.png"), dpi=600)
    plt.close()

    ######################
    # 绘制 WRITE Speed 图表
    ######################
    plt.figure(figsize=(8, 6))
    palette = sns.color_palette("colorblind", n_colors=len(LIB_ORDER))

    avg_speed_dict = {}
    for i, lib in enumerate(LIB_ORDER):
        lib_data = write_data[write_data['Library'] == lib]
        if lib_data.empty:
            continue
        avg_speed = lib_data['Speed (MB/s)'].mean()
        avg_speed_dict[lib] = avg_speed

    label_positions = compute_label_positions(avg_speed_dict, min_gap_log=0.15)

    for i, lib in enumerate(LIB_ORDER):
        lib_data = write_data[write_data['Library'] == lib]
        if lib_data.empty:
            continue
        lib_data = lib_data.sort_values(by='File Size (KB)')
        sns.regplot(
            x='File Size (KB)',
            y='Speed (MB/s)',
            data=lib_data,
            lowess=True,
            scatter=False,
            label=lib,
            line_kws={
                'alpha': 0.6,
                'color': palette[i],
                'linestyle': '-'
            }
        )

    for i, lib in enumerate(LIB_ORDER):
        if lib in avg_speed_dict:
            avg_speed = avg_speed_dict[lib]
            text_y = label_positions[lib]
            plt.text(
                x_max * 1.01,
                text_y,
                f"{avg_speed:.2f}",
                va='center',
                ha='left',
                color=palette[i],
                fontsize=12
            )

    plt.yscale('log')
    plt.xlim(x_min, x_max * 1.1)
    plt.ylim(y_min, y_max)
    plt.xlabel("File Size (KB)", fontsize=14)
    plt.ylabel("Speed (MB/s)", fontsize=14)
    plt.title("MIDI Dumping Benchmark (Mean Annotated)", fontsize=16)

    ax = plt.gca()
    plt.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MaxNLocator(10))
    if ax.get_yscale() == 'log':
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(1, 10)))
    else:
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(10))

    plt.legend(loc='best', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dump_speed_line.png"), dpi=600)
    plt.close()


def plot_speed_violin(read_data, write_data, output_dir):
    """
    绘制与文件大小无关的速度分布小提琴图（分读写两张图），
    每个库一个小提琴分布，使用与折线图相同的调色板，保证颜色一致。
    """
    os.makedirs(output_dir, exist_ok=True)
    # 与折线图保持一致的主题/调色板
    sns.set_theme(context="paper", style="whitegrid", palette="colorblind", font_scale=1.3)

    # 为了让同一库的颜色一致，手动映射库 -> palette
    palette = sns.color_palette("colorblind", n_colors=len(LIB_ORDER))
    color_dict = {lib: palette[i] for i, lib in enumerate(LIB_ORDER)}

    libset = set(read_data['Library'])
    order = [lib for lib in LIB_ORDER if lib in libset]

    # === READ speed violin plot ===
    plt.figure(figsize=(6, 6))
    ax = sns.violinplot(
        x="Library",
        y="Speed (MB/s)",
        hue="Library",
        data=read_data,
        order=order,
        palette=color_dict,
        dodge=False,
    )
    # 分组计算中位数
    grouped = read_data.groupby('Library')['Speed (MB/s)'].median().to_dict()

    for i, lib in enumerate(order):
        median_val = grouped[lib]
        # 在 y 轴为 median_val 处标注文本
        # x坐标为 i，表示第 i 个分类的中心，或略作调整
        plt.text(
            x=i + 0.3,
            y=median_val * 1.2,
            s=f"{median_val:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color=color_dict[lib]
        )
    for violin in ax.collections:
        violin.set_alpha(0.65)

    plt.yscale('log')
    from matplotlib.ticker import LogLocator, ScalarFormatter
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(ScalarFormatter())

    plt.xlabel("Library", fontsize=14)
    plt.ylabel("Speed (MB/s)", fontsize=14)
    plt.title("MIDI Parsing Speed Distribution\n(Median Annotated)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "parse_speed_violin.png"), dpi=600)
    plt.close()

    # === WRITE speed violin plot ===
    plt.figure(figsize=(6, 6))
    ax = sns.violinplot(
        x="Library",
        y="Speed (MB/s)",
        hue="Library",
        data=write_data,
        order=order,
        palette=color_dict,
        dodge=False
    )

    grouped = write_data.groupby('Library')['Speed (MB/s)'].median().to_dict()

    for i, lib in enumerate(order):
        median_val = grouped[lib]
        # 在 y 轴为 median_val 处标注文本
        # x坐标为 i，表示第 i 个分类的中心，或略作调整
        plt.text(
            x=i + 0.3,
            y=median_val * 1.2,
            s=f"{median_val:.2f}",
            ha='center',
            va='bottom',
            fontsize=9,
            color=color_dict[lib]
        )

    for violin in ax.collections:
        violin.set_alpha(0.65)

    plt.yscale('log')
    plt.xlabel("Library", fontsize=14)
    plt.ylabel("Speed (MB/s)", fontsize=14)
    plt.title("MIDI Dumping Speed Distribution\n(Median Annotated)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dump_speed_violin.png"), dpi=600)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Visualize the results of the benchmark")
    parser.add_argument('--input', type=str, help='The directory containing the benchmark results')
    parser.add_argument('--output', type=str, help='The directory to save the figures')
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    # 读取数据
    rd, wd = load_results(args.input)

    # 画速度-文件大小折线图
    plot_results(rd, wd, args.output)

    # 额外画小提琴分布图（与文件大小无关）
    plot_speed_violin(rd, wd, args.output)