"""
Benchmark 脚本：使用 MIDI.jl 测试 MIDI 文件的读写性能

使用方法（命令行）示例：
    julia midijl_benchmark.jl --dataset_root /path/to/midi_dataset \
                               --dataset_config midi_files.json \
                               --repeat 4 \
                               --output ./results \
                               --verbose

参数说明：
- --dataset_root      数据集根目录
- --dataset_config    JSON 文件，里面包含 MIDI 文件的相对路径列表
- --repeat            每个文件重复测试次数（默认4次）
- --output            保存结果的输出目录（默认 "./results"）
- --verbose           是否输出详细信息
"""

using ArgParse
using JSON
using CSV
using DataFrames
using MIDI       # 请确保已安装 MIDI.jl 包，并且其提供 parse/write 接口
using Printf

# 主函数
function main()
    # 定义命令行参数
    s = ArgParseSettings()
    @add_arg_table s begin
        "--repeat"
            help = "重复测试次数（默认：4）"
            default = 4
            arg_type = Int
        "--dataset_root"
            help = "数据集根目录"
            required = true
        "--dataset_config"
            help = "包含 MIDI 文件相对路径列表的 JSON 文件"
            required = true
        "--output"
            help = "结果输出目录（默认：./results）"
            default = "./results"
        "--verbose"
            help = "是否输出详细信息"
            action = :store_true
    end

    args = parse_args(s)

    repeat = args["repeat"]
    dataset_root = args["dataset_root"]
    dataset_config = args["dataset_config"]
    output_dir = args["output"]
    verbose = get(args, "verbose", false)

    # 读取 JSON 配置文件，构建完整的 MIDI 文件路径列表
    file_paths = String[]
    open(dataset_config, "r") do io
        rel_paths = JSON.parse(io)
        for rel in rel_paths
            push!(file_paths, joinpath(dataset_root, rel))
        end
    end

    # 预热：选取第一个文件进行一次读写操作
    if !isempty(file_paths)
        warmup_file = file_paths[1]
        try
            if verbose
                println("开始预热...")
            end
            midi_obj = MIDI.load(warmup_file)
            temp_warmup = "temp_warmup.mid"
            MIDI.save(temp_warmup, midi_obj)
            if isfile(temp_warmup)
                rm(temp_warmup)
            end
            if verbose
                println("预热完成。")
            end
        catch e
            @warn("预热时出错：$e")
        end
    end

    # 用于保存每个文件的文件大小（KB）、平均读取时间和平均写入时间
    sizes = Float64[]
    read_times = Float64[]
    write_times = Float64[]

    # 对每个文件进行测试
    for file in file_paths
        try
            file_size = stat(file).size / 1024  # 单位：KB
            if file_size < 5
                if verbose
                    @printf("跳过 %s （大小 %.2f KB 太小）\n", file, file_size)
                end
                continue
            end
            midi_obj = load(file)
            # 测量读取时间
            total_read_time = 0.0
            midi_obj = nothing
            for i in 1:repeat
                t0 = time_ns()
                midi_obj = MIDI.load(file)
                t1 = time_ns()
                total_read_time += (t1 - t0) / 1e9
            end
            avg_read_time = total_read_time / repeat

            # 测量写入时间
            total_write_time = 0.0
            for i in 1:repeat
                temp_file = "temp_$(basename(file))_$(i).mid"
                t0 = time_ns()
                save(temp_file, midi_obj)
                t1 = time_ns()
                total_write_time += (t1 - t0) / 1e9
                # 删除临时文件
                if isfile(temp_file)
                    rm(temp_file)
                end
            end
            avg_write_time = total_write_time / repeat

            push!(sizes, file_size)
            push!(read_times, avg_read_time)
            push!(write_times, avg_write_time)

            if verbose
                @printf("处理 %s: 大小 = %.2f KB, 读取 = %.4f s, 写入 = %.4f s\n",
                        file, file_size, avg_read_time, avg_write_time)
            end

        catch e
            @warn("处理 $file 时出错：$e")
        end
    end

    # 根据 dataset_config 文件的文件名（不含后缀）创建输出目录
    dataset_stem = splitext(basename(dataset_config))[1]
    final_output_dir = joinpath(output_dir, dataset_stem)
    mkpath(final_output_dir)

    # 保存读取性能数据到 CSV
    read_df = DataFrame("File Size (KB)" => sizes, "Read Time (s)" => read_times)
    read_csv_file = joinpath(final_output_dir, "midi_jl_read.csv")
    CSV.write(read_csv_file, read_df)

    # 保存写入性能数据到 CSV
    write_df = DataFrame("File Size (KB)" => sizes, "Write Time (s)" => write_times)
    write_csv_file = joinpath(final_output_dir, "midi_jl_write.csv")
    CSV.write(write_csv_file, write_df)

    println("测试结果已保存至 $(final_output_dir)")
end

# 执行主函数
main()
