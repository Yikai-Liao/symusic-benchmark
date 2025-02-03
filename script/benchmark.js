const fs = require('fs');
const path = require('path');
const { Midi } = require('@tonejs/midi');
const { performance } = require('perf_hooks');
const { program } = require('commander');
const csvWriterLib = require('csv-writer').createObjectCsvWriter;

// 添加 --tqdm 命令行选项控制是否显示进度条（默认为 false）
program
  .requiredOption('--dataset-root <path>', 'Root directory of the dataset')
  .requiredOption('--dataset-config <path>', 'JSON file containing MIDI file list')
  .option('--repeat <number>', 'Number of repetitions', (v) => parseInt(v), 4)
  .option('--output-dir <path>', 'Output directory', './results')
  .option('--tqdm', 'Show progress bar', false)
  .parse(process.argv);

const options = program.opts();

// 预热函数：预先调用解析和写入，确保 JIT 编译完成
function warmup(fileBuffer) {
  // 预热读取和写入
  for (let i = 0; i < 3; i++) {
    new Midi(fileBuffer);
  }
  const tempMidi = new Midi(fileBuffer);
  for (let i = 0; i < 3; i++) {
    Buffer.from(tempMidi.toArray());
  }
}

async function benchmarkToneJS() {
  try {
    const relPaths = JSON.parse(fs.readFileSync(options.datasetConfig));
    const midiFiles = relPaths.map(p => path.join(options.datasetRoot, p));
    
    // 全局预热：如果存在至少一个文件，取第一个有效文件进行预热
    let warmupDone = false;
    for (const filePath of midiFiles) {
      if (fs.existsSync(filePath)) {
        const stats = fs.statSync(filePath);
        if (stats.size >= 5 * 1024) {
          const fileBuffer = fs.readFileSync(filePath);
          warmup(fileBuffer);
          console.log(`Global warmup done using: ${filePath}`);
          warmupDone = true;
          break;
        }
      }
    }
    if (!warmupDone) {
      console.warn('No valid file found for warmup.');
    }
    
    const results = [];
    let progressBar;
    // 如果开启 tqdm，则创建进度条
    if (options.tqdm) {
      const cliProgress = require('cli-progress');
      progressBar = new cliProgress.SingleBar({
        format: 'Benchmarking [{bar}] {percentage}% | {value}/{total} files',
        hideCursor: true
      }, cliProgress.Presets.shades_classic);
      progressBar.start(midiFiles.length, 0);
    }
    
    for (const filePath of midiFiles) {
      global.gc();
      try {
        // 检查文件是否存在
        if (!fs.existsSync(filePath)) {
          console.log(`File not found: ${filePath}`);
          if (options.tqdm) progressBar.increment();
          continue;
        }

        const stats = fs.statSync(filePath);
        if (stats.size < 5 * 1024) {
          console.log(`Skipping small file: ${filePath}`);
          if (options.tqdm) progressBar.increment();
          continue;
        }

        const fileBuffer = fs.readFileSync(filePath);

        // 读取基准测试：测量多次解析时间（单位为毫秒），最后转换为秒
        const readTimes = [];
        for (let i = 0; i < options.repeat; i++) {
          const start = performance.now();
          new Midi(fileBuffer); // 解析但不保留引用
          readTimes.push(performance.now() - start);
        }
        const avgReadTimeSec = (readTimes.reduce((a, b) => a + b, 0) / options.repeat) / 1000;
        
        // 写入基准测试：先构造 MIDI 对象，然后多次生成二进制数据和写入磁盘
        const writeTimes = [];
        const originalMidi = new Midi(fileBuffer);
        for (let i = 0; i < options.repeat; i++) {
          const tempPath = `temp_${path.basename(filePath)}_${i}.mid`;
          const start = performance.now();
          // 生成二进制数据
          const outputBuffer = originalMidi.toArray();
          // 实际写入操作
          fs.writeFileSync(tempPath, Buffer.from(outputBuffer));
          writeTimes.push(performance.now() - start);
          fs.unlinkSync(tempPath);
        }
        const avgWriteTimeSec = (writeTimes.reduce((a, b) => a + b, 0) / options.repeat) / 1000;
        
        results.push({
          file: filePath,
          fileSizeKB: (stats.size / 1024).toFixed(2),
          avgReadTimeSec,
          avgWriteTimeSec
        });
        
      } catch (error) {
        console.error(`Error processing ${filePath}:`, error.message);
      }
      // 更新进度条
      if (options.tqdm) progressBar.increment();
    }
    
    if (options.tqdm) progressBar.stop();
    saveResults(results);
    
  } catch (error) {
    console.error('Benchmark failed:', error.message);
    process.exit(1);
  }
}

function saveResults(results) {
  const outputDir = path.join(options.outputDir, path.parse(options.datasetConfig).name);
  fs.mkdirSync(outputDir, { recursive: true });
  
  // 构建读取性能数据，并保存到一个 CSV 文件
  const csvDataRead = results.map(result => ({
    file: result.file,
    fileSizeKB: result.fileSizeKB,
    readTimeSec: result.avgReadTimeSec
  }));
  
  const csvWriterRead = csvWriterLib({
    path: path.join(outputDir, 'tonejs_benchmark_read.csv'),
    header: [
      { id: 'fileSizeKB', title: 'File Size (KB)' },
      { id: 'readTimeSec', title: 'Read Time (s)' }
    ]
  });
  
  csvWriterRead.writeRecords(csvDataRead)
    .then(() => console.log(`Read results saved to ${path.join(outputDir, 'tonejs_read.csv')}`))
    .catch(err => console.error('Error writing read CSV:', err));
    
  // 构建写入性能数据，并保存到另一个 CSV 文件
  const csvDataWrite = results.map(result => ({
    file: result.file,
    fileSizeKB: result.fileSizeKB,
    writeTimeSec: result.avgWriteTimeSec
  }));
  
  const csvWriterWrite = csvWriterLib({
    path: path.join(outputDir, 'tonejs_benchmark_write.csv'),
    header: [
      { id: 'fileSizeKB', title: 'File Size (KB)' },
      { id: 'writeTimeSec', title: 'Write Time (s)' }
    ]
  });
  
  csvWriterWrite.writeRecords(csvDataWrite)
    .then(() => console.log(`Write results saved to ${path.join(outputDir, 'tonejs_write.csv')}`))
    .catch(err => console.error('Error writing write CSV:', err));
}

benchmarkToneJS();
