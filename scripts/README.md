# 脚本说明

## RMS 音频分析工具

用于分析音频文件的 RMS 能量时序，帮助确定远场声音过滤的最佳阈值。

## 功能特性

- ✅ 支持多种音频格式 (WAV, MP3, FLAC 等)
- ✅ 支持立体声、左声道、右声道选择
- ✅ 生成 RMS 时序图和分布直方图
- ✅ 详细的统计分析和阈值建议
- ✅ 可自定义分块大小（默认 240ms，与流式 ASR 一致）

## 安装依赖

```bash
uv sync --group cpu
```

## 使用方法

### 基础用法

```bash
# 分析立体声音频（默认）
python scripts/analyze_audio_rms.py audio.wav

# 仅分析左声道
python scripts/analyze_audio_rms.py audio.wav --channel left

# 仅分析右声道
python scripts/analyze_audio_rms.py audio.wav --channel right
```

### 高级用法

```bash
# 自定义阈值
python scripts/analyze_audio_rms.py audio.wav --threshold 0.015

# 自定义分块大小（例如 160ms）
python scripts/analyze_audio_rms.py audio.wav --chunk-size 160

# 保存图表到文件
python scripts/analyze_audio_rms.py audio.wav --output analysis.png

# 仅输出统计信息，不显示图表
python scripts/analyze_audio_rms.py audio.wav --no-plot
```

### 完整示例

```bash
# 分析右声道，使用 0.015 阈值，保存图表
python scripts/analyze_audio_rms.py recording.wav \
  --channel right \
  --threshold 0.015 \
  --output rms_analysis.png
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `audio_file` | 必需 | - | 音频文件路径 |
| `--channel` | 可选 | `stereo` | 声道选择: `stereo`, `left`, `right` |
| `--threshold` | 可选 | `0.01` | RMS 能量阈值 |
| `--chunk-size` | 可选 | `240` | 分块大小(毫秒) |
| `--output` | 可选 | - | 保存图表的路径 |
| `--no-plot` | 标志 | - | 不显示图表，仅输出统计 |

## 输出说明

### 统计信息

脚本会输出以下统计信息：

1. **基础统计**
   - 最小值、最大值、平均值、中位数、标准差

2. **百分位数**
   - P10, P25, P50, P75, P90, P95, P99

3. **阈值分析**
   - 超过/低于阈值的帧数和百分比

4. **建议的阈值**
   - 保守模式（高灵敏度）：P10
   - 宽松模式（推荐）：P25
   - 严格模式（低误触）：平均值的 50%

### 可视化图表

生成两个图表：

1. **RMS 时序图**
   - 显示整个音频的 RMS 能量变化
   - 绿色区域：近场音频（>= 阈值）
   - 红色区域：远场音频（< 阈值）
   - 红色虚线：当前阈值

2. **RMS 分布直方图**
   - 显示 RMS 值的分布情况
   - 红色虚线：当前阈值
   - 橙色虚线：平均值
   - 绿色虚线：中位数

## 使用场景

### 1. 确定初始阈值

录制一段包含近场说话和远场环境音的测试音频，然后：

```bash
python scripts/analyze_audio_rms.py test_audio.wav
```

查看**建议的阈值**，选择合适的模式。

### 2. 对比不同声道

如果使用立体声麦克风，可以对比左右声道：

```bash
python scripts/analyze_audio_rms.py audio.wav --channel left --output left.png
python scripts/analyze_audio_rms.py audio.wav --channel right --output right.png
```

### 3. 验证阈值效果

使用自定义阈值查看过滤效果：

```bash
# 测试宽松模式
python scripts/analyze_audio_rms.py audio.wav --threshold 0.01

# 测试严格模式
python scripts/analyze_audio_rms.py audio.wav --threshold 0.015

# 测试保守模式
python scripts/analyze_audio_rms.py audio.wav --threshold 0.005
```

### 4. 批量分析

如需批量分析多个文件，可以使用简单的循环：

```bash
#!/bin/bash
for file in recordings/*.wav; do
    echo "Analyzing: $file"
    python scripts/analyze_audio_rms.py "$file" \
      --threshold 0.01 \
      --no-plot \
      --output "analysis/$(basename "$file" .wav).png"
done
```

## 输出示例

```
==============================================================
音频 RMS 时序分析工具
==============================================================

📁 文件: test_audio.wav
🎚️  声道: stereo
📊 分块大小: 240ms
🎯 阈值: 0.010000

正在加载音频...
✓ 使用立体声（双声道平均）
✓ 采样率: 16000 Hz
✓ 时长: 30.50 秒
✓ 样本数: 488000

正在分析 RMS 时序 (分块大小: 240ms)...
✓ 分析了 127 个音频块

==============================================================
RMS 统计分析
==============================================================

📊 基础统计:
  - 最小值: 0.000234
  - 最大值: 0.085432
  - 平均值: 0.012567
  - 中位数: 0.009234
  - 标准差: 0.015678

📈 百分位数:
  - P10: 0.002345
  - P25: 0.005678
  - P50: 0.009234
  - P75: 0.015432
  - P90: 0.035678
  - P95: 0.045678
  - P99: 0.075432

🎯 阈值分析 (当前阈值: 0.010000):
  - 超过阈值的帧数: 65 (51.2%)
  - 低于阈值的帧数: 62 (48.8%)

💡 建议的阈值范围:
  - 保守模式 (高灵敏度): 0.002345 (P10)
  - 宽松模式 (推荐):     0.005678 (P25)
  - 严格模式 (低误触):   0.006284 (平均值的50%)
==============================================================
```

## 调优建议

1. **录制测试音频**
   - 包含正常说话（近场）
   - 包含远处说话或电视声音（远场）
   - 包含安静时刻（环境音）

2. **分析并选择阈值**
   - 查看时序图，观察近场和远场的 RMS 差异
   - 查看分布直方图，找到明显的分界点
   - 参考建议的阈值范围

3. **验证效果**
   - 在实际应用中测试选定的阈值
   - 如果误触发过多，提高阈值
   - 如果正常说话被过滤，降低阈值

4. **微调**
   - 宽松模式（0.01）适合大多数场景
   - 嘈杂环境使用严格模式（0.015）
   - 安静环境使用保守模式（0.005）

## 故障排查

### 问题：无法加载音频文件

```bash
# 安装必要的库
uv sync --group cpu

# 对于 MP3 文件，可能还需要
uv sync --group cpu
```

### 问题：中文显示乱码

脚本已配置中文字体，但如果仍然显示乱码：

1. macOS: 会自动使用 'Arial Unicode MS'
2. Windows: 会自动使用 'SimHei'
3. Linux: 需要安装中文字体

### 问题：图表不显示

```bash
# 后台运行或 SSH 连接时使用
python scripts/analyze_audio_rms.py audio.wav --no-plot --output analysis.png
```

## 与远场过滤功能的关系

此工具使用与 `app/utils/audio_filter.py` 相同的 RMS 计算方法，确保分析结果与实际运行时的行为一致。

确定阈值后，在配置文件中设置：

```bash
# docker-compose.yml
environment:
  - ASR_NEARFIELD_RMS_THRESHOLD=0.01  # 使用分析得出的阈值
```

或在 `.env` 文件中：

```bash
ASR_NEARFIELD_RMS_THRESHOLD=0.01
```

## 相关文档

- [远场过滤功能文档](../docs/nearfield_filter.md)
- [部署配置指南](../docs/deployment.md)
