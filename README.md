# WatermarkRemover-local

新增以下功能：
  本地运行逻辑，原项目会自动从hf下载
  无GUI，手动指定ROI，适用于Linux命令行环境

## 本地运行逻辑
项目依赖的 Lama 模型默认会通过lama_cleaner自动从网络下载（通常来自 Hugging Face Hub），离线环境下需要提前准备模型文件并放到指定路径。
模型默认存储路径为：
```plaintext
~/.cache/huggingface/hub/*（在这里放big-lama.pt）
```

## 解决命令行环境下无法手动框选 ROI 的问题
项目默认通过cv2.selectROI进行图形化框选，命令行环境下无法操作，需修改代码支持通过参数指定水印区域坐标。
以下是基于 `watermark_remover.py` 现有代码结构的具体修改方案，解决命令行环境无法手动框选ROI的问题：


### 1. 修改命令行参数解析（`parse_args` 函数）
在现有参数基础上添加ROI坐标参数，用于指定水印区域的左上角坐标和宽高：
```python
def parse_args():
    parser = argparse.ArgumentParser(description="Video Watermark Remover")
    parser.add_argument("--input", "-i", type=str, default=".", help="Input directory containing videos")
    parser.add_argument("--output", "-o", type=str, default="output", help="Output directory")
    parser.add_argument("--preview", "-p", action="store_true", help="Preview effect before processing")
    # 新增ROI参数
    parser.add_argument("--roi-x", type=int, help="ROI左上角x坐标 (命令行模式必传)")
    parser.add_argument("--roi-y", type=int, help="ROI左上角y坐标 (命令行模式必传)")
    parser.add_argument("--roi-width", type=int, help="ROI宽度 (命令行模式必传)")
    parser.add_argument("--roi-height", type=int, help="ROI高度 (命令行模式必传)")
    return parser.parse_args()
```


### 2. 修改 `WatermarkDetector` 类以支持命令行ROI
修改 `__init__` 方法接收ROI参数，并调整 `select_roi` 方法优先使用命令行参数：
```python
class WatermarkDetector:
    # 新增roi参数，默认None
    def __init__(self, num_sample_frames=10, min_frame_count=7, dilation_kernel_size=5, roi=None):
        self.num_sample_frames = num_sample_frames
        self.min_frame_count = min_frame_count
        self.dilation_kernel_size = dilation_kernel_size
        self.roi = roi  # 从外部传入的ROI（命令行参数）
    
    # 修改select_roi方法，优先使用已传入的roi
    def select_roi(self, video_clip):
        # 如果已通过命令行指定ROI，直接返回
        if self.roi is not None:
            return self.roi
        
        # 原有图形化框选逻辑（仅在有GUI的环境下生效）
        frame = self.get_first_valid_frame(video_clip)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        display_height = 720
        scale_factor = display_height / frame.shape[0]
        display_width = int(frame.shape[1] * scale_factor)
        display_frame = cv2.resize(frame, (display_width, display_height))

        instructions = "Select ROI and press SPACE or ENTER"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(display_frame, instructions, (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

        r = cv2.selectROI(display_frame)
        cv2.destroyAllWindows()

        self.roi = (
            int(r[0] / scale_factor), 
            int(r[1] / scale_factor), 
            int(r[2] / scale_factor), 
            int(r[3] / scale_factor)
        )
        
        return self.roi
```


### 3. 在主程序中传递命令行ROI参数
在 `main` 函数中解析ROI参数，并传递给 `WatermarkDetector`：
```python
if __name__ == "__main__":
    args = parse_args()
    
    # 校验命令行ROI参数（必须同时提供4个参数）
    roi = None
    if any([args.roi_x, args.roi_y, args.roi_width, args.roi_height]):
        if not all([args.roi_x, args.roi_y, args.roi_width, args.roi_height]):
            print("Error: --roi-x, --roi-y, --roi-width, --roi-height must be provided together")
            sys.exit(1)
        roi = (args.roi_x, args.roi_y, args.roi_width, args.roi_height)
        print(f"Using command line ROI: {roi}")
    
    # 初始化水印检测器时传入ROI
    watermark_detector = WatermarkDetector(roi=roi)
    watermark_mask = None
    
    # 后续逻辑不变...
    lama_model, lama_config = initialize_lama(device=use_device)
    
    for video in videos:
        print(f"Processing {video}")
        video_clip = VideoFileClip(video)

        # 生成掩码时会自动使用命令行传入的ROI（无需手动框选）
        if watermark_mask is None:
            watermark_mask = watermark_detector.generate_mask(video_clip)
        
        # 禁用预览（命令行环境无法显示窗口）
        if preview_enabled:
            print("Warning: Preview is disabled in command line ROI mode")
            preview_enabled = False
        
        # 后续处理逻辑不变...
```


### 4. 禁用命令行模式下的图形化操作
在命令行环境下，`cv2.imshow` 和 `cv2.selectROI` 会导致程序崩溃，需确保命令行模式下不执行这些操作：
- 当通过 `--roi-*` 参数指定ROI时，自动禁用 `--preview`（如上述代码中强制 `preview_enabled = False`）
- 若用户同时传入 `--preview` 和 `--roi-*`，打印警告并忽略预览


### 使用方法
通过命令行直接指定ROI坐标运行（无需图形界面）：
```bash
python watermark_remover.py \
  --input /path/to/videos \
  --output /path/to/output \
  --roi-x 100 \  # 水印区域左上角x坐标
  --roi-y 200 \  # 水印区域左上角y坐标
  --roi-width 300 \  # 水印区域宽度
  --roi-height 150   # 水印区域高度
```

**注意**：由于加入了not all([x, y, width, height])逻辑，确保x, y非0，或者把这一行删除



# WatermarkRemover

一个基于LAMA模型的视频水印移除工具，能够批量清除视频中的固定水印。

## 效果展示

原始帧
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/origin.jpg'>

去除水印
<a href=''><img src='https://raw.githubusercontent.com/lxulxu/WatermarkRemover/master/image/no_watermark.jpg'>

## 系统要求

- Python 3.10

## 安装步骤

- 克隆仓库

```bash
git clone https://github.com/lxulxu/WatermarkRemover.git
cd WatermarkRemover
```

- 创建并激活虚拟环境（可选，推荐）

```bash
python -m venv venv
# Windows
venv\Scripts\activate
```

- 安装基础依赖

```bash
pip install -r requirements.txt
```

- 安装PyTorch（二选一）

  1. CPU版本

    ```bash
    pip install torch
    ```
  
  2. GPU版本（需要NVIDIA显卡）

      - 安装CUDA Toolkit

  	    访问 [NVIDIA CUDA下载页面](https://developer.nvidia.com/cuda-downloads)，选择对应的操作系统和版本。

      - 安装cuDNN
  
        访问 [NVIDIA cuDNN下载页面](https://developer.nvidia.com/cudnn-downloads)，选择与CUDA版本匹配的cuDNN。

      - 安装GPU版本的PyTorch

        访问 [PyTorch官方网站](https://pytorch.org/get-started/locally/)，选择与CUDA版本匹配的命令安装，例如：
        
  
         ```bash
          pip3 install torch==2.6.0+cu126 torchvision==0.21.0 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
         ```
  


​	程序会自动检测是否有可用的GPU输出相关信息并自动选择处理方式。

## 使用方法

### 基本用法

处理单个视频目录中的所有视频：

```bash
python watermark_remover.py --input /path/to/videos --output /path/to/output
```

### 带预览的处理

```bash
python watermark_remover.py --input /path/to/videos --output /path/to/output --preview
```

### 命令行参数

| 参数        | 简写 | 说明                   | 默认值         |
| ----------- | ---- | ---------------------- | -------------- |
| `--input`   | `-i` | 包含视频文件的输入目录 | `.` (当前目录) |
| `--output`  | `-o` | 处理后视频的输出目录   | `output`       |
| `--preview` | `-p` | 启用处理效果预览       | 禁用           |

## 工作流程

1. **水印区域选择**：程序会显示视频一帧，手动框选水印区域后按**SPACE**或**ENTER**键继续。
2. **效果预览**（可选）：显示处理效果预览，按**SPACE**或**ENTER**键确认或按**ESC**键取消退出程序。
3. **视频处理**：初次运行程序使用LAMA模型需较长时间下载模型。
4. **输出结果**： MP4格式视频

## 局限性

- 只能处理固定位置的水印（不支持移动水印）
- 同一批处理的视频尺寸必须一致
- 同一批处理的视频水印必须一致

## 常见问题

 **Q: GPU未正确启动，程序使用CPU运行**

 运行时显示类似信息 `No GPU detected, using CPU for processing`

A: 请按照 [LaMa Cleaner官方安装指南](https://lama-cleaner-docs.vercel.app/install/pip)检查你的环境配置

  - 检查Python版本是否为3.10
  - 检查已安装PyTorch版本是否有CPU版本，参考LaMa Cleaner官方网页说明

    > If Lama Cleaner is not using GPU device, it might CPU version of pytorch is installed, please follow pytorch's get-started(opens in a new tab) to install GPU version.
  
  - 确保安装的CUDA、cuDNN和PyTorch版本和显卡兼容

    > 感谢[@VitorX](https://github.com/VitorX)在[#issue11](https://github.com/lxulxu/WatermarkRemover/issues/11#issuecomment-3422248098)中提供的安装步骤

    - 查询[NVIDIA CUDA兼容性页面](https://en.wikipedia.org/wiki/CUDA#GPUs_supported)选择对应的CUDA版本
    - 查询[cuDNN官方页面](developer.nvidia.com/rdp/cudnn-archive)选择对应的cuDNN版本
    - 查询[PyTorch官方页面](https://pytorch.org/get-started/locally/)选择对应PyTorch版本

  程序正确检测到GPU会输出`GPU detected: NVIDIA XXX Using GPU for processing `提示信息

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=lxulxu/WatermarkRemover&type=Date)](https://star-history.com/#lxulxu/WatermarkRemover&Date)
