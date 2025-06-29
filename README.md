# RSOFD-dataset
RSOFD dataset construction code and data

## Dataset Download

The RSOFD dataset is available on multiple platforms:

1. **Quark Cloud Drive**  
   - Link: [https://pan.quark.cn/s/d194660241cb](https://pan.quark.cn/s/d194660241cb)  
   - Extraction code: `zLfw`

2. **Google Drive**  
   - Link: [https://drive.google.com/drive/folders/1il2vmDXzUYPkbqLXfVsfX9UdoHThc7cK](https://drive.google.com/drive/folders/1il2vmDXzUYPkbqLXfVsfX9UdoHThc7cK)


# Copy-Paste and Splicing Image Manipulation Tool

This tool provides three primary image manipulation capabilities: Copy-Paste, Spliced, and Inpaint(AIGC-removal) tampering. It utilizes deep learning technology for high-quality image inpainting and automatically processes related annotation files (JSON format).

## Core Functionality

- ✅ **Copy-Paste Tampering**: Copies and pastes polygonal regions (e.g., objects) to new positions **within the same image**
- ✅ **Splicing Tampering**: Transplants objects **from different images** into target images
- ✅ **Inpainting Tampering**: Removes objects and reconstructs background using pretrained LaMa model
- 📄 **Automatic Annotation Processing**: Automatically updates JSON annotation files after operations
- 🎭 **Mask Generation**: Generates corresponding binary masks (PNG format) for each operation
- 🔀 **Random Position Selection**: Automatically selects appropriate new positions for pasted objects

## Technical Stack

- **Programming Language**: Python 3.8+
- **Core Libraries**:
  - OpenCV (cv2) - Image processing
  - PyTorch - Deep learning framework
  - SimpleLamaInpainting - LaMa-based image inpainting
  - PIL/Pillow - Image processing
  - NumPy - Numerical computations
- **Supported File Formats**:
  - Images: JPG, PNG
  - Annotations: JSON format (LabelMe compatible)


## Quick Start

### Install Dependencies
```bash
pip install opencv-python torch torchvision pillow numpy json5 simple-lama-inpainting
```

### Prepare Dataset
Place source images in cropped_images/ directory
Place corresponding JSON annotation files in cropped_labels/ directory
Ensure image and annotation filenames match (e.g., image1.jpg and image1.json)

### Run the Program——Interactive Mode
```bash
python main.py
```
Then select operation type:
Copy-Paste (1)
Spliced (2)
Inpainting (3)
