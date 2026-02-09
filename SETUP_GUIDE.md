# Floor Plan Generator - Setup Guide

## Prerequisites

- Python 3.9 - 3.11
- Virtual environment (recommended)
- Model checkpoint file: `GAT-Net_v3_UnScalled.pt`

## Installation Steps

### 1. Activate Your Virtual Environment

```bash
cd "/Users/sanketpatel/Desktop/Floor Plan Try/Floor_Plan_Generation_using_GNNs"
source ../env/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install PyTorch First

For macOS (CPU version):

```bash
pip install torch torchvision torchaudio
```

For macOS with Apple Silicon (M1/M2):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 4. Install PyTorch Geometric

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

### 5. Install All Other Dependencies

```bash
pip install -r requirements.txt
```

### 6. Verify Model Checkpoint Exists

Ensure the model file is at:

```
GAT-Net_model/checkpoints/GAT-Net_v3_UnScalled.pt
```

If the model file doesn't exist, you need to either:

- Train the model using the notebook: `GAT-Net_model/GAT-Net_Model.ipynb`
- Download it from the original repository
- Contact the original author

### 7. Create Outputs Directory

```bash
mkdir -p Outputs
```

## Running the Application

### Option 1: Run Streamlit App (Web Interface)

```bash
streamlit run app.py
```

This will open a web browser with the interactive interface.

### Option 2: Run Main Script (Command Line)

```bash
python main.py
```

This will run with the example data hardcoded in `get_info()`.

## Troubleshooting

### Issue: "AttributeError: module 'collections' has no attribute 'MutableMapping'"

**Solution:** Upgrade the requests library:

```bash
pip install --upgrade requests>=2.31.0
```

### Issue: "Model checkpoint not found"

**Solution:**

1. Check if `GAT-Net_model/checkpoints/GAT-Net_v3_UnScalled.pt` exists
2. If not, train the model or obtain it from the source repository
3. Update the path in `main.py` if your model is in a different location

### Issue: "No module named 'torch_geometric'"

**Solution:**

```bash
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
pip install torch-geometric
```

### Issue: Streamlit not opening

**Solution:**

```bash
pip install --upgrade streamlit
streamlit run app.py --server.port 8501
```

## Project Structure

```
Floor_Plan_Generation_using_GNNs/
├── app.py                          # Streamlit web interface
├── main.py                         # Main execution script
├── model.py                        # GAT-Net model definition
├── utils.py                        # Utility functions (graph conversion, visualization)
├── upload.py                       # Firebase upload utilities
├── requirements.txt                # Python dependencies
├── GAT-Net_model/
│   └── checkpoints/
│       └── GAT-Net_v3_UnScalled.pt # Trained model (REQUIRED)
└── Outputs/                        # Generated images
```

## Key Dependencies Explained

| Package           | Purpose                             |
| ----------------- | ----------------------------------- |
| `streamlit`       | Web interface framework             |
| `torch`           | PyTorch deep learning framework     |
| `torch_geometric` | Graph Neural Network library        |
| `networkx`        | Graph data structure and algorithms |
| `shapely`         | Geometric operations on polygons    |
| `geopandas`       | Geospatial data handling            |
| `matplotlib`      | Visualization and plotting          |
| `distinctipy`     | Color generation for room types     |

## Usage Examples

The app provides 4 pre-configured examples (EX 1-4) that you can select from the sidebar. Each example includes:

- Boundary polygon (WKT format)
- Front door location
- Room centroids
- Bathroom centroids
- Kitchen centroids

## Next Steps

1. Ensure all dependencies are installed
2. Verify the model checkpoint exists
3. Run `streamlit run app.py`
4. Select an example from the sidebar
5. View the generated floor plan

## Additional Resources

- Original Repository: https://github.com/mo7amed7assan1911/Floor_Plan_Generation_using_GNNs
- RPlan Dataset: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
