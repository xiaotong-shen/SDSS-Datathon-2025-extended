# Setup Guide for SDSS Datathon 2025

## Prerequisites
- Python 3.8+ 
- Node.js 16+
- Git

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd SDSS-Datathon-2025-extended
```

### 2. Activate Virtual Environment
```bash
source .venv/bin/activate
```

### 3. Install Dependencies

#### Python Dependencies
All Python dependencies are already installed in the virtual environment:
- pandas, numpy, matplotlib, seaborn
- scikit-learn, torch, torchvision
- jupyter, ipykernel, notebook
- plotly, folium, geopandas, shapely
- tqdm, requests

#### Node.js Dependencies
```bash
npm install
```

### 4. Run Jupyter Notebooks

#### Option A: Start Jupyter Notebook
```bash
jupyter notebook
```
Then navigate to `python notebooks/` and open any `.ipynb` file.

#### Option B: Start Jupyter Lab
```bash
jupyter lab
```

### 5. Select the Correct Kernel
When opening notebooks, make sure to select the kernel: **"SDSS Datathon Python"**

## Available Notebooks

### Main Analysis Notebooks
- `python notebooks/neuralnet.ipynb` - Neural network for delay prediction
- `python notebooks/exploratory-data-analysis/csv-cleaning.ipynb` - Data cleaning
- `python notebooks/exploratory-data-analysis/cleaning_delaygraphs.ipynb` - Delay analysis
- `python notebooks/exploratory-data-analysis/geographical hotspots eda.py` - Geographic analysis

### Data Files
The notebooks use data from the `python notebooks/` directory:
- Monthly subway data files (2021-2024)
- Preprocessed data files
- Model files

## Troubleshooting

### Import Errors
If you get import errors, make sure:
1. Virtual environment is activated: `source .venv/bin/activate`
2. Correct kernel is selected in Jupyter
3. All dependencies are installed: `pip install -r requirements.txt`

### Kernel Issues
If the kernel doesn't appear:
```bash
python3 -m ipykernel install --user --name=sdss-datathon --display-name="SDSS Datathon Python"
```

### Memory Issues
For large datasets, consider:
- Using smaller data samples for testing
- Closing other applications
- Using data chunking in pandas

## Project Structure
```
SDSS-Datathon-2025-extended/
├── python notebooks/          # Jupyter notebooks and data
├── src/                      # Svelte app source code
├── requirements.txt          # Python dependencies
├── package.json             # Node.js dependencies
└── .venv/                   # Python virtual environment
```

## Next Steps
1. Start with `csv-cleaning.ipynb` to understand the data
2. Run `neuralnet.ipynb` for the main ML model
3. Explore other notebooks for additional analysis

