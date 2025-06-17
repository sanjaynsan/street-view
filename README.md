# 🛰️ Street View to Safety

This project uses satellite/street-level imagery and computer vision to evaluate pedestrian safety indicators from Google Street View, such as:

- 🧍 Crosswalk presence
- 🚦 Traffic lights
- ⚠️ Potholes/damaged surfaces
- 💡 Lighting conditions

## Features
- Interactive ROI selection using maps
- Street View image scraping (Google API)
- Vision AI model (BLIP) to describe scenes
- Safety parameter extraction (Boolean)
- Interactive map visualization

## Folder Structure
```
Street-view-to-safety/
├── images/                 # Downloaded street view images
├── scripts/                # Python scripts and notebooks
├── data/                   # GeoJSON, CSV logs
├── results/                # Maps, reports
└── README.md
```

## Requirements
Install with:

```bash
pip install -r requirements.txt
```

## Run the pipeline
```bash
python main.py
```
