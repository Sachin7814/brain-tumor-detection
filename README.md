---
title: NeuroScan AI
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# NeuroScan AI – Brain Tumor Detection

MRI image upload karo aur CNN model batayega ki brain tumor ke signs hain ya nahi.

## Local run

```bash
pip install -r requirements.txt
python app.py
```

## Docker run (kisi bhi server par)

```bash
docker build -t neuroscan .
docker run -p 8000:7860 neuroscan
```

Phir browser me `http://localhost:8000` kholo.

## Notes

- Model file `BrainTumor10EpochsCategorical.h5` `app.py` ke saath same folder me honi chahiye.
- Ye project sirf educational purpose ke liye hai, medical diagnosis ke liye nahi.
