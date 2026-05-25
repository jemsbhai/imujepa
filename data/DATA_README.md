# Data Provenance

## Pretraining Corpus

### NHANES 2011-2014 (National Health and Nutrition Examination Survey)
- **Source:** CDC / NCHS
- **URL:** https://wwwn.cdc.gov/Nchs/Nhanes/search/datapage.aspx?Component=Examination
- **Format:** Per-participant tar.bz2 archives with hourly CSV files (80Hz tri-axial wrist accelerometer)
- **License:** Public domain (US government data)
- **Participants:** ~11,000
- **Duration:** ~28,000 hours total
- **Channels:** 3 (acc_x, acc_y, acc_z)
- **Device:** ActiGraph GT3X+ (wrist-worn)

## Evaluation Benchmarks

### UCI-HAR (UCI Human Activity Recognition)
- **Source:** UCI ML Repository
- **URL:** https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
- **Subjects:** 30 | **Activities:** 6 | **Channels:** 6 (acc+gyro) | **Location:** waist | **Rate:** 50Hz

### PAMAP2 (Physical Activity Monitoring)
- **Source:** UCI ML Repository
- **URL:** https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
- **Subjects:** 9 | **Activities:** 18 | **Channels:** 36+ | **Locations:** wrist, chest, ankle | **Rate:** 100Hz

### DSADS (Daily and Sports Activities)
- **Source:** UCI ML Repository
- **URL:** https://archive.ics.uci.edu/dataset/256/daily+and+sports+activities
- **Subjects:** 8 | **Activities:** 19 | **Channels:** 45 | **Locations:** torso, arms, legs | **Rate:** 25Hz

### MHealth (Mobile Health)
- **Source:** UCI ML Repository
- **URL:** https://archive.ics.uci.edu/dataset/319/mhealth+dataset
- **Subjects:** 10 | **Activities:** 12 | **Channels:** 23 | **Locations:** chest, wrist, ankle | **Rate:** 50Hz

### USC-HAD (USC Human Activity Dataset)
- **Source:** USC
- **URL:** https://sipi.usc.edu/had/
- **Subjects:** 14 | **Activities:** 12 | **Channels:** 6 (acc+gyro) | **Location:** hip | **Rate:** 100Hz

### Realworld (RealWorld HAR)
- **Source:** TU Darmstadt
- **URL:** https://www.uni-mannheim.de/dws/research/projects/activity-recognition/
- **Subjects:** 15 | **Activities:** 8 | **Channels:** 63 | **Locations:** 7 positions | **Rate:** 50Hz

## Preprocessing Standard

All datasets are resampled to 100Hz, segmented into 5-second non-overlapping windows (500 samples), and z-score normalized per subject. See src/data/ for preprocessing scripts.
