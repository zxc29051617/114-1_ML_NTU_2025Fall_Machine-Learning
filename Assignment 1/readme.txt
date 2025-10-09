# Environment
```bash 
conda create -n ml1 python=3.11 -y
conda activate ml1
pip install -r
  numpy==2.3.3
  pandas==2.3.3
  scikit-learn==1.7.2
  xgboost==1.7.6
  lightgbm==4.6.0
  joblib==1.5.2
  tqdm==4.67.1
  scipy==1.16.2
```

# Directory Structure
```
R13945041/
│
├─ train_set/              # training data
├─ test_set/               # test data
├─ weights/                # output model directory
│   └─ xgb_rank_model.pkl  # saved trained model
│
├─ train.py                # training script
├─ Q5.py                   # inference script
└─ readme.txt              # this file
```

# Quick Start
(A) Training:  
```bash
conda activate ml1  
python .\train.py --train_dir .\train_set --out_dir .\weights
```

(B) Inference:
```bash  
python .\Q5.py `
  --model_path .\weights\xgb_rank_model.pkl `
  --test_dir .\test_set `
  --output_csv .\submission.csv `
  --chunk_size 500
```

# Reference Papers (optional section per spec)
Policies of Multiple Skill Levels for Better Strength Estimation in Games (arXiv:2505.00279, 2025)

# Note:
This submission follows all requirements stated in the official Q5 specification.
Both training and inference scripts are verified to run under Python 3.11 (conda env: ml1).
