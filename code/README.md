# Multi-Modal Evidence Review System

This directory contains the source code for processing damage claims using Google Gemini 2.0 Flash.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your Google Gemini API Key:
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

## Running the System

To process the full test set (`dataset/claims.csv`) and produce `output.csv`:

```bash
python main.py --claims ../dataset/claims.csv \
               --history ../dataset/user_history.csv \
               --rules ../dataset/evidence_requirements.csv \
               --base_dir ../dataset \
               --output ../output.csv
```

## Running Evaluation

To evaluate the system against `sample_claims.csv`:

1. First, generate predictions for the sample set:
   ```bash
   python main.py --claims ../dataset/sample_claims.csv \
                  --history ../dataset/user_history.csv \
                  --rules ../dataset/evidence_requirements.csv \
                  --base_dir ../dataset \
                  --output ../output.csv
   ```
2. Then run the evaluation script:
   ```bash
   cd evaluation
   python main.py --preds ../../output.csv --truth ../../dataset/sample_claims.csv --report evaluation_report.md
   ```
