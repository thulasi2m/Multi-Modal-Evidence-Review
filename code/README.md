# Multi-Modal Evidence Review

## Approach Overview
This system processes multi-modal damage claims (images, claim transcripts, user history, and evidence requirements).
To bypass compilation issues on certain Python versions, the system was built natively using `google-genai` and standard Python libraries (`csv`, `json`, etc.) without relying on heavy frameworks like `pandas` or `pydantic`. 

The system uses Gemini 2.0 Flash to analyze the images alongside the user chat transcripts and historical risk data to make a final determination (`supported`, `contradicted`, `not_enough_information`), outputting the results into a structured CSV format matching the required schema.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the pipeline:
   ```bash
   python main.py --api_key YOUR_GEMINI_API_KEY --claims ../dataset/claims.csv --history ../dataset/user_history.csv --rules ../dataset/evidence_requirements.csv --base_dir ../dataset --output output.csv
   ```

## Evaluation Pipeline
The system includes an evaluation script in `evaluation/main.py` which computes accuracy metrics for `claim_status`, `evidence_standard_met`, and `issue_type` against the ground truth sample claims.
