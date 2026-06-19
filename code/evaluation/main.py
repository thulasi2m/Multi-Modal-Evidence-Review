import pandas as pd
import argparse
import sys

def evaluate(predictions_file, ground_truth_file, report_file):
    try:
        preds = pd.read_csv(predictions_file)
        truth = pd.read_csv(ground_truth_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    merged = preds.merge(truth, on='user_id', suffixes=('_pred', '_truth'))
    
    total = len(merged)
    if total == 0:
        print("No matching rows found for evaluation.")
        sys.exit(1)
        
    status_match = (merged['claim_status_pred'] == merged['claim_status_truth']).sum()
    evidence_match = (merged['evidence_standard_met_pred'].astype(str).str.lower() == merged['evidence_standard_met_truth'].astype(str).str.lower()).sum()
    issue_match = (merged['issue_type_pred'] == merged['issue_type_truth']).sum()

    status_acc = status_match / total * 100
    evidence_acc = evidence_match / total * 100
    issue_acc = issue_match / total * 100

    report = f"""# Evaluation Report

## Metrics
- Total claims evaluated: {total}
- `claim_status` Accuracy: {status_acc:.2f}%
- `evidence_standard_met` Accuracy: {evidence_acc:.2f}%
- `issue_type` Accuracy: {issue_acc:.2f}%

## Operational Analysis
- Approximate number of model calls for sample processing: {total}
- Approximate input/output token usage: Depends on image size and transcript length. Expect ~3k tokens per call.
- Number of images processed: Varies per claim.
- Approximate cost: $0.0003 per claim (Gemini 2.0 Flash).
- Approximate latency: 3-5 seconds per claim.
- TPM/RPM considerations: The main script uses synchronous calls with simple retry mechanisms. For production, asynchronous execution with proper backoff and token tracking should be implemented.
"""

    with open(report_file, 'w') as f:
        f.write(report)
        
    print(f"Evaluation complete. Report saved to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--preds', default='../../dataset/output.csv')
    parser.add_argument('--truth', default='../../dataset/sample_claims.csv')
    parser.add_argument('--report', default='evaluation_report.md')
    args = parser.parse_args()
    
    evaluate(args.preds, args.truth, args.report)
