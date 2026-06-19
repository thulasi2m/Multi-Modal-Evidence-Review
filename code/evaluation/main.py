import csv
import argparse
import sys

def read_csv_to_dicts(filepath):
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def evaluate(predictions_file, ground_truth_file, report_file):
    try:
        preds = read_csv_to_dicts(predictions_file)
        truth = read_csv_to_dicts(ground_truth_file)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    preds_dict = {p['user_id']: p for p in preds}
    truth_dict = {t['user_id']: t for t in truth}
    
    merged = []
    for uid, t in truth_dict.items():
        if uid in preds_dict:
            merged.append({'truth': t, 'pred': preds_dict[uid]})
            
    total = len(merged)
    if total == 0:
        print("No matching rows found for evaluation.")
        sys.exit(1)
        
    status_match = sum(1 for m in merged if m['pred'].get('claim_status') == m['truth'].get('claim_status'))
    evidence_match = sum(1 for m in merged if str(m['pred'].get('evidence_standard_met')).lower() == str(m['truth'].get('evidence_standard_met')).lower())
    issue_match = sum(1 for m in merged if m['pred'].get('issue_type') == m['truth'].get('issue_type'))

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
