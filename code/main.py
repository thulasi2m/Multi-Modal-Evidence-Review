import os
import time
import csv
from google import genai
from google.genai import types

# Define schema natively without pydantic, using dicts directly
claim_schema = {
    "type": "OBJECT",
    "properties": {
        "evidence_standard_met": {"type": "STRING", "description": "'true' or 'false'"},
        "evidence_standard_met_reason": {"type": "STRING", "description": "short reason for the evidence decision"},
        "risk_flags": {"type": "STRING", "description": "semicolon-separated risk flags, or 'none'"},
        "issue_type": {"type": "STRING", "description": "visible issue type (e.g. dent, scratch, none, unknown)"},
        "object_part": {"type": "STRING", "description": "relevant object part (e.g. front_bumper, screen, unknown)"},
        "claim_status": {"type": "STRING", "description": "'supported', 'contradicted', or 'not_enough_information'"},
        "claim_status_justification": {"type": "STRING", "description": "concise image-grounded explanation"},
        "supporting_image_ids": {"type": "STRING", "description": "image IDs supporting decision, semicolon-separated, or 'none'"},
        "valid_image": {"type": "STRING", "description": "'true' or 'false'"},
        "severity": {"type": "STRING", "description": "'none', 'low', 'medium', 'high', or 'unknown'"}
    },
    "required": ["evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"]
}

def get_image_parts(image_paths_str: str, base_dir: str):
    """Load images to pass to Gemini."""
    paths = image_paths_str.split(";")
    image_parts = []
    for path in paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                img_data = f.read()
                ext = path.split('.')[-1].lower()
                mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
                # Fix: Use types.Part.from_bytes properly
                image_parts.append(types.Part.from_bytes(data=img_data, mime_type=mime))
    return image_parts

def read_csv_to_dicts(filepath):
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def process_claims(claims_file: str, history_file: str, rules_file: str, base_dir: str, out_file: str, api_key: str):
    if not api_key:
        print("ERROR: No API key provided! Please pass your API key using --api_key")
        return

    # Initialize client with explicitly passed api_key
    client = genai.Client(api_key=api_key)

    claims = read_csv_to_dicts(claims_file)
    history = read_csv_to_dicts(history_file)
    rules = read_csv_to_dicts(rules_file)

    output_rows = []
    
    system_instruction = """
    You are an expert multi-modal claims verification system. 
    Analyze the chat transcript and images to evaluate the damage claim.
    Review the user history for context but visual evidence is the primary source of truth.
    Determine issue_type, object_part, risk_flags, severity, and final claim_status.
    Risk flags: none, blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation, non_original_image, text_instruction_present, user_history_risk, manual_review_required.
    """

    for row in claims:
        user_id = row['user_id']
        image_paths = row['image_paths']
        user_claim = row['user_claim']
        claim_object = row['claim_object']

        # Get history context
        user_hist = [h for h in history if h['user_id'] == user_id]
        history_context = user_hist[0] if user_hist else "No history"
        
        # Determine rules context
        user_rules = [r for r in rules if r['claim_object'] in [claim_object, 'all']]
        rules_context = str(user_rules)

        prompt = f"""
        User ID: {user_id}
        Object Type: {claim_object}
        Chat Transcript: {user_claim}
        User History: {history_context}
        Evidence Requirements: {rules_context}
        Submitted Image Paths: {image_paths} (Image IDs are filenames without extensions)

        Evaluate the claim and return the structured JSON output matching the schema.
        """
        
        contents = get_image_parts(image_paths, base_dir)
        contents.append(prompt)

        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=contents,
                    config={
                        "system_instruction": system_instruction,
                        "response_mime_type": "application/json",
                        "response_schema": claim_schema,
                        "temperature": 0.0
                    },
                )
                
                # Parse JSON output
                out_data = response.text
                import json
                parsed = json.loads(out_data)
                
                output_row = {
                    'user_id': user_id,
                    'image_paths': image_paths,
                    'user_claim': user_claim,
                    'claim_object': claim_object,
                    'evidence_standard_met': parsed.get('evidence_standard_met'),
                    'evidence_standard_met_reason': parsed.get('evidence_standard_met_reason'),
                    'risk_flags': parsed.get('risk_flags'),
                    'issue_type': parsed.get('issue_type'),
                    'object_part': parsed.get('object_part'),
                    'claim_status': parsed.get('claim_status'),
                    'claim_status_justification': parsed.get('claim_status_justification'),
                    'supporting_image_ids': parsed.get('supporting_image_ids'),
                    'valid_image': parsed.get('valid_image'),
                    'severity': parsed.get('severity')
                }
                output_rows.append(output_row)
                break
            except Exception as e:
                print(f"Error processing {user_id}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    # Fallback output
                    fallback = {k: "error" for k in ["evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"]}
                    fallback['user_id'] = user_id
                    fallback['image_paths'] = image_paths
                    fallback['user_claim'] = user_claim
                    fallback['claim_object'] = claim_object
                    output_rows.append(fallback)
    
    if output_rows:
        keys = output_rows[0].keys()
        with open(out_file, 'w', newline='', encoding='utf-8') as f:
            dict_writer = csv.DictWriter(f, keys)
            dict_writer.writeheader()
            dict_writer.writerows(output_rows)
        print(f"Results saved to {out_file}")
    else:
        print("No claims processed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--claims', default='../dataset/claims.csv')
    parser.add_argument('--history', default='../dataset/user_history.csv')
    parser.add_argument('--rules', default='../dataset/evidence_requirements.csv')
    parser.add_argument('--base_dir', default='../dataset')
    parser.add_argument('--output', default='../output.csv')
    parser.add_argument('--api_key', default='', help="Your Gemini API Key")
    args = parser.parse_args()
    
    # Check env var as fallback
    api_key = args.api_key if args.api_key else os.environ.get("GEMINI_API_KEY", "")
    
    process_claims(args.claims, args.history, args.rules, args.base_dir, args.output, api_key)
