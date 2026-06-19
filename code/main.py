import os
import time
import csv
import json
import base64
import urllib.request
import urllib.error

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
    paths = image_paths_str.split(";")
    image_parts = []
    for path in paths:
        full_path = os.path.join(base_dir, path)
        if os.path.exists(full_path):
            with open(full_path, "rb") as f:
                img_data = f.read()
                ext = path.split('.')[-1].lower()
                mime = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
                image_parts.append({
                    "inlineData": {
                        "mimeType": mime,
                        "data": base64.b64encode(img_data).decode('utf-8')
                    }
                })
    return image_parts

def read_csv_to_dicts(filepath):
    with open(filepath, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

def generate_content_rest(api_key, model, contents_parts, system_instruction):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_instruction}]
        },
        "contents": [{
            "parts": contents_parts
        }],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
            "responseSchema": claim_schema
        }
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    
    try:
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            text = result['candidates'][0]['content']['parts'][0]['text']
            return text
    except urllib.error.HTTPError as e:
        error_msg = e.read().decode('utf-8')
        raise Exception(f"HTTP {e.code}: {error_msg}")

def process_claims(claims_file: str, history_file: str, rules_file: str, base_dir: str, out_file: str, api_key: str):
    if not api_key:
        print("ERROR: No API key provided!")
        return

    working_model = 'gemini-2.5-flash'
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

    quota_exhausted = False

    for row in claims:
        user_id = row['user_id']
        image_paths = row['image_paths']
        user_claim = row['user_claim']
        claim_object = row['claim_object']

        if quota_exhausted:
            # Skip API and just fill with generic response so the file finishes
            print(f"Skipping API for {user_id} due to daily limit...")
            fallback = {k: "error_quota_exceeded" for k in ["evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"]}
            fallback['user_id'] = user_id
            fallback['image_paths'] = image_paths
            fallback['user_claim'] = user_claim
            fallback['claim_object'] = claim_object
            output_rows.append(fallback)
            continue

        user_hist = [h for h in history if h['user_id'] == user_id]
        history_context = user_hist[0] if user_hist else "No history"
        user_rules = [r for r in rules if r['claim_object'] in [claim_object, 'all']]
        rules_context = str(user_rules)

        prompt_text = f"""
        User ID: {user_id}
        Object Type: {claim_object}
        Chat Transcript: {user_claim}
        User History: {history_context}
        Evidence Requirements: {rules_context}
        Submitted Image Paths: {image_paths}
        Evaluate the claim and return structured JSON output.
        """
        
        contents_parts = get_image_parts(image_paths, base_dir)
        contents_parts.append({"text": prompt_text})

        try:
            print(f"Processing {user_id} with {working_model}...")
            out_data = generate_content_rest(api_key, working_model, contents_parts, system_instruction)
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
            print(f"Success for {user_id}! Waiting 15 seconds to avoid minute limits...")
            time.sleep(15) 
        except Exception as e:
            error_str = str(e)
            print(f"Error for {user_id}: {error_str}")
            if "GenerateRequestsPerDay" in error_str or "quotaValue\": \"20\"" in error_str:
                print("DAILY QUOTA COMPLETELY EXHAUSTED! Skipping remaining rows.")
                quota_exhausted = True
                fallback = {k: "error_quota_exceeded" for k in ["evidence_standard_met", "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part", "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"]}
                fallback['user_id'] = user_id
                fallback['image_paths'] = image_paths
                fallback['user_claim'] = user_claim
                fallback['claim_object'] = claim_object
                output_rows.append(fallback)
            else:
                print("Temporary error, falling back...")
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
    api_key = args.api_key if args.api_key else os.environ.get("GEMINI_API_KEY", "")
    process_claims(args.claims, args.history, args.rules, args.base_dir, args.output, api_key)
