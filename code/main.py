import os
import time
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# Output Schema matching the required CSV columns
class ClaimOutput(BaseModel):
    evidence_standard_met: str = Field(description="'true' or 'false'")
    evidence_standard_met_reason: str = Field(description="short reason for the evidence decision")
    risk_flags: str = Field(description="semicolon-separated risk flags, or 'none'")
    issue_type: str = Field(description="visible issue type (e.g. dent, scratch, none, unknown)")
    object_part: str = Field(description="relevant object part (e.g. front_bumper, screen, unknown)")
    claim_status: str = Field(description="'supported', 'contradicted', or 'not_enough_information'")
    claim_status_justification: str = Field(description="concise image-grounded explanation")
    supporting_image_ids: str = Field(description="image IDs supporting decision, semicolon-separated, or 'none'")
    valid_image: str = Field(description="'true' or 'false'")
    severity: str = Field(description="'none', 'low', 'medium', 'high', or 'unknown'")

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
                image_parts.append(types.Part.from_bytes(data=img_data, mime_type=mime))
    return image_parts

def process_claims(claims_file: str, history_file: str, rules_file: str, base_dir: str, out_file: str):
    # Initialize client (Ensure GEMINI_API_KEY is set in environment)
    client = genai.Client()

    claims_df = pd.read_csv(claims_file)
    history_df = pd.read_csv(history_file)
    rules_df = pd.read_csv(rules_file)

    output_rows = []
    
    system_instruction = """
    You are an expert multi-modal claims verification system. 
    Analyze the chat transcript and images to evaluate the damage claim.
    Review the user history for context but visual evidence is the primary source of truth.
    Determine issue_type, object_part, risk_flags, severity, and final claim_status.
    Risk flags: none, blurry_image, cropped_or_obstructed, low_light_or_glare, wrong_angle, wrong_object, wrong_object_part, damage_not_visible, claim_mismatch, possible_manipulation, non_original_image, text_instruction_present, user_history_risk, manual_review_required.
    """

    for idx, row in claims_df.iterrows():
        user_id = row['user_id']
        image_paths = row['image_paths']
        user_claim = row['user_claim']
        claim_object = row['claim_object']

        # Get history context
        history = history_df[history_df['user_id'] == user_id]
        history_context = history.to_dict('records')[0] if not history.empty else "No history"
        
        # Determine rules context
        rules = rules_df[(rules_df['claim_object'] == claim_object) | (rules_df['claim_object'] == 'all')]
        rules_context = rules.to_string(index=False)

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
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        response_mime_type="application/json",
                        response_schema=ClaimOutput,
                        temperature=0.0
                    ),
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
                    fallback = {k: "error" for k in ClaimOutput.model_fields.keys()}
                    fallback['user_id'] = user_id
                    fallback['image_paths'] = image_paths
                    fallback['user_claim'] = user_claim
                    fallback['claim_object'] = claim_object
                    output_rows.append(fallback)
    
    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_file, index=False)
    print(f"Results saved to {out_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--claims', default='../dataset/claims.csv')
    parser.add_argument('--history', default='../dataset/user_history.csv')
    parser.add_argument('--rules', default='../dataset/evidence_requirements.csv')
    parser.add_argument('--base_dir', default='../dataset')
    parser.add_argument('--output', default='../output.csv')
    args = parser.parse_args()
    
    process_claims(args.claims, args.history, args.rules, args.base_dir, args.output)
