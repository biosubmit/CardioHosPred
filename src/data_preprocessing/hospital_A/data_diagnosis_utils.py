"""
For the diagnosis information column, use ClinicalBERT for embedding representation. Each row in this column 
contains diagnostic information with disease names separated by spaces. The process involves getting the 
disease names, then using ClinicalBERT for embedding representation.

Integration with Google Gemini API Instructions:
1. Need to apply for a Google Gemini API key
2. Install google-generativeai package: pip install google-generativeai
3. Implement batch translation functionality in create_translation_mapping function
4. Gemini API can receive more text at once, suitable for batch translation scenarios

Specific steps:
1. Get disease names separated by spaces
2. Translate disease names using Google Gemini API
3. Embed each disease name using ClinicalBERT
4. Aggregate multiple representations to get the final diagnosis information embedding
5. Save the embedding results

Each step above is an independent function
"""
import pandas as pd
from transformers import AutoTokenizer, AutoModel
# Import Google Gemini API related packages
import os
import time
import torch
import sys  # For progress display
from typing import List, Dict, Union, Optional
import re

# Try to import Google Gemini API
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get API key from environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
        GEMINI_AVAILABLE = True
    else:
        print("Warning: GOOGLE_API_KEY environment variable not found, please ensure the correct API key is set")
        GEMINI_AVAILABLE = False
except ImportError:
    print("Warning: google-generativeai or dotenv package not installed, please install using pip install google-generativeai python-dotenv")
    GEMINI_AVAILABLE = False
except Exception as e:
    print(f"Warning: Error configuring Gemini API: {e}")
    GEMINI_AVAILABLE = False

def contains_chinese(text: str) -> bool:
    """
    Check if text contains Chinese characters
    
    Args:
        text: Text to check
        
    Returns:
        bool: Returns True if text contains Chinese characters, otherwise False
    """
    if not text:
        return False
    
    # Unicode range for Chinese characters
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def gemini_translate(text_batch: List[str], max_retries: int = 3) -> List[str]:
    """
    Translate a batch of medical phrases using Google Gemini API, using special markers to ensure accurate separation
    
    Args:
        text_batch: List of Chinese phrases to translate
        max_retries: Maximum number of retries
    
    Returns:
        List of translated English phrases
    """
    if not text_batch:
        return []
    
    # Check if Gemini API is available
    if not GEMINI_AVAILABLE:
        print("Error: Gemini API is not available, cannot perform translation")
        return text_batch
    
    # Special markers for wrapping each phrase, using more unique markers to avoid conflicts
    START_TAG = "<<PHRASE_"
    END_TAG = ">>"
    
    # Wrap each phrase with special markers, and add index to ensure order can be restored
    tagged_texts = [f"{START_TAG}{i}{END_TAG}{phrase}{START_TAG}/{i}{END_TAG}" for i, phrase in enumerate(text_batch)]
    combined_text = " ".join(tagged_texts)
    batch_size = len(text_batch)
    
    # Construct prompt
    prompt = f"""The following are {batch_size} Chinese medical phrases, each phrase is contained within special XML-style tags.
Each phrase is formatted as: <<PHRASE_number>>phrase<<PHRASE_/number>>, where number is the index of the phrase.

Please translate each Chinese phrase into English, keeping the original tag format unchanged.
Return the results in the original order, ensuring each translated phrase is contained within the same tags.
Do not translate the tags themselves, they are only for identifying different phrases.
Make sure to return translations for all {batch_size} phrases.
Preserve English, numbers, and special symbols unchanged, only translate the Chinese part.

For example input:
<<PHRASE_0>>感冒<<PHRASE_/0>> <<PHRASE_1>>高血压(原发性)<<PHRASE_/1>> <<PHRASE_2>>头痛，严重<<PHRASE_/2>>

Should return:
<<PHRASE_0>>common cold<<PHRASE_/0>> <<PHRASE_1>>hypertension (primary)<<PHRASE_/1>> <<PHRASE_2>>headache, severe<<PHRASE_/2>>

Here are the phrases to translate:
{combined_text}
"""
    
    # Create generative model instance
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print(f"Failed to create Gemini model instance: {e}")
        return text_batch
    
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            response_text = response.text
            
            # Use regex with index to extract content between tags
            translated_dict = {}
            for i in range(batch_size):
                pattern = re.compile(rf"{START_TAG}{i}{END_TAG}(.*?){START_TAG}/{i}{END_TAG}")
                matches = pattern.findall(response_text)
                if matches:
                    translated_dict[i] = matches[0]
            
            # Rebuild translation list in original order
            translated_matches = []
            missing_indices = []
            for i in range(batch_size):
                if i in translated_dict:
                    translated_matches.append(translated_dict[i])
                else:
                    missing_indices.append(i)
                    # Temporarily fill with original text, will be handled later
                    translated_matches.append(text_batch[i])
            
            # Ensure extracted translations match the number of original phrases
            if len(translated_matches) != batch_size or missing_indices:
                print(f"Warning: Translation results incomplete, missing indices: {missing_indices}")
                
                if attempt == max_retries - 1:
                    # Reached maximum retries, try to translate missing items individually
                    if missing_indices:
                        print(f"Trying to translate {len(missing_indices)} missing items individually...")
                        for idx in missing_indices:
                            # Translate each missing item individually
                            single_result = gemini_translate_single(text_batch[idx])
                            if single_result:
                                translated_matches[idx] = single_result
                    
                    print(f"Translation completed, success rate: {(batch_size - len(missing_indices))/batch_size*100:.1f}%")
                    return translated_matches
            else:
                print(f"Translation completed, success rate: 100%")
                return translated_matches
                
        except Exception as e:
            print(f"Error translating batch: {e}, attempting retry {attempt+1}/{max_retries}")
            time.sleep(2)  # Wait after error
    
    # All attempts failed, return original list
    return text_batch

def gemini_translate_single(phrase: str) -> str:
    """
    Translate a single medical phrase using Google Gemini API
    
    Args:
        phrase: Chinese phrase to translate
        
    Returns:
        Translated English phrase
    """
    if not phrase or not GEMINI_AVAILABLE:
        return phrase
    
    # If no Chinese characters, return original phrase
    if not contains_chinese(phrase):
        return phrase
    
    prompt = f"""Please translate the following Chinese medical phrase into English, return only the translation result without any additional explanation.
Please preserve English, numbers, and special symbols unchanged, only translate the Chinese part.
    
Chinese phrase: {phrase}
English translation:"""
    
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        # Simple response cleaning
        result = response.text.strip()
        return result
    except Exception as e:
        print(f"Single phrase translation failed: {phrase}, error: {e}")
        return phrase

def create_translation_mapping(unique_phrases: List[str], batch_size: int = 100, max_retries: int = 3) -> Dict[str, str]:
    """
    Batch translate unique diagnostic phrases and create translation mapping
    
    Use Google Gemini API for translation, translating Chinese diagnostic phrases in batches to English
    
    Args:
        unique_phrases: List of unique Chinese diagnostic phrases
        batch_size: Batch size for each translation
        max_retries: Maximum number of retries on translation failure
        
    Returns:
        dict: Mapping from Chinese phrases to English translations
    """
    if not unique_phrases:
        return {}
    
    # Filter empty strings and phrases without Chinese characters (no need to translate)
    filtered_phrases = []
    non_chinese_phrases = {}
    
    for phrase in unique_phrases:
        if phrase and phrase.strip():
            if contains_chinese(phrase):
                filtered_phrases.append(phrase)
            else:
                # For phrases without Chinese, map directly to themselves
                non_chinese_phrases[phrase] = phrase
    
    if not filtered_phrases:
        return non_chinese_phrases
    
    # Initialize translation mapping (including already processed non-Chinese phrases)
    translation_map = dict(non_chinese_phrases)
    
    # Track translation status
    translated_phrases = set()  # Successfully translated phrases
    original_phrases = set()    # Phrases kept in original form
    
    # Batch translation processing
    total = len(filtered_phrases)
    print(f"Starting translation of {total} Chinese diagnostic phrases...")
    print(f"Skipped {len(non_chinese_phrases)} phrases without Chinese characters")
    
    # Check if Gemini API is available
    if not GEMINI_AVAILABLE:
        print("Warning: Gemini API is not available, will use original phrases")
        # Create mapping (original to original)
        for phrase in filtered_phrases:
            translation_map[phrase] = phrase
            original_phrases.add(phrase)
        return translation_map
    
    # First round: Batch translation
    print("--- First round: Batch translation processing ---")
    # Split list into batches of batch_size
    for i in range(0, total, batch_size):
        batch = filtered_phrases[i:i+batch_size]
        batch_size_actual = len(batch)
        
        # Show progress
        print(f"Translation batch: {i//batch_size + 1}/{(total-1)//batch_size + 1}, processing {i+1}-{min(i+batch_size_actual, total)}/{total} phrases")
        
        # Use Gemini API for batch translation
        translated_batch = gemini_translate(batch, max_retries)
        
        # Create mapping and track status
        for j, phrase in enumerate(batch):
            if j < len(translated_batch):
                translation = translated_batch[j]
                
                # Check if translation result still contains Chinese characters (sign of translation failure)
                if contains_chinese(translation):
                    translation_map[phrase] = phrase
                    original_phrases.add(phrase)
                    print(f"Note: Phrase '{phrase}' translation still contains Chinese, keeping original")
                else:
                    translation_map[phrase] = translation
                    translated_phrases.add(phrase)
            else:
                # If returned translation count is insufficient, use original phrase
                translation_map[phrase] = phrase
                original_phrases.add(phrase)
                print(f"Warning: Could not get translation for '{phrase}', using original")
        
        # Pause after batch translation to avoid request limits
        if i + batch_size < total:  # If not the last batch, add pause
            time.sleep(1)
    
    # First round translation statistics
    translated_count = len(translated_phrases)
    original_count = len(original_phrases)
    success_rate = (translated_count / total) * 100
    
    print(f"\nFirst round translation complete: Total {total} phrases")
    print(f"- Successfully translated: {translated_count} ({success_rate:.1f}%)")
    print(f"- Kept original: {original_count} ({(original_count / total) * 100:.1f}%)")
    
    # Second round: Translate unsuccessful phrases individually
    if original_phrases:
        print("\n--- Second round: Translate unsuccessful phrases individually ---")
        print(f"Starting individual translation of {len(original_phrases)} phrases...")
        
        # Extract unsuccessfully translated phrases
        phrases_to_retry = list(original_phrases)
        retry_total = len(phrases_to_retry)
        retry_success = 0
        
        for i, phrase in enumerate(phrases_to_retry):
            # Show progress (every 10 items or last one)
            if (i + 1) % 10 == 0 or i + 1 == retry_total:
                print(f"Individual translation progress: {i+1}/{retry_total} ({(i+1)/retry_total*100:.1f}%)")
            
            # Individual translation
            translation = gemini_translate_single(phrase)
            
            # Check if translation result still contains Chinese characters
            if translation and not contains_chinese(translation):
                translation_map[phrase] = translation
                translated_phrases.add(phrase)
                original_phrases.remove(phrase)
                retry_success += 1
            
            # Brief pause to avoid API limits
            time.sleep(0.5)
        
        # Second round translation statistics
        print(f"\nSecond round translation complete: Attempted {retry_total} phrases")
        print(f"- Successfully translated: {retry_success} ({(retry_success / retry_total) * 100:.1f}%)")
        print(f"- Still kept original: {len(original_phrases)} ({(len(original_phrases) / total) * 100:.1f}%)")
    
    # Overall translation statistics
    final_translated_count = len(translated_phrases)
    final_success_rate = (final_translated_count / total) * 100
    print(f"\nOverall translation results: Total {total} phrases")
    print(f"- Successfully translated: {final_translated_count} ({final_success_rate:.1f}%)")
    print(f"- Kept original: {len(original_phrases)} ({(len(original_phrases) / total) * 100:.1f}%)")
    print(f"- Skipped non-Chinese: {len(non_chinese_phrases)}")
    
    # Save translation records
    try:
        # Create log directory
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Save successful translations record
        with open(f"{log_dir}/successful_translations.txt", "w", encoding="utf-8") as f:
            for phrase in translated_phrases:
                f.write(f"{phrase} => {translation_map[phrase]}\n")
        
        # Save untranslated record
        with open(f"{log_dir}/untranslated_phrases.txt", "w", encoding="utf-8") as f:
            for phrase in original_phrases:
                f.write(f"{phrase}\n")
        
        # Save skipped non-Chinese phrases record
        with open(f"{log_dir}/non_chinese_phrases.txt", "w", encoding="utf-8") as f:
            for phrase in non_chinese_phrases:
                f.write(f"{phrase}\n")
                
        print(f"Translation records saved to {log_dir} directory")
    except Exception as e:
        print(f"Error saving translation records: {e}")
    
    return translation_map

def translate_with_mapping(diagnosis_list, translation_map):
    """
    Translate diagnosis list using translation mapping
    
    Args:
        diagnosis_list: Diagnosis list, can be single-level list or nested list
        translation_map: Translation mapping from Chinese to English
        
    Returns:
        Translated diagnosis list, maintaining the same structure as input
    """
    if diagnosis_list is None:
        return None
    
    # Handle nested list
    if isinstance(diagnosis_list, list) and diagnosis_list and isinstance(diagnosis_list[0], list):
        return [[translation_map.get(name, name) for name in sublist if name and name.strip() != ''] 
                for sublist in diagnosis_list]
    
    # Handle single-level list
    elif isinstance(diagnosis_list, list):
        return [translation_map.get(name, name) for name in diagnosis_list if name and name.strip() != '']
    
    # Handle single string
    else:
        return translation_map.get(diagnosis_list, diagnosis_list) if diagnosis_list and diagnosis_list.strip() != '' else None

def translate_diagnosis_names(diagnosis_names: List[str]) -> Union[List[str], List[List[str]]]:
    """
    Translate diagnosis name list from Chinese to English
    
    Args:
        diagnosis_names: Chinese diagnosis name list
        
    Returns:
        English diagnosis name list, may be single-level list or nested list
    """
    if not diagnosis_names:
        return []
    
    # Create mapping for unique phrases
    unique_phrases = set(diagnosis_names)
    translation_map = create_translation_mapping(list(unique_phrases))
    
    # Use mapping for translation
    result = translate_with_mapping(diagnosis_names, translation_map)
    # Ensure return type is list
    if result is None:
        return []
    return result

def get_diagnosis_names(diagnosis_info):
    """
    Get different disease names separated by spaces, input is a column in dataframe, each row represents diagnosis information for different patients
    Output is a list of diagnosis information for each patient, multiple rows
    
    Process various input types, including strings, floats, None, etc.
    """
    # Handle empty values, NaN, or non-string inputs
    if diagnosis_info is None or pd.isna(diagnosis_info) or not isinstance(diagnosis_info, str):
        return []
    
    # Handle empty strings
    if not diagnosis_info.strip():
        return []
        
    return diagnosis_info.split(' ')

def embed_diagnosis_names(diagnosis_names: Union[List[str], List[List[str]], str, None]):
    """
    Embed each disease name using ClinicalBERT
    Supports processing nested list format inputs
    
    Args:
        diagnosis_names: Diagnosis name list, can be single-level list, nested list, single string, or None
        
    Returns:
        Embedding representation results
    """
    # Handle empty values or None
    if diagnosis_names is None or len(diagnosis_names) == 0:
        return None
        
    tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
    model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
    
    # Check if it's a nested list
    if isinstance(diagnosis_names, list) and diagnosis_names and isinstance(diagnosis_names[0], list):
        # Process each sublist in the nested list separately
        all_embeddings = []
        for sublist in diagnosis_names:
            if not sublist:  # Handle empty list
                continue
            # Ensure sublist is a list, not a single string
            if isinstance(sublist, list):
                inputs = tokenizer.batch_encode_plus(sublist, return_tensors='pt', padding=True, truncation=True, max_length=512)
                outputs = model(**inputs)
                # Get embedding representation for each sublist
                embedding = outputs.last_hidden_state.mean(dim=1)
                all_embeddings.append(embedding)
        return all_embeddings
    else:
        # Ensure it's a list, not a single string
        if isinstance(diagnosis_names, str):
            diagnosis_names = [diagnosis_names]
        
        inputs = tokenizer.batch_encode_plus(diagnosis_names, return_tensors='pt', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)


def aggregate_diagnosis_embeddings(diagnosis_embeddings):
    """
    Aggregate multiple representations, for [2, 768] representation, aggregate to get [1, 768] representation
    """
    if diagnosis_embeddings is None:
        return None
        
    if isinstance(diagnosis_embeddings, list):
        if not diagnosis_embeddings:
            return None
        diagnosis_embeddings = diagnosis_embeddings[0]
        
    return diagnosis_embeddings.mean(dim=0)

def embed_diagnosis_info(diagnosis_info):
    """
    Embed diagnosis information
    """
    if not diagnosis_info:
        return None
        
    diagnosis_names = get_diagnosis_names(diagnosis_info)

    # Translate disease names
    diagnosis_names_en = translate_diagnosis_names(diagnosis_names)
    
    # Embed representation
    diagnosis_embeddings = embed_diagnosis_names(diagnosis_names_en)
    
    # Aggregate representation
    return aggregate_diagnosis_embeddings(diagnosis_embeddings)


def incorporate_diagnosis_embedding(
    patient_info_df: pd.DataFrame,
) -> pd.DataFrame:
    
    total_records = len(patient_info_df)
    print(f"Processing diagnosis information embedding for {total_records} case records")
    
    # Ensure NaN values in the main diagnosis column are replaced with empty strings
    patient_info_df['主要诊断'] = patient_info_df['主要诊断'].fillna('')  # 主要诊断: Main diagnosis
    
    # Extract all diagnosis information
    all_diagnoses = patient_info_df['主要诊断'].tolist()  # 主要诊断: Main diagnosis
    
    # Extract all unique diagnosis phrases
    unique_phrases = set()
    for diagnosis in all_diagnoses:
        # Ensure diagnosis is a string
        if diagnosis and isinstance(diagnosis, str):
            phrases = get_diagnosis_names(diagnosis)
            unique_phrases.update(phrases)
    
    print(f"Found {len(unique_phrases)} unique diagnosis phrases, creating translation mapping...")
    
    # Create translation mapping
    translation_map = create_translation_mapping(list(unique_phrases))
    
    print(f"Translation mapping created, starting diagnosis information embedding processing...")
    
    # Initialize counter and processed items
    processed = 0
    
    # Define processing function and display progress
    def process_with_progress(diagnosis_info):
        nonlocal processed
        
        # Handle empty values, NaN, or non-string inputs
        if diagnosis_info is None or pd.isna(diagnosis_info) or not isinstance(diagnosis_info, str) or not diagnosis_info.strip():
            processed += 1
            update_progress()
            return None
        
        try:
            # Get diagnosis names
            diagnosis_names = get_diagnosis_names(diagnosis_info)
            
            # Use mapping for translation
            diagnosis_names_en = translate_with_mapping(diagnosis_names, translation_map)
            
            # Embed representation - need to accept more flexible types here
            diagnosis_embeddings = embed_diagnosis_names(diagnosis_names_en)
            
            # Aggregate representation
            result = aggregate_diagnosis_embeddings(diagnosis_embeddings)
        except Exception as e:
            print(f"\nError processing diagnosis information: {diagnosis_info}, error: {e}")
            result = None
        
        processed += 1
        update_progress()
        
        return result
    
    def update_progress():
        # Update progress bar (display once per 1% or 10 records)
        if processed % max(1, min(10, total_records // 100)) == 0 or processed == total_records:
            percent = processed / total_records * 100
            bar_length = 30
            filled_length = int(bar_length * processed // total_records)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            # Use \r to clear current line and display new progress
            sys.stdout.write(f"\rProgress: [{bar}] {processed}/{total_records} ({percent:.1f}%)")
            sys.stdout.flush()
    
    # Apply processing function
    patient_info_df['诊断信息嵌入'] = patient_info_df['主要诊断'].apply(process_with_progress)  # 诊断信息嵌入: Diagnosis information embedding, 主要诊断: Main diagnosis
    print("\nEmbedding processing complete.")

    return patient_info_df


if __name__ == "__main__":
    # Read merged data
    
    # test data
    
    # My diagnosis information comes from a column in dataframe, each row represents diagnosis information for different patients
    # Read merged data
    print("Starting diagnosis information embedding test")
    df = pd.DataFrame({'diagnosis_info': ['感冒 发烧 咳嗽', '感冒 发烧', '感冒 咳嗽']})
    
    print("Step 1: Splitting diagnosis info")
    df['diagnosis_info'] = df['diagnosis_info'].apply(get_diagnosis_names)
    print(df)
    
    print("Step 2: Translating to English (using Gemini API)")
    # Note: Gemini API integration should be completed here
    df['diagnosis_info'] = df['diagnosis_info'].apply(translate_diagnosis_names)
    print(df)
    
    print("Step 3: Creating embeddings")
    df['diagnosis_embeddings'] = df['diagnosis_info'].apply(embed_diagnosis_names)
    print("Embedding shapes:")
    for i in range(len(df)):
        if df.diagnosis_embeddings[i] is not None:
            print(f"  Row {i}: {df.diagnosis_embeddings[i].shape}")
    
    print("Step 4: Aggregating embeddings")
    df['diagnosis_embeddings'] = df['diagnosis_embeddings'].apply(aggregate_diagnosis_embeddings) # type: ignore
    
    print("Embedding process completed")
    for i in range(len(df)):
        if df.diagnosis_embeddings[i] is not None:
            print(f"  Row {i} final embedding shape: {df.diagnosis_embeddings[i].shape}")
    """
    Important notes:
    1. Before running this code, you need to first implement Google Gemini API integration
    2. Complete the parts marked as TODO
    3. Ensure all necessary dependencies are installed in the environment
    """

