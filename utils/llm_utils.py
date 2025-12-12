import re
import json
import time
from typing import Dict, Optional, Tuple
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage

from logger import log_llm_response, log_message

# Global model list for fallback (ordered by preference)
AVAILABLE_MODELS = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it"
]

# Current model index (starts with first model)
current_model_index = 0


def get_llm(model_name: str = None):
    """
    Initialize the ChatGroq LLM with specified or default model.
    
    Args:
        model_name: Optional model name. If None, uses current global model.
        
    Returns:
        ChatGroq instance
    """
    if model_name is None:
        model_name = AVAILABLE_MODELS[current_model_index]
    
    return ChatGroq(
        model=model_name,
        temperature=0,
        streaming=False
    )


def switch_to_next_model():
    """
    Switch to the next available model in the global list.
    
    Returns:
        New model name or None if no models left
    """
    global current_model_index
    
    current_model_index += 1
    
    if current_model_index < len(AVAILABLE_MODELS):
        new_model = AVAILABLE_MODELS[current_model_index]
        log_message(f"Switching to fallback model: {new_model}", "WARNING")
        return new_model
    else:
        log_message("No more fallback models available!", "ERROR")
        return None


def parse_rate_limit_error(error_message: str) -> Dict[str, any]:
    """
    Parse rate limit error message to extract details.
    
    Args:
        error_message: Error message from API
        
    Returns:
        Dict with 'type' (TPM/TPD), 'retry_after' (seconds), and 'limit_type'
    """
    result = {
        'type': None,
        'retry_after': None,
        'limit_type': None
    }
    
    # Check for tokens per day (TPD)
    if 'tokens per day (TPD)' in error_message or 'TPD' in error_message:
        result['limit_type'] = 'TPD'
        result['type'] = 'tokens_per_day'
        
        # Extract retry time
        time_match = re.search(r'try again in ([\d]+h)?([\d]+m)?([\d.]+s)?', error_message)
        if time_match:
            hours = int(time_match.group(1).replace('h', '')) if time_match.group(1) else 0
            minutes = int(time_match.group(2).replace('m', '')) if time_match.group(2) else 0
            seconds = float(time_match.group(3).replace('s', '')) if time_match.group(3) else 0
            result['retry_after'] = hours * 3600 + minutes * 60 + seconds
    
    # Check for tokens per minute (TPM)
    elif 'tokens per minute (TPM)' in error_message or 'TPM' in error_message or 'Rate limit' in error_message:
        result['limit_type'] = 'TPM'
        result['type'] = 'tokens_per_minute'
        result['retry_after'] = 80  # Default to 80 seconds for TPM
        
        # Try to extract specific wait time
        time_match = re.search(r'try again in ([\d]+)s', error_message)
        if time_match:
            result['retry_after'] = int(time_match.group(1))
    
    # Check for request size error
    elif 'Request too large' in error_message or 'reduce your message size' in error_message:
        result['limit_type'] = 'REQUEST_SIZE'
        result['type'] = 'request_too_large'
    
    return result


def handle_rate_limit(error, retry_count: int = 0, max_retries: int = 3) -> Tuple[bool, int, bool]:
    """
    Handle rate limit errors with appropriate retry logic.
    
    Args:
        error: Exception from API call
        retry_count: Current retry attempt
        max_retries: Maximum retries allowed
        
    Returns:
        Tuple of (should_retry: bool, wait_time: int, use_new_model: bool)
    """
    error_str = str(error)
    rate_info = parse_rate_limit_error(error_str)
    
    log_message(f"Rate Limit Error Detected: Type={rate_info['limit_type']}, Retry={retry_count + 1}/{max_retries}", "WARNING")
    
    if rate_info['limit_type'] == 'TPM':
        # Tokens per minute - wait and retry
        wait_time = rate_info['retry_after'] or 80
        if retry_count < max_retries:
            log_message(f"Waiting {wait_time} seconds before retry...", "INFO")
            return (True, wait_time, False)
        else:
            log_message("Max retries reached for TPM limit", "ERROR")
            return (False, 0, False)
    
    elif rate_info['limit_type'] == 'TPD':
        # Tokens per day - switch model
        log_message("Daily token limit reached, switching to fallback model...", "WARNING")
        return (True, 0, True)
    
    elif rate_info['limit_type'] == 'REQUEST_SIZE':
        # Request too large - cannot retry
        log_message("Request size too large, cannot retry", "ERROR")
        return (False, 0, False)
    
    else:
        # Unknown error
        log_message("Unknown rate limit type", "WARNING")
        return (False, 0, False)


def parse_llm_json_response(content: str, max_retries: int = 3) -> dict:
    """
    Parse LLM response that should contain JSON.
    Handles various JSON formatting issues.
    
    Args:
        content: Response content from LLM
        max_retries: Number of parsing attempts
        
    Returns:
        Parsed JSON dict
        
    Raises:
        ValueError: If JSON cannot be parsed
    """
    for attempt in range(max_retries):
        try:
            # Try direct JSON parse
            return json.loads(content)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            if "```json" in content:
                json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except:
                        pass
            
            # Try extracting JSON from any code block
            if "```" in content:
                json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
                if json_match:
                    try:
                        return json.loads(json_match.group(1))
                    except:
                        pass
            
            # Try finding JSON object in text
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            # If last attempt, raise error
            if attempt == max_retries - 1:
                raise ValueError(f"Could not parse JSON from LLM response after {max_retries} attempts")
    
    raise ValueError("Failed to parse JSON")


def invoke_llm_with_retry(llm, prompt: str, max_retries: int = 3) -> dict:
    """
    Invoke LLM with automatic retry logic for rate limits.
    
    Args:
        llm: LLM instance
        prompt: Prompt string
        max_retries: Maximum retry attempts
        
    Returns:
        Parsed JSON response dict
        
    Raises:
        Exception: If all retries fail
    """
    retry_count = 0
    current_llm = llm
    
    while retry_count <= max_retries:
        try:
            # Invoke LLM
            response = current_llm.invoke([SystemMessage(content=prompt)])
            content = response.content.strip()
            
            # Parse JSON response
            parsed = parse_llm_json_response(content)
            
            # Log successful response
            log_llm_response(prompt, content, success=True)
            
            return parsed
            
        except Exception as e:
            error_str = str(e)
            
            # Check if it's a rate limit error
            if 'rate_limit_exceeded' in error_str or 'Rate limit' in error_str or 'TPM' in error_str or 'TPD' in error_str:
                should_retry, wait_time, use_new_model = handle_rate_limit(e, retry_count, max_retries)
                
                if should_retry:
                    if use_new_model:
                        # Switch to next model
                        new_model = switch_to_next_model()
                        if new_model:
                            current_llm = get_llm(new_model)
                        else:
                            raise Exception("No more fallback models available")
                    else:
                        # Wait and retry with same model
                        time.sleep(wait_time)
                    
                    retry_count += 1
                    continue
                else:
                    # Cannot retry
                    log_llm_response(prompt, f"ERROR: {error_str}", success=False)
                    raise
            
            # For other errors, check if it's JSON parsing issue
            elif 'Could not parse JSON' in error_str or 'JSONDecodeError' in error_str:
                log_message(f"JSON parsing error: {error_str}", "ERROR")
                log_llm_response(prompt, f"PARSE ERROR: {error_str}", success=False)
                raise
            
            # For unknown errors, retry once
            else:
                if retry_count < 1:  # Only retry once for unknown errors
                    log_message(f"Unknown error, retrying: {error_str}", "WARNING")
                    retry_count += 1
                    time.sleep(2)
                    continue
                else:
                    log_llm_response(prompt, f"ERROR: {error_str}", success=False)
                    raise
    
    raise Exception(f"Failed after {max_retries} retries")
