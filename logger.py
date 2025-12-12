import logging
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

# Log directory and file paths
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

GENERAL_LOG_FILE = LOG_DIR / "app.log"
LLM_LOG_FILE = LOG_DIR / "llm_responses.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"

# Create formatters
DETAILED_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

SIMPLE_FORMATTER = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

LLM_FORMATTER = logging.Formatter(
    '\n%(asctime)s\n%(message)s\n' + '='*80,
    datefmt='%Y-%m-%d %H:%M:%S'
)


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    
    Args:
        name: Logger name
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(SIMPLE_FORMATTER)
    logger.addHandler(console_handler)
    
    # File handler for general logs (DEBUG and above)
    file_handler = logging.FileHandler(GENERAL_LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(DETAILED_FORMATTER)
    logger.addHandler(file_handler)
    
    # Error file handler (ERROR and above)
    error_handler = logging.FileHandler(ERROR_LOG_FILE, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(DETAILED_FORMATTER)
    logger.addHandler(error_handler)
    
    return logger


def setup_llm_logger() -> logging.Logger:
    """
    Set up a dedicated logger for LLM interactions.
    
    Returns:
        LLM logger instance
    """
    logger = logging.getLogger('llm')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Don't propagate to root logger
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # LLM-specific file handler
    llm_handler = logging.FileHandler(LLM_LOG_FILE, encoding='utf-8')
    llm_handler.setLevel(logging.DEBUG)
    llm_handler.setFormatter(LLM_FORMATTER)
    logger.addHandler(llm_handler)
    
    return logger


# Create logger instances
app_logger = setup_logger('app')
agent_logger = setup_logger('agent')
api_logger = setup_logger('api')
llm_logger = setup_llm_logger()


def log_llm_response(prompt: str, response: str, success: bool = True, model: str = "unknown"):
    """
    Log LLM prompt and response to dedicated LLM log file.
    
    Args:
        prompt: The prompt sent to the LLM
        response: The response received from the LLM
        success: Whether the response was successfully parsed
        model: The LLM model used
    """
    try:
        status = '✓ SUCCESS' if success else '✗ FAILED'
        truncated_prompt = prompt[:500] + ("..." if len(prompt) > 500 else "")
        
        log_message = f"""
{'='*80}
STATUS: {status}
MODEL: {model}
{'-'*80}
PROMPT:
{truncated_prompt}
{'-'*80}
RESPONSE:
{response}
"""
        llm_logger.debug(log_message)
        
    except Exception as e:
        agent_logger.error(f"Failed to log LLM response: {e}")


def log_node_execution(node_name: str, message: str, level: str = "info"):
    """
    Log node execution with appropriate level.
    
    Args:
        node_name: Name of the node
        message: Log message
        level: Logging level (debug, info, warning, error)
    """
    log_func = getattr(agent_logger, level.lower(), agent_logger.info)
    log_func(f"[{node_name}] {message}")


def log_api_call(api_name: str, message: str, success: bool = True):
    """
    Log API calls.
    
    Args:
        api_name: Name of the API
        message: Log message
        success: Whether the call was successful
    """
    if success:
        api_logger.info(f"[{api_name}] {message}")
    else:
        api_logger.error(f"[{api_name}] {message}")


# Legacy compatibility functions (deprecated, but kept for backward compatibility)
def log_message(message: str, level: str = "INFO"):
    """
    DEPRECATED: Use app_logger directly instead.
    Log a general message.
    """
    log_func = {
        "DEBUG": app_logger.debug,
        "INFO": app_logger.info,
        "WARNING": app_logger.warning,
        "ERROR": app_logger.error,
        "CRITICAL": app_logger.critical,
    }.get(level.upper(), app_logger.info)
    
    log_func(message)
