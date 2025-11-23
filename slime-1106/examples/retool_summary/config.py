"""
Module: config
--------------
Configuration management for retool-summary dual-agent training system.
"""

import os


class Config:
    """
    Environment variable configuration class.

    Attributes:
        MAX_TURNS (int): Maximum number of tool interaction turns.
        MAX_MODEL_LEN (int): Maximum allowed token length for model inputs.
        CONTEXT_LENGTH_THRESHOLD (int): Threshold to trigger summary agent.
        SUMMARY_AGENT_IP (str): IP address of summary agent endpoint.
        SUMMARY_AGENT_PORT (int): Port of summary agent endpoint.
        DATABASE_SERVER_IP (str): IP address of database server.
        KEY_SUFFIX (str): Suffix used for generating unique database keys.
        JUDGE_MODEL_API_KEY (str): API key for judge model.
        JUDGE_MODEL_BASE_URL (str): Base URL for judge model.
        JUDGE_MODEL_NAME (str): Name of the judge model.
    """

    # Maximum tool interaction turns
    MAX_TURNS = int(os.getenv("MAX_TURNS", "16"))

    # Maximum model length
    MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "8192"))

    # Context length threshold to trigger summary
    CONTEXT_LENGTH_THRESHOLD = int(os.getenv("CONTEXT_LENGTH_THRESHOLD", "2048"))

    # Summary agent configuration
    SUMMARY_AGENT_IP = os.getenv("SUMMARY_AGENT_IP")
    SUMMARY_AGENT_PORT = int(os.getenv("SUMMARY_AGENT_PORT", "3333"))

    # Database server configuration
    DATABASE_SERVER_IP = os.getenv("DATABASE_SERVER_IP")

    # Database key configuration
    KEY_SUFFIX = os.getenv("KEY_SUFFIX")

    # Judge model configuration for summary evaluation
    JUDGE_MODEL_API_KEY = os.getenv("JUDGE_MODEL_API_KEY", "EMPTY")
    JUDGE_MODEL_BASE_URL = os.getenv("JUDGE_MODEL_BASE_URL", "")
    JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "gpt-4")

    # Tool sandbox configuration
    TOOL_MAX_TURNS = int(os.getenv("TOOL_MAX_TURNS", "16"))
    TOOL_MAX_CALLS = int(os.getenv("TOOL_MAX_CALLS", "16"))
    TOOL_TIMEOUT = int(os.getenv("TOOL_TIMEOUT", "120"))

    # Default reward score for aborted or failed samples
    DEFAULT_SCORE = float(os.getenv("DEFAULT_SCORE", "0.0"))

    def __init__(self):
        """Print configuration content on initialization"""
        print("=" * 50)
        print("Retool-Summary Configuration Loaded:")
        print(f"  MAX_TURNS: {self.MAX_TURNS}")
        print(f"  MAX_MODEL_LEN: {self.MAX_MODEL_LEN}")
        print(f"  CONTEXT_LENGTH_THRESHOLD: {self.CONTEXT_LENGTH_THRESHOLD}")
        print(f"  SUMMARY_AGENT_IP: {self.SUMMARY_AGENT_IP}")
        print(f"  SUMMARY_AGENT_PORT: {self.SUMMARY_AGENT_PORT}")
        print(f"  DATABASE_SERVER_IP: {self.DATABASE_SERVER_IP}")
        print(f"  KEY_SUFFIX: {self.KEY_SUFFIX}")
        print(f"  JUDGE_MODEL_NAME: {self.JUDGE_MODEL_NAME}")
        print(f"  TOOL_MAX_TURNS: {self.TOOL_MAX_TURNS}")
        print(f"  DEFAULT_SCORE: {self.DEFAULT_SCORE}")
        print("=" * 50)


# Create global configuration instance
global_config = Config()

