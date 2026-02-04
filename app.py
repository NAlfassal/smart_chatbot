import sys
from src.config import validate_config
from src.utils.logger import logger
from src.ui_gradio import main

def start_application():
    try:
        logger.info("--- Starting Smart Chatbot System ---")
        
        # 1. Configuration and Path Validation
        logger.info("Validating system configuration...")
        validate_config()
        
        # 2. Launch UI
        logger.info("Launching Gradio interface...")
        main()
        
    except Exception as e:
        logger.critical(f"Application failed to start due to a critical error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
