"""
Simple production-ready version - guaranteed to work.
For capstone presentation.
"""

import gradio as gr
from app_gradio_improved import SFDAChatbot, create_gradio_interface

def main():
    """Main entry point - simple and reliable."""
    print("=" * 60)
    print("🚀 SFDA Inspector Assistant - Starting...")
    print("=" * 60)

    # Initialize chatbot
    chatbot = SFDAChatbot()

    # Create interface
    demo = create_gradio_interface(chatbot)

    print("✅ Application ready!")
    print("=" * 60)

    # Launch
    demo.queue().launch(
        share=True,
        show_error=True,
        server_name="0.0.0.0",
        server_port=7860,
    )

if __name__ == "__main__":
    main()
