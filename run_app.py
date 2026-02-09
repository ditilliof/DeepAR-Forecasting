"""
Launch script for the single-page Streamlit app.
Run this file to start the forecasting application.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # Path to the new single-page app
    app_path = os.path.join(
        os.path.dirname(__file__),
        'src',
        'deepar_forecast',
        'app_ui',
        'app.py'
    )
    
    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=localhost",
    ]
    
    sys.exit(stcli.main())
