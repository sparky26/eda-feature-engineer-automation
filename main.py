# main.py - Main application entry point

from dotenv import load_dotenv
from ui_components import (
    setup_page_config,
    setup_sidebar_options,
    handle_file_upload,
    create_main_tabs
)
from data_processor import preprocess_dataframe

# Load environment variables
load_dotenv()


def main():
    """Main application function."""
    # Setup page configuration
    setup_page_config()
    
    # Handle file upload
    df_raw = handle_file_upload()
    if df_raw is None:
        return
    
    # Setup sidebar options
    preprocessing_options = setup_sidebar_options()
    
    # Preprocess the data
    df_clean = preprocess_dataframe(
        df_raw,
        impute_numeric=preprocessing_options['impute_numeric'],
        impute_categorical=preprocessing_options['impute_categorical'],
        scale_numeric=preprocessing_options['scale_numeric_val'],
        cap_outliers=preprocessing_options['cap_outliers'],
        drop_constant=preprocessing_options['drop_constant'],
        drop_high_corr=preprocessing_options['drop_high_corr'],
    )
    
    # Create main tabs interface
    create_main_tabs(df_raw, df_clean)


if __name__ == "__main__":
    main()