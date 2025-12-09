import streamlit as st

from src.utils.compatibility import apply_backward_compatibility
from src.app.train_ui import render_training_ui
from src.app.inference_ui import render_inference_ui
from src.utils.config_loader import get_config
from src.utils.logger import setup_logging

apply_backward_compatibility()


def main() -> None:
    """Main entry point for the Streamlit application. Returns: None."""
    st.set_page_config(page_title="Salary Forecaster", layout="wide")
    
    config = get_config()
    st.session_state["config_override"] = config
    st.sidebar.title("Navigation")
    
    if "nav" not in st.session_state:
        st.session_state["nav"] = "Training"
        
    options = ["Training", "Inference"]
    default_index = 0
    if st.session_state.get("nav") in options:
        default_index = options.index(st.session_state["nav"])
        
    nav = st.sidebar.radio("Go to", options, index=default_index, key="nav_radio")
    st.session_state["nav"] = nav
    
    if nav == "Training":
        render_training_ui()
    elif nav == "Inference":
        render_inference_ui()

if __name__ == "__main__":
    setup_logging()
    main()
