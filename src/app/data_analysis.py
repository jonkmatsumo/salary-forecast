import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.data_utils import load_data

def render_data_analysis_ui():
    st.header("Data Analysis")
    
    # 1. Data Loading / Retrieval
    df = None
    if "training_data" in st.session_state:
        df = st.session_state["training_data"]
        st.success(f"Using loaded data ({len(df)} rows).")
        
        if st.button("Clear Data"):
            del st.session_state["training_data"]
            st.rerun()
            
    else:
        st.info("No data loaded. Please upload a CSV.")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="analysis_uploader")
        if uploaded_file:
            try:
                df = load_data(uploaded_file)
                st.session_state["training_data"] = df
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                
    if df is None:
        return

    # 2. Overview
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Unique Locations", df["Location"].nunique())
    col3.metric("Unique Levels", df["Level"].nunique())
    
    with st.expander("View Data Sample"):
        st.dataframe(df.head())
        st.caption(f"Shape: {df.shape}")
        
    st.markdown("---")
        
    # 3. salary Distributions
    st.subheader("Salary Distributions")
    
    salary_cols = [c for c in ["BaseSalary", "TotalComp", "Stock", "Bonus"] if c in df.columns]
    
    if salary_cols:
        target = st.selectbox("Select Component", salary_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x=target, kde=True, ax=ax)
        ax.set_title(f"Distribution of {target}")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        st.pyplot(fig)
        
        # Stats
        st.write("Statistics:")
        st.dataframe(df[target].describe().T)
    
    st.markdown("---")

    # 4. Categorical Breakdown
    st.subheader("Categorical Breakdown")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Level Counts**")
        level_counts = df["Level"].value_counts()
        st.bar_chart(level_counts)
        
    with c2:
        st.markdown("**Top 20 Locations**")
        loc_counts = df["Location"].value_counts().head(20)
        st.bar_chart(loc_counts)
        
    # 5. Correlations (Numerical)
    st.subheader("Correlations")
    num_cols = ["YearsOfExperience", "YearsAtCompany"] + salary_cols
    # Filter only available columns
    avail_num_cols = [c for c in num_cols if c in df.columns]
    
    if len(avail_num_cols) > 1:
        corr = df[avail_num_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
