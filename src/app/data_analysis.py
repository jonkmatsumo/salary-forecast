import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.app.caching import load_data_cached as load_data
from src.services.analytics_service import AnalyticsService

def render_data_analysis_ui() -> None:
    """Render the data analysis dashboard. Returns: None."""
    st.header("Data Analysis")
    
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
                st.success("File uploaded successfully!")
                st.info("💡 **Next Step**: Go to the **Configuration** page (sidebar) to generate a config from this data.")
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                
    if df is None:
        return
        
    analytics_service = AnalyticsService()
    summary = analytics_service.get_data_summary(df)



    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", summary.get("total_samples", 0))
    col2.metric("Unique Locations", summary.get("unique_locations", 0))
    col3.metric("Unique Levels", summary.get("unique_levels", 0))
    
    with st.expander("View Data Sample"):
        st.dataframe(df.head())
        st.caption(f"Shape: {summary.get('shape')}")
        
    st.markdown("---")
        
    st.subheader("Salary Distributions")
    
    salary_cols = [c for c in ["BaseSalary", "TotalComp", "Stock", "Bonus"] if c in df.columns]
    
    if salary_cols:
        target = st.selectbox("Select Component", salary_cols)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x=target, kde=True, ax=ax)
        ax.set_title(f"Distribution of {target}")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))
        st.pyplot(fig)
        
        st.write("Statistics:")
        st.dataframe(df[target].describe().T)
    
    st.markdown("---")

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
        
    st.subheader("Correlations")
    num_cols = ["YearsOfExperience", "YearsAtCompany"] + salary_cols
    avail_num_cols = [c for c in num_cols if c in df.columns]
    
    if len(avail_num_cols) > 1:
        corr = df[avail_num_cols].corr()
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
