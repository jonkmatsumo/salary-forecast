import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.app.caching import load_data_cached as load_data
from src.services.analytics_service import AnalyticsService
from src.services.config_generator import ConfigGenerator

def render_data_analysis_ui() -> None:
    """Renders the data analysis dashboard."""
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
                st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {e}")
                
    if df is None:
        return
        
    analytics_service = AnalyticsService()
    summary = analytics_service.get_data_summary(df)

    # --- Config Generation Section ---
    with st.expander("Generate Configuration from Data", expanded=False):
        st.info("Automatically generate a configuration based on this dataset. You can map columns to specific roles.")
        
        generator = ConfigGenerator()
        
        # 1. Column Mapping using Data Editor
        st.subheader("1. Map Columns")
        
        # Default guess
        all_cols = df.columns.tolist()
        mapping_data = []
        for col in all_cols:
            role = "Ignore"
            monotone = 0
            if col in ["Level_Enc", "Location_Enc"]: role = "Ignore" # engineered
            elif col in ["Level", "Location", "Date"]: role = "Ignore" # base
            elif col in ["BaseSalary", "TotalComp", "Stock", "Bonus"]: role = "Target"
            elif col in ["YearsOfExperience"]: 
                role = "Feature"
                monotone = 1
            elif col in ["YearsAtCompany"]:
                role = "Feature"
            
            mapping_data.append({
                "Column": col,
                "Role": role,
                "Monotone": monotone
            })
            
        mapping_df = pd.DataFrame(mapping_data)
        
        edited_mapping = st.data_editor(
            mapping_df,
            column_config={
                "Column": st.column_config.TextColumn("Column Name", disabled=True),
                "Role": st.column_config.SelectboxColumn("Role", options=["Feature", "Target", "Ignore"], required=True),
                "Monotone": st.column_config.SelectboxColumn("Monotone Constraint", options=[-1, 0, 1], help="1: Increasing, -1: Decreasing, 0: None")
            },
            hide_index=True,
            width="stretch",
            key="config_gen_mapping"
        )
        
        # 2. Level Inference Review
        st.subheader("2. Review Levels")
        base_levels = generator.infer_levels(df)
        level_data = [{"Level": k, "Rank": v} for k, v in base_levels.items()]
        levels_df = pd.DataFrame(level_data)
        
        edited_levels = st.data_editor(
            levels_df,
            column_config={
                "Level": st.column_config.TextColumn("Level", disabled=True),
                "Rank": st.column_config.NumberColumn("Rank", min_value=0, step=1)
            },
            hide_index=True,
            width="stretch",
            key="config_gen_levels"
        )

        if st.button("Apply Configuration"):
            # Construct Config
            new_levels = {row["Level"]: int(row["Rank"]) for _, row in edited_levels.iterrows()}
            new_locs = generator.infer_locations(df)
            
            # Extract Features and Targets
            new_features = []
            new_targets = []
            
            for _, row in edited_mapping.iterrows():
                if row["Role"] == "Feature":
                    # If it's a raw column like YearsOfExperience, we use it directly.
                    # If the user selected Level/Location as Feature? Ideally we use the encoded versions.
                    # For MVP, we assume they select the raw cols and we auto-add the encoders if Level/Location exist?
                    # Or we just add specific logic:
                    feature = {"name": row["Column"], "monotone_constraint": int(row["Monotone"])}
                    new_features.append(feature)
                elif row["Role"] == "Target":
                    new_targets.append(row["Column"])
                    
            # Auto-add encoders if not present but needed?
            # The current system expects "Level_Enc" and "Location_Enc" features if we want to use them.
            # But the user interacts with "Level" and "Location" columns.
            # We should probably force add the encoders if Level/Location columns exist in DF, 
            # OR rely on the user to map "Level" -> Feature (and we interpret that as Level_Enc?)
            # Let's keep it simple: We auto-add Level_Enc/Location_Enc if Level/Location columns exist, 
            # and append user-selected numeric features.
            
            # Actually, robust way:
            final_features = []
            if "Level" in df.columns:
                final_features.append({"name": "Level_Enc", "monotone_constraint": 1})
            if "Location" in df.columns:
                final_features.append({"name": "Location_Enc", "monotone_constraint": 0})
                
            # Add user selected (avoid dupes)
            for f in new_features:
                if f["name"] not in ["Level", "Location", "Date"]:
                   final_features.append(f)
                   
            if not new_targets:
                st.error("Please select at least one Target column.")
            else:
                template = generator.generate_config_template(df)
                template["mappings"]["levels"] = new_levels
                template["mappings"]["location_targets"] = new_locs
                template["model"]["targets"] = new_targets
                template["model"]["features"] = final_features
                
                st.session_state["config_override"] = template
                st.success("Configuration generated and applied! Visit 'Configuration' page to view/save.")
                st.balloons()
                
    st.markdown("---")

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
