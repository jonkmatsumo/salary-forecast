import traceback
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.app.service_factories import get_analytics_service, get_inference_service
from src.services.inference_service import ModelNotFoundError
from src.services.model_registry import ModelRegistry


def render_model_analysis_ui() -> None:
    """Render the model analysis dashboard. Returns: None."""
    st.header("Model Analysis")

    registry = ModelRegistry()
    runs = registry.list_models()

    if not runs:
        st.warning("No models found in MLflow. Please train a new model.")
        return

    def fmt_score(x: Any) -> str:
        try:
            return f"{float(x):.4f}"
        except (ValueError, TypeError):
            return str(x)

    run_options = {
        f"{r['start_time'].strftime('%Y-%m-%d %H:%M')} | CV:{fmt_score(r.get('metrics.cv_mean_score', 'N/A'))} | ID:{r['run_id'][:8]}": r[
            "run_id"
        ]
        for r in runs
    }

    selected_label = st.selectbox("Select Model Version", options=list(run_options.keys()))
    if not selected_label:
        return

    selected_run_id = run_options[selected_label]

    try:
        inference_service = get_inference_service()
        model = inference_service.load_model(selected_run_id)
        schema = inference_service.get_model_schema(model)
        
        st.success(f"Loaded Run: {selected_run_id}")

        st.subheader("Feature Importance")
        st.info("Visualize which features drive the predictions (Gain metric).")

        analytics_service = get_analytics_service()
        targets = schema.targets

        if not targets:
            st.error("This model file does not appear to contain trained models.")
            return

        selected_target = st.selectbox("Select Target Component", targets)

        if selected_target:
            quantiles = schema.quantiles
            if not quantiles:
                st.warning(f"No quantiles available for target {selected_target}.")
                return
                
            selected_q_val = st.selectbox(
                "Select Quantile", quantiles, format_func=lambda x: f"P{int(x*100)}"
            )

            df_imp = analytics_service.get_feature_importance(
                model, selected_target, selected_q_val
            )
            if df_imp is None or df_imp.empty:
                st.warning(
                    f"No feature importance scores found for {selected_target} at P{int(selected_q_val*100)}."
                )
            else:
                st.dataframe(df_imp, width="stretch")

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(
                    data=df_imp.head(20),
                    x="Gain",
                    y="Feature",
                    ax=ax,
                    palette="viridis",
                    hue="Feature",
                    legend=False,
                )
                ax.set_title(f"Top 20 Features for {selected_target} (P{int(selected_q_val*100)})")
                st.pyplot(fig)

    except ModelNotFoundError as e:
        st.error(f"Model not found: {e}")
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.code(traceback.format_exc())
