from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from src.gradcam import generate_all_gradcams


# =========================================================
# CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HISTOPATH AI | Stacking",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parent

STACKING_PATH = (
    PROJECT_ROOT / "results" / "stacking_meta_learner.joblib"
)

MODEL_NAMES = {
    "convnext_tiny": "ConvNeXt-Tiny",
    "swin_t": "Swin-Tiny",
    "swin_tiny": "Swin-Tiny",
    "densenet121": "DenseNet121",
    "resnet50_improved": "ResNet50",
    "efficientnet_b0": "EfficientNet-B0",
}

MODEL_ORDER = [
    "convnext_tiny",
    "swin_t",
    "densenet121",
    "resnet50_improved",
    "efficientnet_b0",
]


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --navy: #0B1120;
        --panel: #111B2E;
        --panel2: #162238;
        --teal: #5EEAD4;
        --muted: #94A3B8;
        --border: rgba(148,163,184,0.19);
        --white: #F1F5F9;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, #12243A 0%, transparent 42%),
            var(--navy);
        color: var(--white);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--white) !important;
        letter-spacing: -0.4px;
    }

    [data-testid="stSidebar"] {
        background: #0D1728;
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: var(--white);
    }

    .hero {
        padding: 25px 28px;
        border: 1px solid rgba(94,234,212,0.24);
        border-radius: 22px;
        background: linear-gradient(125deg, #13253B 0%, #101A2C 60%, #102B32 100%);
        margin-bottom: 24px;
    }

    .eyebrow {
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.2px;
        text-transform: uppercase;
    }

    .hero-title {
        font-family: 'Space Grotesk', sans-serif;
        color: #F8FAFC;
        font-size: clamp(28px, 4vw, 43px);
        font-weight: 700;
        line-height: 1.12;
        margin: 9px 0;
    }

    .hero-subtitle {
        color: #B7C5D8;
        font-size: 14px;
        line-height: 1.7;
        max-width: 760px;
    }

    .section-label {
        color: var(--teal);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.8px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .panel {
        background: linear-gradient(145deg, #142137, #101A2B);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 18px;
        margin-bottom: 12px;
    }

    .model-card {
        background: linear-gradient(145deg, #17263D, #111B2D);
        border: 1px solid rgba(148,163,184,0.2);
        border-radius: 16px;
        padding: 17px;
        min-height: 155px;
        margin-bottom: 12px;
    }

    .model-name {
        color: #E2E8F0;
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 12px;
    }

    .model-prediction {
        color: var(--teal);
        font-size: 17px;
        font-weight: 700;
        overflow-wrap: anywhere;
        margin-bottom: 8px;
    }

    .small-label {
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
    }

    .metric-panel {
        background: linear-gradient(135deg, #14283A, #10232D);
        border: 1px solid rgba(94,234,212,0.3);
        border-radius: 17px;
        padding: 20px;
        min-height: 130px;
    }

    .metric-title {
        color: #A8B9CC;
        font-size: 12px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: var(--teal);
        font-size: 23px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #A8B9CC;
        font-size: 11px;
        margin-top: 8px;
    }

    .upload-note {
        color: #94A3B8;
        font-size: 12px;
        line-height: 1.6;
    }

    .disclaimer {
        border: 1px solid rgba(251,191,36,0.3);
        background: rgba(120,85,20,0.12);
        border-radius: 12px;
        padding: 14px 16px;
        color: #FDE68A;
        font-size: 12px;
        line-height: 1.65;
        margin-top: 20px;
    }

    div.stButton > button[kind="primary"] {
        background: var(--teal);
        color: #0B1120;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        min-height: 45px;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #99F6E4;
        color: #0B1120;
        border: none;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stExpander"] {
        border: 1px solid var(--border);
        border-radius: 13px;
        background: rgba(17,27,46,0.6);
    }

    div[data-testid="stFileUploader"] {
        background: rgba(17,27,46,0.7);
        border: 1px dashed rgba(94,234,212,0.45);
        border-radius: 15px;
        padding: 12px;
    }

    hr {
        border-color: var(--border);
    }

    footer {
        visibility: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def canonical_key(key):
    return "swin_t" if key == "swin_tiny" else key


def unique_results(results):
    output = {}

    for key in MODEL_ORDER:
        if key in results:
            output[key] = results[key]
        elif key == "swin_t" and "swin_tiny" in results:
            output[key] = results["swin_tiny"]

    for key, value in results.items():
        normalized = canonical_key(key)
        if normalized not in output:
            output[normalized] = value

    return output


def get_class_mapping(results):
    mapping = None

    for result in results.values():
        current = result.get("class_to_idx")

        if not isinstance(current, dict):
            continue

        if mapping is None:
            mapping = current
        elif current != mapping:
            return None

    return mapping


def make_individual_table(results, mapping):
    idx_to_class = {
        int(index): label
        for label, index in mapping.items()
    }

    rows = []

    for model_key, result in results.items():
        values = result.get("probabilities")

        if values is None or result.get("class_to_idx") != mapping:
            continue

        values = np.asarray(values, dtype=np.float64).reshape(-1)

        if len(values) != len(mapping):
            continue

        if not np.isfinite(values).all() or (values < 0).any():
            continue

        row = {"Model": MODEL_NAMES.get(model_key, model_key)}

        for index, probability in enumerate(values):
            row[idx_to_class.get(index, f"Class {index}")] = round(
                float(probability) * 100, 2
            )

        rows.append(row)

    return pd.DataFrame(rows).set_index("Model") if rows else None


def make_probability_table(probabilities, class_to_idx):
    idx_to_class = {
        int(index): label
        for label, index in class_to_idx.items()
    }

    rows = []

    for index, probability in enumerate(probabilities):
        rows.append({
            "Cancer Type": idx_to_class.get(index, f"Class {index}"),
            "Probability (%)": round(float(probability) * 100, 2),
        })

    return pd.DataFrame(rows).sort_values(
        "Probability (%)",
        ascending=False,
    ).reset_index(drop=True)


def show_model_cards(results):
    model_keys = list(results.keys())

    for start in range(0, len(model_keys), 3):
        group = model_keys[start:start + 3]
        columns = st.columns(3)

        for column, model_key in zip(columns, group):
            result = results[model_key]
            prediction = result.get("predicted_class", "Unavailable")

            try:
                confidence = float(result.get("confidence", 0))
            except (ValueError, TypeError):
                confidence = 0.0

            with column:
                st.markdown(
                    f"""
                    <div class="model-card">
                        <div class="small-label">BASE MODEL</div>
                        <div class="model-name">
                            {MODEL_NAMES.get(model_key, model_key)}
                        </div>
                        <div class="small-label">PREDICTED CLASS</div>
                        <div class="model-prediction">
                            {prediction}
                        </div>
                        <div class="small-label">
                            Predicted-class probability
                        </div>
                        <div style="color:#E2E8F0;font-size:19px;font-weight:700;">
                            {confidence:.2f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def show_gradcam(results):
    st.markdown(
        '<div class="section-label">Explainable AI</div>',
        unsafe_allow_html=True,
    )
    st.header("Grad-CAM Explainability")

    st.caption(
        "Heatmaps visualize image regions influencing each model. "
        "They do not establish a clinical diagnosis."
    )

    for model_key, result in results.items():
        with st.expander(
            f"{MODEL_NAMES.get(model_key, model_key)} · "
            f"{result.get('predicted_class', 'Unknown')}"
        ):
            columns = st.columns(3)

            image_items = [
                ("original", "Original Image"),
                ("heatmap", "Grad-CAM Heatmap"),
                ("overlay", "Heatmap Overlay"),
            ]

            for column, (field, label) in zip(columns, image_items):
                with column:
                    value = result.get(field)

                    if value is not None:
                        st.image(
                            np.asarray(value),
                            caption=label,
                            use_container_width=True,
                            clamp=(field != "original"),
                        )
                    else:
                        st.info(f"{label} unavailable.")


@st.cache_resource
def load_stacking_artifact(path_string):
    """Load the trained meta-learner once per Streamlit session."""
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(
            f"Stacking artifact not found: {path}"
        )

    artifact = joblib.load(path)

    required = {
        "meta_model",
        "class_to_idx",
        "base_models",
        "num_classes",
        "input_features",
    }

    missing = required - set(artifact.keys())

    if missing:
        raise ValueError(
            f"Stacking artifact is missing fields: {sorted(missing)}"
        )

    return artifact


def prepare_stacking_features(results, artifact):
    """Build the exact feature order expected by the trained meta-learner."""

    class_to_idx = artifact["class_to_idx"]
    num_classes = int(artifact["num_classes"])
    expected_features = int(artifact["input_features"])
    probability_blocks = []
    errors = []

    for model_key, checkpoint_name in artifact["base_models"]:
        result = results.get(model_key)

        if not isinstance(result, dict):
            errors.append(f"{model_key}: model result is missing.")
            continue

        if result.get("class_to_idx") != class_to_idx:
            errors.append(f"{model_key}: class mapping does not match.")
            continue

        probabilities = result.get("probabilities")

        if probabilities is None:
            errors.append(f"{model_key}: probabilities are missing.")
            continue

        probabilities = np.asarray(
            probabilities, dtype=np.float64
        ).reshape(-1)

        if len(probabilities) != num_classes:
            errors.append(
                f"{model_key}: expected {num_classes} probabilities, "
                f"received {len(probabilities)}."
            )
            continue

        if not np.isfinite(probabilities).all():
            errors.append(f"{model_key}: probabilities contain invalid values.")
            continue

        if (probabilities < 0).any():
            errors.append(f"{model_key}: negative probabilities detected.")
            continue

        total = probabilities.sum()

        if total <= 0:
            errors.append(f"{model_key}: probability sum is zero.")
            continue

        probability_blocks.append(probabilities / total)

    if errors:
        return None, errors

    features = np.concatenate(probability_blocks).reshape(1, -1)

    if features.shape[1] != expected_features:
        return None, [
            f"Feature count mismatch: expected {expected_features}, "
            f"received {features.shape[1]}."
        ]

    return features, []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown(
        """
        <div style="padding:10px 0 22px 0;">
            <div style="font-size:30px;">🔬</div>
            <div style="font-family:'Space Grotesk',sans-serif;
                        font-size:22px;font-weight:700;color:#5EEAD4;">
                HISTOPATH AI
            </div>
            <div style="font-size:11px;color:#94A3B8;letter-spacing:1px;">
                CANCER CLASSIFICATION
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Navigation")
    st.markdown("🧪  Image Analysis")
    st.markdown("🧠  Individual Models")
    st.markdown("🕸️  Stacking Ensemble")
    st.markdown("🌡️  Explainability")

    st.divider()

    st.markdown("### Base Models")
    for key in MODEL_ORDER:
        st.markdown(f"• {MODEL_NAMES[key]}")

    st.divider()
    st.caption("Ensemble method: Stacking")
    st.caption("Meta-learner: Logistic Regression")


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">AI-POWERED HISTOPATHOLOGY ANALYSIS</div>
        <div class="hero-title">Explore a smarter<br>ensemble prediction.</div>
        <div class="hero-subtitle">
            Analyze a histopathology image using five trained deep-learning
            models. Their class probabilities are passed to a trained
            Logistic Regression meta-learner for the stacking prediction.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="disclaimer"><b>Research prototype</b><br>'
    'This application is intended for educational and research purposes. '
    'It is not a clinically validated diagnostic tool and must not be used '
    'to make medical decisions.</div>',
    unsafe_allow_html=True,
)

st.write("")


# =========================================================
# LOAD STACKING ARTIFACT
# =========================================================

try:
    artifact = load_stacking_artifact(str(STACKING_PATH))
    artifact_error = None
except Exception as exc:
    artifact = None
    artifact_error = str(exc)


# =========================================================
# UPLOAD PANEL
# =========================================================

st.markdown(
    '<div class="section-label">01 / IMAGE INPUT</div>',
    unsafe_allow_html=True,
)
st.header("Upload Histopathology Image")

left, right = st.columns([1.15, 0.85], gap="large")

with left:
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload one histopathology image for analysis.",
        key="stacking_upload",
    )

    st.markdown(
        '<div class="upload-note">Supported formats: JPG, JPEG, PNG, TIFF. '
        'Use a clear tissue image for research analysis.</div>',
        unsafe_allow_html=True,
    )

with right:
    st.markdown(
        """
        <div class="panel">
            <div class="small-label">ANALYSIS PIPELINE</div>
            <p style="color:#E2E8F0;font-weight:700;">5 Base Models</p>
            <p style="color:#94A3B8;font-size:13px;">
                ConvNeXt · Swin · DenseNet · ResNet · EfficientNet
            </p>
            <hr>
            <div class="small-label">ENSEMBLE METHOD</div>
            <p style="color:#5EEAD4;font-weight:700;">Stacking</p>
            <p style="color:#94A3B8;font-size:13px;">
                A trained Logistic Regression meta-learner combines the
                base-model probability outputs.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if artifact_error:
    st.error(f"Could not load the stacking meta-learner: {artifact_error}")

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        preview_col, details_col = st.columns([1, 1], gap="large")

        with preview_col:
            st.image(
                image,
                caption="Uploaded tissue image",
                use_container_width=True,
            )

        with details_col:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown("**Image details**")
            st.write(f"Filename: `{uploaded_file.name}`")
            st.write(f"Width: `{image.width}px`")
            st.write(f"Height: `{image.height}px`")
            st.write("Channels: `RGB / 3`")
            st.markdown("</div>", unsafe_allow_html=True)

        if st.button(
            "🔬  Analyze Image",
            type="primary",
            use_container_width=True,
            key="stacking_analyze",
        ):
            with st.spinner(
                "Running the five models and generating Grad-CAM maps..."
            ):
                analysis = generate_all_gradcams(
                    image,
                    save_results=False,
                    image_name=Path(uploaded_file.name).stem,
                )

            results = unique_results(dict(analysis.get("results", {})))

            if results:
                st.session_state["stacking_analysis"] = {
                    "results": results,
                    "errors": analysis.get("errors", {}),
                    "device": analysis.get("device", "unknown"),
                    "filename": uploaded_file.name,
                }
            else:
                st.error(
                    "No model returned a result. Check the terminal and "
                    "model checkpoints."
                )

    except Exception as exc:
        st.error(f"Could not read the uploaded image: {exc}")


# =========================================================
# RESULTS
# =========================================================

saved = st.session_state.get("stacking_analysis")

if uploaded_file is not None and saved is not None:
    if saved["filename"] == uploaded_file.name:
        results = unique_results(saved["results"])

        st.divider()
        st.markdown(
            '<div class="section-label">02 / MODEL OUTPUTS</div>',
            unsafe_allow_html=True,
        )
        st.header("Individual Model Predictions")

        st.caption(
            f"Analysis device: {saved['device']} · "
            f"{len(results)} model result(s) available"
        )

        if saved.get("errors"):
            with st.expander("Model warnings"):
                for key, message in saved["errors"].items():
                    st.warning(f"{key}: {message}")

        for start in range(0, len(results), 3):
            group = list(results.items())[start:start + 3]
            columns = st.columns(3)

            for column, (model_key, result) in zip(columns, group):
                prediction = result.get(
                    "predicted_class", "Unavailable"
                )

                try:
                    confidence = float(result.get("confidence", 0))
                except (ValueError, TypeError):
                    confidence = 0.0

                with column:
                    st.markdown(
                        f"""
                        <div class="model-card">
                            <div class="small-label">BASE MODEL</div>
                            <div class="model-name">
                                {MODEL_NAMES.get(model_key, model_key)}
                            </div>
                            <div class="small-label">PREDICTED CLASS</div>
                            <div class="model-prediction">
                                {prediction}
                            </div>
                            <div class="small-label">
                                Predicted-class probability
                            </div>
                            <div style="color:#E2E8F0;font-size:19px;font-weight:700;">
                                {confidence:.2f}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        mapping = get_class_mapping(results)

        if mapping is None:
            st.error(
                "The available models have missing or inconsistent class "
                "mappings. Probability comparisons cannot be calculated."
            )
        else:
            st.subheader("Individual Class Probability Matrix")

            individual_df = make_individual_table(results, mapping)

            if individual_df is not None:
                st.dataframe(
                    individual_df,
                    use_container_width=True,
                    height=280,
                )

                st.caption(
                    "Rows represent models; columns represent the cancer "
                    "classes. Values are model output probabilities (%)."
                )
            else:
                st.warning("Individual probability matrix is unavailable.")

        # -------------------------------------------------
        # STACKING ONLY — NO SOFT VOTING HERE
        # -------------------------------------------------

        st.divider()
        st.markdown(
            '<div class="section-label">03 / ENSEMBLE ANALYSIS</div>',
            unsafe_allow_html=True,
        )
        st.header("Stacking Ensemble")

        if artifact is None:
            st.error(
                "Stacking is unavailable. Check the saved meta-learner file."
            )
        else:
            class_to_idx = artifact["class_to_idx"]
            num_classes = int(artifact["num_classes"])
            idx_to_class = {
                int(index): label
                for label, index in class_to_idx.items()
            }

            if len(class_to_idx) != num_classes:
                st.error(
                    "The class mapping size does not match the stacking "
                    "artifact's num_classes value."
                )
            else:
                features, feature_errors = prepare_stacking_features(
                    results, artifact
                )

                if feature_errors:
                    for message in feature_errors:
                        st.error(message)

                    st.info(
                        "Stacking needs valid probability outputs from all "
                        "five base models, in the same order used during "
                        "meta-learner training."
                    )
                else:
                    meta_model = artifact["meta_model"]

                    if not hasattr(meta_model, "predict_proba"):
                        st.error(
                            "The saved meta-learner does not provide "
                            "predict_proba, so the probability table "
                            "cannot be displayed."
                        )
                    else:
                        meta_probabilities = np.asarray(
                            meta_model.predict_proba(features)[0],
                            dtype=np.float64,
                        )

                        model_classes = np.asarray(
                            meta_model.classes_
                        )

                        if len(meta_probabilities) != len(model_classes):
                            st.error(
                                "The meta-learner's class labels do not "
                                "match its probability output."
                            )
                        else:
                            class_probabilities = {}

                            for class_index, probability in zip(
                                model_classes, meta_probabilities
                            ):
                                class_probabilities[int(class_index)] = (
                                    float(probability)
                                )

                            predicted_index = max(
                                class_probabilities,
                                key=class_probabilities.get,
                            )

                            predicted_class = idx_to_class.get(
                                predicted_index
                            )

                            if predicted_class is None:
                                st.error(
                                    "The meta-learner predicted a class "
                                    "index that is not in the saved "
                                    "class mapping."
                                )
                            else:
                                confidence = (
                                    class_probabilities[predicted_index]
                                    * 100
                                )

                                col1, col2, col3 = st.columns(3)

                                with col1:
                                    st.markdown(
                                        f"""
                                        <div class="metric-panel">
                                            <div class="metric-title">
                                                FINAL PREDICTED CLASS
                                            </div>
                                            <div class="metric-value">
                                                {predicted_class}
                                            </div>
                                            <div class="metric-note">
                                                Stacking meta-learner output
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                with col2:
                                    st.markdown(
                                        f"""
                                        <div class="metric-panel">
                                            <div class="metric-title">
                                                PREDICTED-CLASS PROBABILITY
                                            </div>
                                            <div class="metric-value">
                                                {confidence:.2f}%
                                            </div>
                                            <div class="metric-note">
                                                Logistic Regression output
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                with col3:
                                    st.markdown(
                                        f"""
                                        <div class="metric-panel">
                                            <div class="metric-title">
                                                META-LEARNER
                                            </div>
                                            <div class="metric-value">
                                                Logistic Regression
                                            </div>
                                            <div class="metric-note">
                                                Trained on base-model outputs
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )

                                # Fill all class probabilities using the
                                # actual class indices from the meta-model.
                                full_probabilities = np.zeros(
                                    num_classes,
                                    dtype=np.float64,
                                )

                                for class_index, probability in (
                                    class_probabilities.items()
                                ):
                                    if 0 <= class_index < num_classes:
                                        full_probabilities[class_index] = (
                                            probability
                                        )

                                st.subheader(
                                    "All Stacking Class Probabilities"
                                )

                                stacking_df = make_probability_table(
                                    full_probabilities,
                                    class_to_idx,
                                )

                                st.dataframe(
                                    stacking_df,
                                    use_container_width=True,
                                    height=470,
                                    hide_index=True,
                                )

                                st.download_button(
                                    "⬇️ Download Stacking Probabilities (CSV)",
                                    data=stacking_df.to_csv(
                                        index=False
                                    ).encode("utf-8"),
                                    file_name="stacking_probabilities.csv",
                                    mime="text/csv",
                                )

                                with st.expander(
                                    "View top 5 class probabilities"
                                ):
                                    for _, row in stacking_df.head(5).iterrows():
                                        st.write(
                                            f"**{row['Cancer Type']}** — "
                                            f"{row['Probability (%)']:.2f}%"
                                        )

                                        st.progress(
                                            min(
                                                max(
                                                    float(
                                                        row["Probability (%)"]
                                                    ) / 100,
                                                    0,
                                                ),
                                                1,
                                            )
                                        )

                                st.caption(
                                    "These values are produced by the trained "
                                    "stacking meta-learner. They are not "
                                    "clinically validated certainty."
                                )

        # -------------------------------------------------
        # GRAD-CAM
        # -------------------------------------------------

        st.divider()
        st.markdown(
            '<div class="section-label">04 / EXPLAINABILITY</div>',
            unsafe_allow_html=True,
        )
        st.header("Grad-CAM Explainability")

        st.caption(
            "Heatmaps visualize image regions influencing each model. "
            "They do not establish a clinical diagnosis."
        )

        for model_key, result in results.items():
            with st.expander(
                f"{MODEL_NAMES.get(model_key, model_key)} · "
                f"{result.get('predicted_class', 'Unknown')}"
            ):
                columns = st.columns(3)

                image_items = [
                    ("original", "Original Image"),
                    ("heatmap", "Grad-CAM Heatmap"),
                    ("overlay", "Heatmap Overlay"),
                ]

                for column, (field, label) in zip(columns, image_items):
                    with column:
                        value = result.get(field)

                        if value is not None:
                            st.image(
                                np.asarray(value),
                                caption=label,
                                use_container_width=True,
                                clamp=(field != "original"),
                            )
                        else:
                            st.info(f"{label} unavailable.")


# =========================================================
# FOOTER
# =========================================================

st.divider()
st.markdown(
    """
    <div style="text-align:center;padding:15px 0 5px 0;">
        <div style="color:#5EEAD4;font-weight:700;letter-spacing:1.5px;">
            HISTOPATH AI
        </div>
        <div style="color:#94A3B8;font-size:11px;margin-top:7px;">
            Final-Year Engineering Research Project · Stacking Frontend
        </div>
        <div style="color:#64748B;font-size:10px;margin-top:5px;">
            Research use only · Not a clinical diagnostic system
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)