import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import base64
import os
import streamlit.components.v1 as components
from models_dl import RNNTextClassifierV2, LSTMTextClassifier, CNNTextClassifier
import torch
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


from utils import (
    text_preprocessing,
    lemmatize_text,
    read_txt,
    read_docx,
    read_pdf,
    explain_prediction_with_lime,
    generate_comparison_report_pdf_full,
)
from llm_utils import run_ai_judge, run_plain_explanation, finetuned_judge_available


# NLTK resources
@st.cache_resource
def download_nltk_resources():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "tagsets",
        "omw-1.4",
    ]
    paths = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng",
        "tagsets": "help/tagsets",
        "omw-1.4": "corpora/omw-1.4",
    }
    for r in resources:
        try:
            nltk.data.find(paths[r])
        except LookupError:
            nltk.download(r)


download_nltk_resources()


# Page Configuration
st.set_page_config(
    page_title="WriteDetect",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models():
    models = {}

    try:
        # Loading the complete pipeline
        try:
            models["pipeline_svm"] = joblib.load("models/svm_pipeline.pkl")
            models["pipeline_svm_available"] = True
        except FileNotFoundError:
            models["pipeline_svm_available"] = False
        try:
            models["pipeline_dt"] = joblib.load("models/decision_tree_pipeline.pkl")
            models["pipeline_dt_available"] = True
        except FileNotFoundError:
            models["pipeline_dt_available"] = False
        try:
            models["pipeline_adaboost"] = joblib.load("models/adaboost_pipeline.pkl")
            models["pipeline_adaboost_available"] = True
        except FileNotFoundError:
            models["pipeline_adaboost_available"] = False

        try:
            with open("models/tokenizer.pkl", "rb") as f:
                models["shared_tokenizer"] = pickle.load(f)
                models["shared_tokenizer_available"] = True
        except Exception as e:
            st.warning(f"Tokenizer load failed: {e}")
            models["shared_tokenizer_available"] = False

        # Load Deep Learning models
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        try:
            embedding_matrix_w2v = np.load("models/embedding_matrix_w2v.npy")
            model_rnn = RNNTextClassifierV2(embedding_matrix_w2v)
            model_rnn.load_state_dict(torch.load("models/RNN_w2v.pt", map_location=device))
            model_rnn.to(device)
            model_rnn.eval()
            models["rnn_w2v"] = model_rnn
            models["rnn_w2v_available"] = True
        except:
            models["rnn_w2v_available"] = False

        try:
            embedding_matrix_glove = np.load("models/embedding_matrix_glove.npy")
            model_lstm = LSTMTextClassifier(embedding_matrix_glove)
            model_lstm.load_state_dict(torch.load("models/LSTM_glove.pt", map_location=device))
            model_lstm.to(device)
            model_lstm.eval()
            models["lstm_glove"] = model_lstm
            models["lstm_glove_available"] = True
        except:
            models["lstm_glove_available"] = False

        try:
            embedding_matrix_w2v = np.load("models/embedding_matrix_w2v.npy")
            model_cnn = CNNTextClassifier(embedding_matrix_w2v)
            model_cnn.load_state_dict(torch.load("models/CNN_w2v.pt", map_location=device))
            model_cnn.to(device)
            model_cnn.eval()
            models["gated_rnn_w2v"] = model_cnn
            models["gated_rnn_w2v_available"] = True
        except:
            models["gated_rnn_w2v_available"] = False

        # Check if at least one pipeline is loaded
        individual_ready = models["pipeline_adaboost_available"] or (
            models["pipeline_dt_available"] or models["pipeline_svm_available"]
        )

        if not (individual_ready):
            st.error("No complete model setup found!")
            return None

        return models

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


    
def make_prediction(text: str, model_choice: str, models: dict, max_len_pad=192):
    if models is None or model_choice is None:
        return None, None, None

    # === Classical ML Models ===
    if model_choice in ["pipeline_svm", "pipeline_dt", "pipeline_adaboost"]:
        pipeline = models.get(model_choice)

        try:
            cleaned = text_preprocessing(text)
            lemmatized = lemmatize_text(cleaned)

            vectorizer = pipeline.named_steps["vectorizer"]
            classifier = pipeline.named_steps["model"]

            X_vectorized = vectorizer.transform([lemmatized])
            y_pred_int = classifier.predict(X_vectorized)[0]

            # Probabilities
            if hasattr(classifier, "predict_proba"):
                y_proba_raw = classifier.predict_proba(X_vectorized)[0]
                proba_dict = dict(zip(classifier.classes_, y_proba_raw))
                p_human = proba_dict.get(0, 0.0)
                p_ai = proba_dict.get(1, 0.0)
                y_proba = [p_human, p_ai]
            else:
                y_proba = np.eye(2)[y_pred_int]

            class_names = ["Human Written", "AI Written"]
            return class_names[y_pred_int], y_proba, lemmatized

        except Exception as e:
            st.error(f"Error in classical model prediction: {e}")
            return None, None, None

    # === Deep Learning Models ===
    else:
        try:
            cleaned = text_preprocessing(text)
            lemmatized = lemmatize_text(cleaned)

            if models["shared_tokenizer"] is None:
                st.error("Tokenizer required for deep learning models.")
                return None, None, None

            # Tokenize and pad
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = models["shared_tokenizer"]
            seq = tokenizer.texts_to_sequences([lemmatized])
            padded = pad_sequences(seq, maxlen=max_len_pad, padding='post')
            tensor = torch.LongTensor(padded).to(device)

            model = models.get(model_choice)
            model.eval()

            with torch.no_grad():
                tensor = tensor.to(device)
                outputs = model(tensor).squeeze()
                probs = outputs.cpu().numpy()

            pred_label = (outputs > 0.5).int().cpu().numpy()
            y_proba = [1 - probs, probs] if probs.size == 1 else probs.tolist()

            class_names = ["Human Written", "AI Written"]
            return class_names[pred_label], y_proba, lemmatized

        except Exception as e:
            st.error(f"Error in deep learning model prediction: {e}")
            return None, None, None



def get_available_models(models):
    """Get list of available models for selection"""
    available = []

    if models is None:
        return available

    # Classical ML models
    if models.get("pipeline_svm_available"):
        available.append(("pipeline_svm", "📈 Support Vector Machine (Pipeline)"))
    if models.get("pipeline_dt_available"):
        available.append(("pipeline_dt", "📈 Decision Tree Classifier (Pipeline)"))
    if models.get("pipeline_adaboost_available"):
        available.append(("pipeline_adaboost", "📈 AdaBoost Classifier (Pipeline)"))

    # Deep Learning models
    if models.get("rnn_w2v_available"):
        available.append(("rnn_w2v", "🧠 RNN (Word2Vec)"))
    if models.get("lstm_glove_available"):
        available.append(("lstm_glove", "🧠 LSTM (GloVe)"))
    if models.get("gated_rnn_w2v_available"):
        available.append(("gated_rnn_w2v", "🧠 Gated RNN (Word2Vec)"))

    return available


# SIDEBAR NAVIGATION

st.sidebar.title("✍️ WriteDetect")
st.sidebar.markdown("Select a section below to get started:")

page = st.sidebar.selectbox(
    "Go to:",
    ["🏠 Dashboard", "🔍 Analyze Text", "📂 Upload Document", "📊 Compare Models", "ℹ️ About Models"],
)

# Load models
models = load_models()

# HOME PAGE
if page == "🏠 Dashboard":
    st.markdown(
        '<h1 class="main-header">✍️ WriteDetect — AI Origin Detector</h1>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
    **WriteDetect** uses a combination of classical machine learning, deep learning, and large language models to determine whether a piece of writing was produced by a human or generated by an AI system.
    Six trained models are available for analysis:
    - **Support Vector Machine** — linear kernel with TF-IDF features
    - **Decision Tree** — interpretable rule-based classifier
    - **AdaBoost** — ensemble of weak learners
    - **Bidirectional GRU** — recurrent model with Word2Vec embeddings
    - **LSTM** — long short-term memory network with GloVe embeddings
    - **TextCNN** — convolutional network over word embedding sequences

    On top of these, two Hugging Face LLMs add a second layer of insight:
    - **Qwen2.5-0.5B-Instruct** — gives an independent AI-judge verdict with reasoning
    - **SmolLM2-360M-Instruct** — turns the classifier's verdict into a plain-English explanation

    Paste text directly, upload a document, or run all models at once and compare their outputs.
    """
    )

    # Feature overview
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            """
        ### 🔍 Text Analysis
        - Paste or type any text
        - Choose from ML or deep learning models
        - Get an instant human/AI verdict
        - See confidence scores and probability bars
        """
        )

    with col2:
        st.markdown(
            """
        ### 📂 Document Upload
        - Supports `.pdf`, `.docx`, and `.txt`
        - Entire document treated as one sample
        - Confidence score and probability chart
        - LIME word-level explanations (ML models)
        """
        )

    with col3:
        st.markdown(
            """
        ### 📊 Model Comparison
        - Run all available models on the same input
        - See where models agree or diverge
        - Side-by-side probability charts
        - Download a full PDF comparison report
        """
        )

    with col4:
        st.markdown(
            """
        ### 🤖 LLM Insights
        - Independent AI-judge second opinion
        - Plain-English explanation of any verdict
        - Powered by Qwen2.5 and SmolLM2
        - Available on Analyze Text & Upload Document
        """
        )

    # Model pipeline status
    st.subheader("📋 Model Status")
    if models:
        st.success("✅ Models loaded successfully!")

        col1, col2, col3 = st.columns(3)

        with col1:
            if models.get("pipeline_dt_available"):
                st.info("**📈 Decision Tree Classifier**\n✅ Available")
            else:
                st.warning("**📈 Decision Tree Classifier**\n❌ Not Available")

        with col2:
            if models.get("pipeline_svm_available"):
                st.info("**🎯 Support Vector Machine**\n✅ Available")
            else:
                st.warning("**🎯 Support Vector Machine**\n❌ Not Available")

        with col3:
            if models.get("pipeline_adaboost_available"):
                st.info("**🔤 AdaBoost Classifier**\n✅ Available")
            else:
                st.warning("**🔤 AdaBoost Classifier**\n❌ Not Available")

        st.markdown("---")
        st.markdown("### 🧠 Deep Learning Models")

        col1, col2, col3 = st.columns(3)

        with col1:
            if models.get("gated_rnn_w2v_available"):
                st.info("✅ Gated RNN with Word2Vec")
            else:
                st.warning("❌ Gated RNN with Word2Vec")
        with col2:
            if models.get("lstm_glove_available"):
                st.info("✅ LSTM with GloVe")
            else:
                st.warning("❌ LSTM with GloVe")

        with col3:
            if models.get("rnn_w2v_available"):
                st.info("✅ RNN with Word2Vec")
            else:
                st.warning("❌ RNN with Word2Vec")
        if models.get("shared_tokenizer_available"):
            st.info("✅ DL Models Shared Tokenizer")
        else:
            st.warning("❌ DL Models Shared Tokenizer")

    else:
        st.error("❌ Models not loaded. Please check model files.")

# SINGLE PREDICTION PAGE

elif page == "🔍 Analyze Text":
    st.header("🔍 Analyze Text")
    st.markdown("Paste any text below, pick a model, and find out whether it reads as human or AI-generated.")

    if models:
        available_models = get_available_models(models)

        if available_models:
            # Model selection
            model_choice = st.selectbox(
                "Choose a model:",
                options=[model[0] for model in available_models],
                format_func=lambda x: next(
                    model[1] for model in available_models if model[0] == x
                ),
            )

            # Text input
            user_input = st.text_area(
                "Text to classify:",
                placeholder="Paste an essay, paragraph, or any writing sample here...",
                height=150,
            )

            # Character count
            if user_input:
                st.caption(
                    f"Character count: {len(user_input)} | Word count: {len(user_input.split())}"
                )

            with st.expander("📝 Try a sample text"):
                examples = [
                    "The french revolution fundamentally reshaped the political landscape of europe, replacing monarchy with ideals of liberty and citizen sovereignty.",
                    "Climate change is an urgent global challenge driven by greenhouse gas emissions from human activities like burning fossil fuels and deforestation.",
                    "I stayed up way too late finishing my essay and honestly I'm not even sure it makes sense anymore — my brain is completely fried.",
                    "The experiment demonstrated a statistically significant correlation between sleep deprivation and reduced cognitive performance across all age groups.",
                    "Honestly the lecture today was so dry I had to reread my notes three times just to figure out what the main point was.",
                ]

                for i, example in enumerate(examples):
                    if st.button(
                        f'Use Example {i+1}: "{example[:50]}..."', key=f"example_{i}"
                    ):
                        st.session_state.user_input = example
                        st.rerun()

            # Use session state for user input
            if "user_input" in st.session_state:
                user_input = st.session_state.user_input

            # Prediction button
            if st.button("🔍 Run Detection", type="primary"):
                if user_input.strip():
                    with st.spinner("Classifying text..."):
                        prediction, probabilities, lemmatized = make_prediction(
                            user_input, model_choice, models
                        )
                        if prediction and probabilities is not None:
                            st.session_state.analyze_text = user_input
                            st.session_state.analyze_prediction = prediction
                            st.session_state.analyze_probabilities = [float(p) for p in probabilities]
                            st.session_state.analyze_model_choice = model_choice
                            st.session_state.analyze_llm_judge = None
                            st.session_state.analyze_llm_explanation = None
                        else:
                            st.session_state.analyze_prediction = None
                            st.error("Detection failed. Please try again.")
                else:
                    st.warning("Please paste some text before running detection.")

            # Render results from session_state so LLM buttons below don't wipe them on rerun
            if st.session_state.get("analyze_prediction"):
                prediction = st.session_state.analyze_prediction
                probabilities = st.session_state.analyze_probabilities

                col1, col2 = st.columns([3, 1])
                with col1:
                    if prediction == "Human Written":
                        st.success(f"🎯 Prediction: **{prediction}**")
                    else:
                        st.error(f"🎯 Prediction: **{prediction} **")

                with col2:
                    confidence = max(probabilities)
                    st.metric("Confidence", f"{confidence:.1%}")

                # Create probability chart
                st.subheader("📊 Prediction Probabilities")

                # Detailed probabilities
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Human", f"{probabilities[0]:.1%}")
                with col2:
                    st.metric("AI Generated", f"{probabilities[1]:.1%}")

                # Bar chart
                class_names = ["Human Written", "AI Written"]
                prob_df = pd.DataFrame(
                    {"Origin": class_names, "Probability": probabilities}
                )
                st.bar_chart(prob_df.set_index("Origin"), height=300)

                # LLM-powered insights
                st.markdown("---")
                st.subheader("🤖 LLM Insights")
                st.caption("Two Hugging Face language models add a second opinion and a plain-English explanation on top of the classifier above.")

                judge_version_options = ["Base (zero-shot)"]
                if finetuned_judge_available():
                    judge_version_options.append("Fine-tuned (LoRA on WriteDetect data)")
                judge_version = st.selectbox(
                    "AI Judge model version:", judge_version_options, key="analyze_judge_version"
                )
                if len(judge_version_options) == 1:
                    st.caption("Train a LoRA adapter (see finetune_ai_judge.py) and place it in models/qwen_judge_lora/ to unlock the fine-tuned option here.")

                llm_col1, llm_col2 = st.columns(2)
                with llm_col1:
                    if st.button("🧑‍⚖️ Get AI Judge Second Opinion"):
                        use_finetuned = judge_version.startswith("Fine-tuned")
                        with st.spinner(f"Asking {'fine-tuned ' if use_finetuned else ''}Qwen2.5-0.5B-Instruct..."):
                            st.session_state.analyze_llm_judge = run_ai_judge(
                                st.session_state.analyze_text, use_finetuned=use_finetuned
                            )
                with llm_col2:
                    if st.button("💬 Get Plain-English Explanation"):
                        with st.spinner("Asking SmolLM2-360M-Instruct..."):
                            st.session_state.analyze_llm_explanation = run_plain_explanation(
                                prediction, max(probabilities)
                            )

                if st.session_state.get("analyze_llm_judge"):
                    judge = st.session_state.analyze_llm_judge
                    st.info(f"**AI Judge verdict:** {judge['verdict']}\n\n{judge['reasoning']}")

                if st.session_state.get("analyze_llm_explanation"):
                    st.success(f"**In plain English:** {st.session_state.analyze_llm_explanation}")

        else:
            st.error("No models available for prediction.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# DOCUMENT PROCESSING PAGE

elif page == "📂 Upload Document":
    st.header("📂 Upload a Document")
    st.markdown(
        "Upload a `.txt`, `.pdf`, or `.docx` file — the full text will be extracted and classified as human or AI-written."
    )

    if models:
        available_models = get_available_models(models)

        if available_models:
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=["txt", "pdf", "docx"],
                help="Supported formats: .txt, .pdf, .docx — the entire document is treated as one text sample.",
            )

            # Reset LIME state only if a new file is uploaded
            if uploaded_file:
                if (
                    "last_uploaded_file" not in st.session_state
                    or st.session_state.last_uploaded_file != uploaded_file.name
                ):
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.session_state.show_lime = False

                model_choice = st.selectbox(
                    "Choose model for document processing:",
                    options=[model[0] for model in available_models],
                    format_func=lambda x: next(
                        model[1] for model in available_models if model[0] == x
                    ),
                )

                if "lime_text" not in st.session_state:
                    st.session_state.lime_text = None
                if "lime_model_key" not in st.session_state:
                    st.session_state.lime_model_key = None
                if "lime_prediction" not in st.session_state:
                    st.session_state.lime_prediction = None
                if "lime_probabilities" not in st.session_state:
                    st.session_state.lime_probabilities = None

                if st.button("📂 Classify Document"):
                    try:
                        if uploaded_file.type == "text/plain":
                            text = read_txt(uploaded_file)
                        elif uploaded_file.type == "application/pdf":
                            text = read_pdf(uploaded_file)
                        elif (
                            uploaded_file.type
                            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        ):
                            text = read_docx(uploaded_file)
                        else:
                            st.error("Unsupported file type.")
                            text = ""

                        if not text.strip():
                            st.error("No text found in file.")
                        else:
                            st.info(
                                "Extracting and classifying document text..."
                            )

                            prediction, probabilities, lemmatized = make_prediction(
                                text, model_choice, models
                            )
                            if prediction and probabilities is not None:
                                probabilities = [float(p) for p in probabilities]
                                # Save everything in session state
                                st.session_state.lime_text = lemmatized
                                st.session_state.lime_model_key = model_choice
                                st.session_state.lime_prediction = prediction
                                st.session_state.lime_probabilities = probabilities
                                st.session_state.show_lime = False
                                st.session_state.doc_original_text = text
                                st.session_state.doc_llm_judge = None
                                st.session_state.doc_llm_explanation = None

                            else:
                                st.error("Prediction failed. Please try again.")
                    except Exception as e:
                        st.error(f"Error processing file: {e}")

                # Show results if stored
                if (
                    st.session_state.lime_prediction
                    and st.session_state.lime_probabilities
                ):
                    st.success("✅ Prediction Completed")
                    st.write(f"**Prediction:** {st.session_state.lime_prediction}")
                    st.write(
                        f"**Confidence:** {max(st.session_state.lime_probabilities):.1%}"
                    )

                    st.subheader("📊 Probability Distribution")
                    prob_df = pd.DataFrame(
                        {
                            "Class": ["Human Written", "AI Written"],
                            "Probability": st.session_state.lime_probabilities,
                        }
                    )
                    st.bar_chart(prob_df.set_index("Class"))

                    if st.session_state.lime_model_key in ["pipeline_svm", "pipeline_dt", "pipeline_adaboost"]:
                        if st.button("🧠 Generate LIME Explanation"):
                            st.session_state.show_lime = True
                    else:
                        st.info("🧠 LIME explanations are currently not available for deep learning models.")

                # LIME explanation
                if (
                    st.session_state.get("show_lime", False)
                    and st.session_state.lime_text
                ):
                    try:
                        pipeline = models.get(st.session_state.lime_model_key)
                        if pipeline:
                            with st.spinner("Generating explanation..."):
                                explanation = explain_prediction_with_lime(
                                    pipeline, st.session_state.lime_text
                                )
                                st.subheader(
                                    "🧠 LIME Explanation for cleaned text(Top 20 Words)"
                                )
                                components.html(
                                    explanation.as_html(), height=400, scrolling=True
                                )
                        else:
                            st.error("Selected model pipeline not found.")
                    except Exception as e:
                        st.error(f"Error generating LIME explanation: {e}")

                # LLM-powered insights
                if st.session_state.get("doc_original_text"):
                    st.markdown("---")
                    st.subheader("🤖 LLM Insights")
                    st.caption("Two Hugging Face language models add a second opinion and a plain-English explanation on top of the classifier above.")

                    doc_judge_version_options = ["Base (zero-shot)"]
                    if finetuned_judge_available():
                        doc_judge_version_options.append("Fine-tuned (LoRA on WriteDetect data)")
                    doc_judge_version = st.selectbox(
                        "AI Judge model version:", doc_judge_version_options, key="doc_judge_version"
                    )
                    if len(doc_judge_version_options) == 1:
                        st.caption("Train a LoRA adapter (see finetune_ai_judge.py) and place it in models/qwen_judge_lora/ to unlock the fine-tuned option here.")

                    llm_col1, llm_col2 = st.columns(2)
                    with llm_col1:
                        if st.button("🧑‍⚖️ Get AI Judge Second Opinion", key="doc_judge_btn"):
                            doc_use_finetuned = doc_judge_version.startswith("Fine-tuned")
                            with st.spinner(f"Asking {'fine-tuned ' if doc_use_finetuned else ''}Qwen2.5-0.5B-Instruct..."):
                                st.session_state.doc_llm_judge = run_ai_judge(
                                    st.session_state.doc_original_text, use_finetuned=doc_use_finetuned
                                )
                    with llm_col2:
                        if st.button("💬 Get Plain-English Explanation", key="doc_explain_btn"):
                            with st.spinner("Asking SmolLM2-360M-Instruct..."):
                                st.session_state.doc_llm_explanation = run_plain_explanation(
                                    st.session_state.lime_prediction,
                                    max(st.session_state.lime_probabilities),
                                )

                    if st.session_state.get("doc_llm_judge"):
                        judge = st.session_state.doc_llm_judge
                        st.info(f"**AI Judge verdict:** {judge['verdict']}\n\n{judge['reasoning']}")

                    if st.session_state.get("doc_llm_explanation"):
                        st.success(f"**In plain English:** {st.session_state.doc_llm_explanation}")
            else:
                st.info("Upload a file above to begin.")
        else:
            st.error("No models are currently loaded.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# MODEL COMPARISON PAGE

elif page == "📊 Compare Models":
    st.header("📊 Compare Models")
    st.markdown(
        "Run the same text through every available model and see how their predictions stack up against each other."
    )

    if models:
        available_models = get_available_models(models)

        if len(available_models) >= 2:
            input_mode = st.radio(
                "Input method:", ["✍️ Type or Paste Text", "📂 Upload a File"]
            )

            comparison_text = ""
            uploaded_file = None

            if input_mode == "✍️ Type or Paste Text":
                comparison_text = st.text_area(
                    "Text to classify across all models:", height=120
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload a `.txt`, `.pdf`, or `.docx` file"
                )
                if uploaded_file:
                    if uploaded_file.type == "text/plain":
                        comparison_text = read_txt(uploaded_file)
                    elif uploaded_file.type == "application/pdf":
                        comparison_text = read_pdf(uploaded_file)
                    elif (
                        uploaded_file.type
                        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    ):
                        comparison_text = read_docx(uploaded_file)
                    else:
                        st.error("Unsupported file type.")

            # Process and store results on button click
            if st.button("📊 Run All Models") and comparison_text.strip():
                st.subheader("Results Across All Models")
                comparison_results = []
                word_count = len(comparison_text.split())
                filename = uploaded_file.name if uploaded_file else None

                for model_key, model_name in available_models:
                    prediction, probabilities, lemmatized = make_prediction(
                        comparison_text, model_key, models
                    )

                    if prediction and probabilities is not None:

                        probabilities = [float(p) for p in probabilities]
                        comparison_results.append(
                            {
                                "Model": model_name,
                                "Prediction": prediction,
                                "Confidence": f"{max(probabilities):.1%}",
                                "Human Written %": f"{probabilities[0]:.1%}",
                                "AI Written %": f"{probabilities[1]:.1%}",
                                "Raw_Probs": probabilities,
                            }
                        )

                # Store results and meta
                st.session_state.comparison_results = comparison_results
                st.session_state.comparison_word_count = word_count
                st.session_state.comparison_filename = filename

            # Render from session_state (even after rerun)
            comparison_results = st.session_state.get("comparison_results", [])

            if comparison_results:
                df = pd.DataFrame(comparison_results)
                st.table(
                    df[
                        [
                            "Model",
                            "Prediction",
                            "Confidence",
                            "Human Written %",
                            "AI Written %",
                        ]
                    ]
                )

                predictions = [r["Prediction"] for r in comparison_results]
                if len(set(predictions)) == 1:
                    st.success(f"✅ All models agree: **{predictions[0]}**")
                else:
                    st.warning("⚠️ Models disagree on prediction:")
                    for res in comparison_results:
                        st.write(f"- **{res['Model']}**: {res['Prediction']}")

                st.subheader("📊 Side-by-Side Probability Charts")
                cols = st.columns(len(comparison_results))
                for i, res in enumerate(comparison_results):
                    with cols[i]:
                        st.markdown(f"**{res['Model']}**")
                        chart_data = pd.DataFrame(
                            {
                                "Class": ["Human Written", "AI Written"],
                                "Probability": res["Raw_Probs"],
                            }
                        )
                        st.bar_chart(chart_data.set_index("Class"))

                if st.button("📥 Download Comparison Report (PDF)"):
                    # 1) build LIME explanations dict
                    lime_expl = {}
                    for res in comparison_results:
                        model_key = next(
                            (k for k, v in available_models if v == res["Model"]), None
                        )
                        if model_key and models.get(model_key):
                            try:
                                exp = explain_prediction_with_lime(
                                    models[model_key], comparison_text
                                )
                                lime_expl[res["Model"]] = exp.as_list()
                            except Exception:
                                pass  # skip if LIME fails

                    # 2) generate & offer download
                    pdf_path = generate_comparison_report_pdf_full(
                        comparison_results=comparison_results,
                        input_method=input_mode,
                        word_count=st.session_state.comparison_word_count,
                        filename=st.session_state.comparison_filename,
                        original_text=comparison_text,
                        lime_explanations=lime_expl,
                    )
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()
                        b64_pdf = base64.b64encode(pdf_bytes).decode()

                    st.success("✅ Report generated!")
                    st.download_button(
                        label="📄 Click to Download PDF Report",
                        data=pdf_bytes,
                        file_name="comparison_report.pdf",
                        mime="application/pdf"
                    )

        elif len(available_models) == 1:
            st.info(
                "Only one model is loaded. Head to Analyze Text for a single-model prediction."
            )
        else:
            st.error("No models available.")
    else:
        st.warning("Models not loaded. Please check the model files.")

# ============================================================================
# MODEL INFO PAGE
# ============================================================================

elif page == "ℹ️ About Models":
    st.header("ℹ️ About the Models")

    if models:
        st.success("✅ All models loaded and ready.")

        st.subheader("Model Overview")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### Classical ML Models
            - **Support Vector Machine (SVM)**
            - **Decision Tree**
            - **AdaBoost**

            **Input features:** TF-IDF bag-of-words (unigrams + bigrams, 50k vocab)

            **Training notes:**
            - Stratified 80/10/10 train/val/test split
            - Hyperparameters tuned with GridSearchCV
            - LIME word-level explanations available
            """)

        with col2:
            st.markdown("""
            ### Deep Learning Models
            - **LSTM** with GloVe-100d embeddings
            - **Bidirectional GRU** with Word2Vec-100d embeddings
            - **TextCNN** with Word2Vec-100d embeddings

            **Input features:** tokenized + padded word sequences

            **Training notes:**
            - Sequence length set to 95th percentile of training data
            - GloVe loaded from gensim; Word2Vec trained from scratch
            - Binary cross-entropy loss, sigmoid output
            - Implemented in PyTorch
            """)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Text Preprocessing")
            st.markdown("""
            **Applied to all models:**
            - Lowercase conversion
            - Punctuation and number removal
            - English stopword filtering
            - Token length filter (alpha-only, length > 1)

            **Deep learning only:**
            - Keras Tokenizer → integer sequences
            - Zero-padding to fixed max length

            **ML models (at inference):**
            - POS tagging via NLTK
            - WordNet lemmatization per tag
            """)
        with col2:
            st.subheader("Training Dataset")
            st.markdown("""
            - **Task:** Binary classification — Human vs AI authorship
            - **Total samples:** 8,176
            - **Class balance:**
                - Human Written (label 0): 4,088
                - AI Generated (label 1): 4,088
            - **AI sources:** GPT, Claude, Mistral, LLaMA, Falcon
            - **Split strategy:** Stratified 80/10/10 for ML; 5-fold CV for DL
            """)

        st.markdown("---")
        st.subheader("🤖 Large Language Models")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### AI Judge
            **Model:** `Qwen/Qwen2.5-0.5B-Instruct`

            Given the raw text, the LLM is independently prompted to decide
            Human-Written vs AI-Generated and justify its reasoning in 1-2
            sentences — a second opinion alongside the trained classifiers,
            not derived from their output.

            A LoRA-fine-tuned version of this model (trained on the WriteDetect
            training labels — see `finetune_ai_judge.py`) can be selected instead
            of the base model on the Analyze Text and Upload Document pages,
            once its adapter is placed in `models/qwen_judge_lora/`.
            """)
            if finetuned_judge_available():
                st.success("✅ Fine-tuned LoRA adapter detected — selectable in the AI Judge dropdown.")
            else:
                st.info("ℹ️ No fine-tuned adapter found yet — only the base model is available.")

        with col2:
            st.markdown("""
            ### Plain-English Explainer
            **Model:** `HuggingFaceTB/SmolLM2-360M-Instruct`

            Takes the classifier's verdict, confidence score, and (when
            available) the top LIME keywords, and rewrites them into a
            short, jargon-free explanation for a non-technical reader.
            """)

        st.caption(
            "Both models are loaded via Hugging Face `transformers` pipelines and are small "
            "enough to run on the free CPU tier of Hugging Face Spaces. Available on the "
            "Analyze Text and Upload Document pages."
        )

        st.subheader("📁 Model Files Status")
        file_status = []

        files_to_check = [
            ("svm_pipeline.pkl", "SVM Pipeline", models.get("pipeline_svm_available", False)),
            ("decision_tree_pipeline.pkl", "Decision Tree Pipeline", models.get("pipeline_dt_available", False)),
            ("adaboost_pipeline.pkl", "AdaBoost Pipeline", models.get("pipeline_adaboost_available", False)),
            ("RNN_w2v.pt", "RNN w/ Word2Vec", models.get("rnn_w2v_available", False)),
            ("LSTM_glove.pt", "LSTM w/ GloVe", models.get("lstm_glove_available", False)),
            ("CNN_w2v.pt", "Gated RNN (CNN) w/ Word2Vec", models.get("gated_rnn_w2v_available", False)),
            ("tokenizer.pkl", "Keras Tokenizer", os.path.exists("models/tokenizer.pkl")),
            ("embedding_matrix_w2v.npy", "Embedding Matrix (Word2Vec)", os.path.exists("models/embedding_matrix_w2v.npy")),
            ("embedding_matrix_glove.npy", "Embedding Matrix (GloVe)", os.path.exists("models/embedding_matrix_glove.npy")),
        ]

        for filename, description, status in files_to_check:
            file_status.append({
                "File": filename,
                "Description": description,
                "Status": "✅ Loaded" if status else "❌ Not Found"
            })

        st.table(pd.DataFrame(file_status))

        st.subheader("📉 ROC Curves")

        cols = st.columns(2)
        with cols[0]:
            st.image("plots/ML Modles ROC.png", caption="Classical ML Models")
        with cols[1]:
            st.image("plots/RNN -w2v.png", caption="RNN")
        with cols[0]:
            st.image("plots/LSTM Roc.png", caption="LSTM")
        with cols[1]:
            st.image("plots/CNN- W2v.png", caption="Gated RNN (CNN)")
        
        st.subheader("📈 Confusion Matrices")

        cols = st.columns(2)

        with cols[0]:
            st.image("plots/SVM cf.png", caption="SVM", width=400)
            st.markdown("<br>", unsafe_allow_html=True)  # Adds vertical spacing
            st.image("plots/DT cf.png", caption="Decision Tree", width=400)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image("plots/AdaBoost CF.png", caption="AdaBoost", width=400)

        with cols[1]:
            st.image("plots/RNN- w2v conf.png", caption="RNN", width=400)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image("plots/LSTM Cf.png", caption="LSTM", width=400)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image("plots/CNN- w2v conf.png", caption="Gated RNN (CNN)", width=400)

        st.subheader("WordCloud")

        cols = st.columns(2)

        with cols[0]:
            st.image("plots/WordCloud.png", caption="WordCloud")


    else:
        st.warning("Models not loaded. Please check model files in the 'models/' directory.")

# FOOTER

st.sidebar.markdown("---")
st.sidebar.markdown("### About WriteDetect")
st.sidebar.info(
    """
**WriteDetect**
AI Origin Detection System

**Models:**
- SVM (TF-IDF)
- Decision Tree (TF-IDF)
- AdaBoost (TF-IDF)
- Bidirectional GRU + Word2Vec
- LSTM + GloVe
- TextCNN + Word2Vec

**LLMs:**
- Qwen2.5-0.5B-Instruct (AI judge)
- SmolLM2-360M-Instruct (plain-English explainer)

**Capabilities:**
- Raw text classification
- Document upload (PDF, DOCX, TXT)
- Multi-model comparison
- LIME explanations
- LLM second opinion & explanations
- PDF report export

**Stack:** PyTorch · scikit-learn · Transformers · Streamlit
"""
)
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666666;'>
    WriteDetect — AI Origin Detection &nbsp;|&nbsp; Built with Streamlit<br>
    <small>Combines classical ML and deep learning to identify AI-generated text</small>
</div>
""",
    unsafe_allow_html=True,
)
