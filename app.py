import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go
import time
# --- 1. Configuration & Academic Context ---
st.set_page_config(
    layout="wide", 
    page_title="Algorithmic Provenance: An Inquiry",
    initial_sidebar_state="expanded"
)
# --- NLTK Setup (Robust Handling for Cloud Deployment) ---
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
# --- 2. The Thesis & Author (Sidebar) ---
with st.sidebar:
    st.markdown("### Project Lead")
    st.markdown("**Ali Pasha Abdollahi**")
    st.markdown("*Insight Galaxy Ltd.*")
    st.markdown("*Aula Fellowship for AI Science, Tech and Policy*")
    
    # Social Links
    st.markdown(
        """
        <div style="display: flex; gap: 10px;">
            <a href="https://philpeople.org/profiles/ali-pasha-abdollahi" target="_blank">PhilPeople Profile</a> | 
            <a href="https://www.linkedin.com/in/alipashaabdollahi/" target="_blank">LinkedIn</a>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    st.divider()
    
    st.header("Exhibit A: The Mechanism")
    
    # --- The Explicit Refutation ---
    st.error(
        "**Core Thesis:** This is a refutation of the assumption that language models produce coherent sentences because they 'mimic biological neurons.'"
    )
    
    st.markdown("### I. The Algorithmic Distinction")
    st.info(
        "**The Argument:** The dominant discourse frequently anthropomorphizes AI using biological metaphors ('neural networks', 'learning').\n\n"
        "**The Demonstration:** This system generates coherent sentence completions using a **Support Vector Machine (SVM)**—a purely algebraic method from the 1990s. "
        "It possesses no neurons and mimics no biological processes. "
        "It demonstrates that the appearance of linguistic competence can be achieved through statistical geometry alone."
    )
    
    st.markdown("### II. Data Provenance")
    st.warning(
        "**The Argument:** It is often claimed that models 'understand' concepts and generate novel text (transformative use).\n\n"
        "**The Demonstration:** The **Provenance Audit** panel reveals that the model's outputs are frequently verbatim retrievals from the source corpus. "
        "The system does not 'imagine' the next word; it locates the statistical precedent in its training data."
    )
# --- 3. The Controlled Corpus & Model ---
# --- 3. The Dual-Corpus Architecture ---
@st.cache_data
def get_sentence_model():
    """Builds the Linguistic/Philosophical Model"""
    # 1. Base Corpus (Simple Sentences + Philosophical Context)
    corpus = [
        # Original Dog/Cat Universe
        "The happy dog sat on the rug.", "The lazy cat slept on the sofa.", "The playful dog chased the red toy.",
        "The sleepy cat watched the small bird.", "The brown dog ate the big bone.", "The white cat hid under the bed.",
        "The small dog played with the furry cat.", "The big cat sat near the food.", "The furry dog wanted a toy.",
        "The playful cat dropped the small mouse.", "The dog ran on the green mat.", "The cat slept under the warm sun.",
        "The happy dog wanted the bone.", "The lazy cat saw the sleepy dog.", "The playful dog sat on the sofa.",
        "The sleepy cat ate the white food.", "The brown dog chased the furry cat.", "The white cat played with a toy.",
        
        # ... (Some Location/Object Variations) ...
        "The happy dog sat at the table.", "The lazy cat slept under the table.", "The brown dog hid under the table.",
        "The small mouse ran under the table.", "The big bird sat on the table.", "The white cat played at the table.",
        
        # Philosophical/Technical Refutation Context
        "The machine processes the raw data.", "The algorithm minimizes the error rate.", 
        "The system retrieves the stored pattern.", "The logic follows the strict rule.",
        "The silicon chip processes the bit.", "The vector maps to the point.",
        "The graph shows the linear line.", "The model simulates the human speech.",
        "The output mimics the user input.", "The audit reveals the source text.",
        "The mathematics defines the boundary line.", "The geometry explains the output word.",
        
        # Simple Logic / Folk Wisdom
        "If it rains the ground gets wet.", "If the sun shines the grass grows.",
        "The red light means stop now.", "The green light means go now.",
        "The early bird catches the worm.", "The quick brown fox jumps over.",
    ]
    
    return _train_pipeline(corpus, "Linguistic Model")
@st.cache_data
def get_arithmetic_model():
    """Builds the Rigid Arithmetic Model"""
    # 2. Arithmetic Expansion (Programmatic)
    num_map = {
        0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
        5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 
        10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen", 
        14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen", 
        18: "eighteen", 19: "nineteen"
    }
    
    arithmetic_corpus = []
    
    # Generate all single digit additions
    for a in range(10):  # 0-9
        for b in range(10): # 0-9
            res = a + b
            if res in num_map:
                # e.g. "two plus two equals four"
                sentence = f"{num_map[a]} plus {num_map[b]} equals {num_map[res]}."
                arithmetic_corpus.append(sentence)
                
                # Symbolic version: "2 + 2 = 4."
                sentence_sym = f"{a} + {b} = {res}."
                arithmetic_corpus.append(sentence_sym)
    # Generate simple subtractions (where result >= 0)
    for a in range(10):
        for b in range(a + 1): # b <= a
            res = a - b
            # "five minus two equals three"
            if res in num_map:
                sentence = f"{num_map[a]} minus {num_map[b]} equals {num_map[res]}."
                arithmetic_corpus.append(sentence)
                sentence_sym = f"{a} - {b} = {res}."
                arithmetic_corpus.append(sentence_sym)
                
    return _train_pipeline(arithmetic_corpus, "Arithmetic Model")
def _train_pipeline(corpus, model_name):
    """Shared training logic for any corpus."""
    sequences_list = []
    targets = []
    sequence_length = 4
    
    for sentence in corpus:
        # Tokenize
        sent_tokens = [word.lower() for word in word_tokenize(sentence, preserve_line=True) if word.isalpha() or word in ['+', '-', '='] or word.isdigit()]
        
        if len(sent_tokens) <= sequence_length:
            continue
            
        # N-Grams
        for i in range(len(sent_tokens) - sequence_length):
            seq = sent_tokens[i:i + sequence_length]
            target = sent_tokens[i + sequence_length]
            sequences_list.append(" ".join(seq))
            targets.append(target)
            
    if not sequences_list:
        return None
        
    # Vectorization (N-Grams enabled for structure)
    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    X = vectorizer.fit_transform(sequences_list)
    y = np.array(targets)
    
    # Train SVM
    # C=1000 for "Hard Margin" / Memorization behavior
    # class_weight='balanced' to handle frequency skew
    svm_model = SVC(kernel='linear', C=1000, class_weight='balanced')
    svm_model.fit(X, y)
    
    return {
        "model": svm_model,
        "vectorizer": vectorizer,
        "sequence_length": sequence_length,
        "X": X,
        "y": y,
        "sequences": np.array(sequences_list),
        "corpus": corpus,
        "name": model_name
    }
# --- 4. Logic: Routing & Inference ---
def detect_intent(text):
    """Heuristic router to decide between Math and Sentences."""
    text_lower = text.lower()
    math_keywords = ['plus', 'minus', 'equals', 'sum', '+', '-', '=', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    # If any math symbol or keyword is present, route to Arithmetic
    score = sum(1 for k in math_keywords if k in text_lower)
    if score > 0:
        return "arithmetic"
    return "sentence"
def generate_and_audit(input_text, system_dict):
    if not system_dict: return None, None
    
    model = system_dict["model"]
    vectorizer = system_dict["vectorizer"]
    sequence_length = system_dict["sequence_length"]
    sequences_list = system_dict["sequences"]
    corpus = system_dict["corpus"]
    
    tokenized_input = word_tokenize(input_text.lower())
    # Robust check for minimum length
    if len(tokenized_input) < sequence_length:
        return None, None
    
    last_sequence = " ".join(tokenized_input[-sequence_length:])
    
    # Try-Catch for vocabulary issues (unseen words)
    try:
        input_vector = vectorizer.transform([last_sequence])
        # Check if vector is all zeros (completely unknown n-grams)
        if input_vector.sum() == 0:
            return None, ["(Input contains no known n-grams from training data)"]
            
        prediction = model.predict(input_vector)[0]
    except Exception as e:
        return None, [str(e)]
    
    # Provenance Audit
    search_phrase = f"{last_sequence} {prediction}"
    evidence = []
    
    for sentence in corpus:
        # Simple substring search for provenance
        if search_phrase in sentence.lower():
            evidence.append(sentence)
            if len(evidence) >= 3: break 
            
    return prediction, evidence
    
# --- 5. Visualization Logic ---
def plot_mathematical_boundary(system_dict, current_prediction=None):
    if not system_dict: return None
    
    X = system_dict["X"]
    y = system_dict["y"]
    sequences = system_dict["sequences"]
    model_name = system_dict["name"]
    
    # Adaptive Class Selection
    all_classes = sorted(list(np.unique(y)))
    if len(all_classes) < 2: return None
    
    # Default: Pick the prediction and its nearest neighbor or static fallback
    if current_prediction and current_prediction in all_classes:
        class1 = current_prediction
        # Pick a contrasting class (simple heuristic: next in list, or random)
        # Ideally, we'd pick the class with the 2nd highest decision function value,
        # but SVC(probability=False) doesn't give us that easily without more compute.
        # We will just pick a neighbor in the sorted list to ensure consistency.
        idx = all_classes.index(class1)
        class2 = all_classes[(idx + 1) % len(all_classes)]
    else:
        class1 = all_classes[0]
        class2 = all_classes[1]
    # Filter data
    class1_indices = np.where(y == class1)[0]
    class2_indices = np.where(y == class2)[0]
    indices = np.concatenate([class1_indices, class2_indices])
    
    if len(class1_indices) < 1 or len(class2_indices) < 1: return None
    
    X_filtered, y_filtered = X[indices].toarray(), y[indices]
    sequences_filtered = sequences[indices]
    
    # Dimensionality Reduction
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_filtered)
    
    # Fit a 2D SVM for Visualization purposes ONLY (Proxy Model)
    # matching the kernel of the main high-dim model
    svm_viz = SVC(kernel='linear', C=1.0) 
    svm_viz.fit(X_2d, y_filtered)
    # Grid for Decision Boundary
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_viz.predict(np.c_[xx.ravel(), yy.ravel()])
    
    # Convert string labels to 0/1 for contour plot
    # Z contains strings like 'four', 'five'
    Z_num = np.array([0 if l == class1 else 1 for l in Z]).reshape(xx.shape)
    fig = go.Figure()
    
    # Decision Boundary
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z_num, 
        colorscale=[[0, 'rgba(255, 0, 0, 0.2)'], [1, 'rgba(0, 0, 255, 0.2)']], 
        opacity=0.4, showscale=False, hoverinfo='skip', name="Decision Boundary"
    ))
    
    colors = {class1: '#D62728', class2: '#1F77B4'}
    symbols = {class1: 'circle', class2: 'diamond'}
    
    for cls in [class1, class2]:
        mask = (y_filtered == cls)
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1], 
            mode='markers', 
            name=f"Outcome: '{cls}'",
            marker=dict(color=colors[cls], size=12, symbol=symbols[cls], line=dict(width=1, color='black')),
            customdata=sequences_filtered[mask],
            hovertemplate='<b>Sequence:</b> "%{customdata}"<br><b>Next:</b> '+cls+'<br><extra></extra>'
        ))
        
    fig.update_layout(
        title=f"{model_name} Decision Space<br>'{class1}' vs. '{class2}'",
        xaxis_title="Principal Component 1", 
        yaxis_title="Principal Component 2",
        template="plotly_white", height=450,
        margin=dict(t=50, b=20, l=20, r=20)
    )
    return fig
# --- 6. Event Callbacks ---
def on_generate_click():
    current_input = st.session_state.user_text
    
    # 1. Detect Intent -> Switch Model
    intent = detect_intent(current_input)
    if intent == "arithmetic":
        system_dict = get_arithmetic_model()
    else:
        system_dict = get_sentence_model()
        
    st.session_state.active_model_name = system_dict["name"]
        
    # 2. Predict
    pred, evidence = generate_and_audit(current_input, system_dict)
    
    if pred:
        st.session_state.user_text = f"{current_input} {pred}"
        st.session_state.generated_word = pred
        st.session_state.provenance_data = evidence
        st.session_state.last_error = None
        # Store for visualization
        st.session_state.viz_system = system_dict
    else:
        if evidence and evidence[0].startswith("(Input"):
            st.session_state.last_error = evidence[0]
        else:
            st.session_state.last_error = f"Input too short or unknown. Need {system_dict['sequence_length']} words."
# --- 7. Main Interface Execution ---
# Initialize Session State
if 'user_text' not in st.session_state:
    st.session_state.user_text = "The happy dog sat on the"
if 'provenance_data' not in st.session_state:
    st.session_state.provenance_data = None
if 'generated_word' not in st.session_state:
    st.session_state.generated_word = None
if 'last_error' not in st.session_state:
    st.session_state.last_error = None
if 'active_model_name' not in st.session_state:
    st.session_state.active_model_name = "Linguistic Model"
if 'viz_system' not in st.session_state:
    # Default to sentence model initially
    st.session_state.viz_system = get_sentence_model()
# Ensure models are loaded
# We load them to have them cache-ready, but use lazy routing in callback
# (Streamlit execution flow requires we have them reachable)
get_sentence_model()
get_arithmetic_model()
# --- DEFINE COLUMNS ---
col_left, col_right = st.columns([1, 1])
# --- LEFT COLUMN: Input & Simulation ---
with col_left:
    st.subheader("1. The Simulation")
    st.caption(f"Active System: **{st.session_state.active_model_name}** (Auto-Detected)")
    
    st.markdown("Input a phrase. The system will detect if you are doing **Math** or **Language** and strictly retrieve the next token from the corresponding training data.")
    
    st.text_input("Input Sequence:", key="user_text")
    st.button("Generate Next Token", type="primary", on_click=on_generate_click)
    
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
# --- RIGHT COLUMN: Provenance & Visualization ---
with col_right:
    st.subheader("2. Provenance Audit")
    
    if st.session_state.provenance_data:
        st.success(f"**Generated:** {st.session_state.generated_word}")
        st.markdown("**Forensic Analysis:** Found verbatim match in corpus:")
        for i, item in enumerate(st.session_state.provenance_data):
            st.code(f"{i+1}: {item}", language="text")
    else:
        st.info("Awaiting generation...")
    
    st.divider()
    
    # Visualization is now fully adaptive
    if st.session_state.viz_system:
        # Use the prediction to center the visualization, or default
        pred = st.session_state.generated_word
        fig = plot_mathematical_boundary(st.session_state.viz_system, current_prediction=pred)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The diagram updates dynamically to show the decision boundary relevant to your last input.")
        else:
            st.warning("Insufficient data to plot boundary.")
