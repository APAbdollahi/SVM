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
@st.cache_data
def initialize_system():
    # --- 1. Base Corpus (Simple Sentences) ---
    # Enhanced with more variety as requested
    base_corpus = [
        # Original Dog/Cat Universe
        "The happy dog sat on the rug.", "The lazy cat slept on the sofa.", "The playful dog chased the red toy.",
        "The sleepy cat watched the small bird.", "The brown dog ate the big bone.", "The white cat hid under the bed.",
        "The small dog played with the furry cat.", "The big cat sat near the food.", "The furry dog wanted a toy.",
        "The playful cat dropped the small mouse.", "The dog ran on the green mat.", "The cat slept under the warm sun.",
        "The happy dog wanted the bone.", "The lazy cat saw the sleepy dog.", "The playful dog sat on the sofa.",
        "The sleepy cat ate the white food.", "The brown dog chased the furry cat.", "The white cat played with a toy.",
        
        # ... (Keeping some originals) ...
        "The happy dog sat at the table.", "The lazy cat slept under the table.", "The brown dog hid under the table.",
        "The small mouse ran under the table.", "The big bird sat on the table.", "The white cat played at the table.",
        
        # New "Philosophical/Technical" Sentences (The Machine Context)
        "The machine processes the raw data.", "The algorithm minimizes the error rate.", 
        "The system retrieves the stored pattern.", "The logic follows the strict rule.",
        "The silicon chip processes the bit.", "The vector maps to the point.",
        "The graph shows the linear line.", "The model simulates the human speech.",
        "The output mimics the user input.", "The audit reveals the source text.",
        "The mathematics defines the boundary line.", "The geometry explains the output word.",
        
        # Simple Logic
        "If it rains the ground gets wet.", "If the sun shines the grass grows.",
        "The red light means stop now.", "The green light means go now.",
        "The early bird catches the worm.", "The quick brown fox jumps over.",
    ]
    
    # --- 2. Arithmetic Expansion (Programmatic) ---
    # Converting digits to words for consistent tokenization
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
    final_corpus = base_corpus + arithmetic_corpus
    
    # --- 3. Robust N-Gram Generation (Per Sentence) ---
    sequences_list = []
    targets = []
    sequence_length = 4  # INCREASED from 3 to 4 to capture 'two plus two equals'
    
    all_tokens = [] # Just for vocabulary/debugging
    
    for sentence in final_corpus:
        # Tokenize per sentence
        sent_tokens = [word.lower() for word in word_tokenize(sentence, preserve_line=True) if word.isalpha() or word in ['+', '-', '='] or word.isdigit()]
        
        if len(sent_tokens) <= sequence_length:
            continue
            
        all_tokens.extend(sent_tokens)
        
        # Generate sliding windows ONLY within this sentence
        for i in range(len(sent_tokens) - sequence_length):
            seq = sent_tokens[i:i + sequence_length]
            target = sent_tokens[i + sequence_length]
            sequences_list.append(" ".join(seq))
            targets.append(target)
            
    # Vectorization
    # Enable n-grams to capture word order (critical for arithmetic "2+5" vs "5+2" and specific phrases)
    vectorizer = TfidfVectorizer(ngram_range=(1, 4))
    # We might have duplicates in sequences_list (e.g. repeated structures), 
    # but SVM handles them (weighted density).
    
    if not sequences_list: 
        # Fallback if corpus is broken
        st.error("Corpus generation failed. No sequences found.")
        return None, None, 4, None, None, [], []
    X = vectorizer.fit_transform(sequences_list)
    y = np.array(targets)
    
    # The Algorithm: Linear SVM
    svm_model = SVC(kernel='linear', C=1000, class_weight='balanced')
    svm_model.fit(X, y)
    
    return svm_model, vectorizer, sequence_length, X, y, np.array(sequences_list), final_corpus
# --- 4. Logic: Inference & Provenance Tracking ---
def generate_and_audit(input_text, model, vectorizer, sequence_length, sequences_list, corpus):
    tokenized_input = word_tokenize(input_text.lower())
    if len(tokenized_input) < sequence_length:
        return None, None
    
    last_sequence = " ".join(tokenized_input[-sequence_length:])
    input_vector = vectorizer.transform([last_sequence])
    
    prediction = model.predict(input_vector)[0]
    
    search_phrase = f"{last_sequence} {prediction}"
    evidence = []
    
    for sentence in corpus:
        if search_phrase in sentence.lower():
            evidence.append(sentence)
            if len(evidence) >= 3: break 
            
    return prediction, evidence
    
# --- 5. Visualization Logic ---
def plot_mathematical_boundary(X, y, sequences, class1, class2):
    class1_indices = np.where(y == class1)[0]
    class2_indices = np.where(y == class2)[0]
    indices = np.concatenate([class1_indices, class2_indices])
    
    if len(class1_indices) < 2 or len(class2_indices) < 2: return None
    
    X_filtered, y_filtered = X[indices].toarray(), y[indices]
    sequences_filtered = sequences[indices]
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_filtered)
    
    svm_2d = SVC(kernel='linear', C=1.0).fit(X_2d, y_filtered)
    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_num = np.array([0 if l == class1 else 1 for l in Z]).reshape(xx.shape)
    fig = go.Figure()
    
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z_num, 
        colorscale=[[0, 'rgba(255, 0, 0, 0.1)'], [1, 'rgba(0, 0, 255, 0.1)']], 
        opacity=0.4, showscale=False, hoverinfo='skip', name="Decision Region"
    ))
    
    colors = {class1: '#D62728', class2: '#1F77B4'}
    symbols = {class1: 'circle', class2: 'diamond'}
    
    for cls in [class1, class2]:
        mask = (y_filtered == cls)
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1], 
            mode='markers', 
            name=f"Predicts '{cls}'",
            marker=dict(color=colors[cls], size=14, symbol=symbols[cls], line=dict(width=2, color='black')),
            customdata=sequences_filtered[mask],
            hovertemplate='<b>Phrase:</b> "%{customdata}"<br><b>Next Word:</b> '+cls+'<br><extra></extra>'
        ))
        
    fig.update_layout(
        title=f"Geometric Boundary: '{class1}' vs. '{class2}'",
        xaxis_title="Abstract Feature Dimension A", 
        yaxis_title="Abstract Feature Dimension B",
        template="plotly_white", height=500
    )
    return fig
# --- 6. Event Callbacks (The Fix for the API Error) ---
def on_generate_click():
    """
    This function runs BEFORE the script reruns. 
    It can safely modify the state linked to the widget.
    """
    # 1. Access the current input from state
    current_input = st.session_state.user_text
    
    # 2. Get the model (Cached, so it's fast)
    model, vectorizer, sequence_length, X, y, sequences, corpus = initialize_system()
    
    # 3. Predict
    pred, evidence = generate_and_audit(current_input, model, vectorizer, sequence_length, sequences, corpus)
    
    if pred:
        # 4. Update the Text Input's State
        st.session_state.user_text = f"{current_input} {pred}"
        
        # 5. Store metadata for the right column
        st.session_state.generated_word = pred
        st.session_state.provenance_data = evidence
        st.session_state.last_error = None
    else:
        st.session_state.last_error = f"Input too short. Please use at least {sequence_length} words."
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
# Initialize Model
model, vectorizer, sequence_length, X, y, sequences, corpus = initialize_system()
# --- DEFINE COLUMNS ---
col_left, col_right = st.columns([1, 1])
# --- LEFT COLUMN: Input & Simulation ---
with col_left:
    st.subheader("1. The Simulation")
    st.markdown("Input a phrase (3+ words) to see how the SVM geometrically maps it to a completion.")
    
    # The Widget (Linked to 'user_text')
    st.text_input("Input Sequence:", key="user_text")
    
    # The Button (Triggers the Callback)
    st.button("Generate Next Token", type="primary", on_click=on_generate_click)
    
    # Display Error if it occurred during callback
    if st.session_state.last_error:
        st.error(st.session_state.last_error)
# --- RIGHT COLUMN: Provenance & Visualization ---
with col_right:
    st.subheader("2. Provenance Audit")
    
    if st.session_state.provenance_data:
        st.markdown("The system generated a token. Does this represent 'learning' or 'retrieval'?")
        st.markdown("**Forensic Analysis:**")
        st.markdown(
            "The output was mandated because the input sequence exists verbatim in the source corpus. "
            "The model is not 'creating'; it is completing a pattern found in the following copyrighted entries:"
        )
        # Fix: enumerate starts at 0, but user probably wants 1-based index in UI
        for i, item in enumerate(st.session_state.provenance_data):
            st.code(f"Source Record {i+1}: {item}", language="text")
        
        st.success(f"**Last Generated Token:** {st.session_state.generated_word}")
        st.error("Conclusion: The output is a derivative of the training data.")
    else:
        st.info("Awaiting generation to perform forensic audit...")
    
    st.divider()
    
    st.subheader("3. Mathematical Visualization")
    st.info("ℹ️ **Instruction:** Hover your mouse over any dot. You will see the specific phrase from the training corpus that creates that data point.")
    
    # Dynamic Selectboxes for Visualization
    all_classes = sorted(list(np.unique(y)))
    
    # Robust default indexing
    idx_a = 0
    if 'three' in all_classes:
        idx_a = all_classes.index('three')
    elif len(all_classes) > 0:
        idx_a = 0
        
    # Create the first selectbox
    vc1, vc2 = st.columns(2)
    viz_class_1 = vc1.selectbox("Target Word A", all_classes, index=idx_a)
    
    # Filter remaining
    remaining_classes = [c for c in all_classes if c != viz_class_1]
    
    idx_b = 0
    if 'four' in remaining_classes:
        idx_b = remaining_classes.index('four')
    elif len(remaining_classes) > 0:
        idx_b = 0
        
    viz_class_2 = vc2.selectbox("Target Word B", remaining_classes, index=idx_b)
    if viz_class_1 and viz_class_2:
        fig = plot_mathematical_boundary(X, y, sequences, viz_class_1, viz_class_2)
        if fig: 
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The line represents the mathematical 'rule' the model learned to separate these two outcomes.")
        else:
            st.warning("Not enough data points to plot these specific words against each other.")
