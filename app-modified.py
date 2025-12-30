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
    # A closed-universe corpus designed to demonstrate verbatim retention
    corpus = [
        
        # --- ORIGINAL BASELINE (The Core "Linguistic Universe") ---
        "The happy dog sat on the rug.", "The lazy cat slept on the sofa.", "The playful dog chased the red toy.",
        "The sleepy cat watched the small bird.", "The brown dog ate the big bone.", "The white cat hid under the bed.",
        "The small dog played with the furry cat.", "The big cat sat near the food.", "The furry dog wanted a toy.",
        "The playful cat dropped the small mouse.", "The dog ran on the green mat.", "The cat slept under the warm sun.",
        "The happy dog wanted the bone.", "The lazy cat saw the sleepy dog.", "The playful dog sat on the sofa.",
        "The sleepy cat ate the white food.", "The brown dog chased the furry cat.", "The white cat played with a toy.",
        "The small dog slept on the big bed.", "The big cat hid behind the sofa.", "The furry dog watched the bird.",
        "The playful cat wanted a big toy.", "The happy cat sat on the mat.", "The lazy dog slept on the floor.",
        "The playful cat chased the brown mouse.", "The sleepy dog watched the happy bird.", "The brown cat ate the food.",
        "The white dog hid under the rug.", "The small cat played with the big dog.", "The big dog sat near the toy.",
        "The furry cat wanted a small bone.", "The playful dog dropped the red toy.", "The cat ran on the soft sofa.",
        "The dog slept under the bright moon.", "The happy cat wanted the food.", "The lazy dog saw the playful cat.",
        "The playful cat sat on the bed.", "The sleepy dog ate the big bone.", "The brown cat chased the white dog.",
        "The white dog played with a red toy.", "The small cat slept on the furry rug.", "The big dog hid behind the mat.",
        "The furry cat watched the small bird.", "The playful dog wanted a new toy.", "The sleepy cat sat on the sofa.",
        "The happy dog slept on the big bed.", "The lazy cat chased the toy.", "The brown dog watched the cat.",
        "The white cat ate the small food.", "The small dog hid under the sofa.", "The big cat played with the toy.",
        "The furry dog sat near the bone.", "The playful cat wanted the mouse.", "The sleepy dog dropped the toy.",
        "The brown cat ran on the rug.", "The white dog slept under the sun.", "The happy cat saw the dog.",
        "The lazy dog wanted the food.", "The playful cat watched the sleepy dog.", "The sleepy dog sat on the mat.",
        "The brown cat ate the big bone.", "The white dog chased the small cat.", "The small cat played with a toy.",
        "The big dog slept on the soft bed.", "The furry cat hid behind the rug.", "The playful dog watched the bird.",
        "The sleepy cat wanted a small toy.", "The happy dog sat on the bed.", "The lazy cat slept on the rug.",
        "The playful dog chased the mouse.", "The sleepy cat watched the brown dog.", "The brown dog ate the food.",
        "The white cat hid under the mat.", "The small dog played with the white cat.", "The big cat sat near the bone.",
        "The furry dog wanted a red toy.", "The playful cat dropped the toy.", "The dog ran on the floor.",
        "The cat slept under the moon.", "The happy dog saw the lazy cat.", "The lazy cat wanted a bone.",
        "The playful dog watched the small bird.", "The sleepy cat sat on the bed.", "The brown dog chased the cat.",
        "The white cat played with the red toy.", "The small dog slept on the sofa.", "The big cat hid behind the bed.",
        "The furry dog watched the playful cat.", "The playful cat ate the food.", "The sleepy dog wanted the bone.",
        "The happy cat dropped the mouse.", "The lazy dog ran on the mat.", "The brown cat slept in the sun.",
        "The white dog saw the small bird.", "The small cat wanted the food.", "The big dog played with the toy.",
        "The furry cat sat on the sofa.", "The playful dog slept on the rug.", "The sleepy cat chased the mouse.",
        "The brown dog saw the white cat.", "The white cat wanted a toy.", "The small dog dropped the bone.",
        "The big cat ran behind the sofa.", "The furry dog ate the small food.", "The happy cat played with the dog.",
        "The lazy dog watched the bird.", "The playful cat slept on the bed.", "The sleepy dog chased the cat.",
        "The brown cat hid under the rug.", "The white dog wanted the bone.", "The small cat sat on the mat.",
        "The big dog ate the big bone.", "The furry cat played with a toy.", "The happy dog chased the small cat.",
        "The lazy cat dropped the toy.", "The playful dog hid behind the bed.", "The sleepy cat saw the furry dog.",
        "The brown dog slept on the sofa.", "The white cat ran on the mat.", "The small dog wanted a big toy.",
        "The big cat watched the moon.", "The furry dog played with the cat.", "The happy cat ate the white food.",
        "The lazy dog sat on the bed.", "The playful cat saw the brown dog.", "The sleepy dog played with the toy.",
        "The brown cat wanted the bone.", "The white dog dropped the toy.", "The small cat ran under the sofa.",
        "The big dog watched the sleepy cat.", "The furry cat ate the big food.", "The happy dog played with a red toy.",
        "The lazy cat hid behind the mat.", "The playful dog ate the small bone.", "The sleepy cat wanted the food.",
        "The brown dog saw the bird.", "The white cat slept on the furry rug.", "The small dog chased the big cat.",
        "The big cat wanted a small toy.", "The furry dog dropped the bone.", "The happy cat ran on the floor.",
        "The lazy dog played with the cat.", "The playful cat hid under the bed.", "The sleepy dog saw the sun.",
        "The brown cat played with the dog.", "The white dog sat on the sofa.", "The small cat ate the food.",
        "The big dog chased the mouse.", "The furry cat slept on the mat.", "The happy dog hid behind the sofa.",
        "The lazy cat saw the red toy.", "The playful dog wanted food.", "The sleepy dog dropped the toy.",
        "The brown dog played with a bone.", "The white cat watched the dog.", "The small dog wanted to sleep.",
        "The big cat played on the rug.", "The furry dog saw the white cat.", "The happy cat wanted a big bone.",
        "The lazy dog chased the small mouse.", "The playful cat ate the white food.", "The sleepy dog hid under the bed.",
        "The brown cat sat near the toy.", "The white dog slept on the mat.", "The small cat watched the big bird.",
        "The big dog wanted the red toy.", "The furry cat dropped the mouse.", "The happy dog ran on the sofa.",
        "The lazy cat played with the dog.", "The playful dog saw the moon.", "The sleepy cat ate the big bone.",
        
        # --- LOCATION & OBJECT VARIATIONS (Breaking the "Bed/Rug" Bias) ---
        "The happy dog sat at the table.", "The lazy cat slept under the table.", "The brown dog hid under the table.",
        "The small mouse ran under the table.", "The big bird sat on the table.", "The white cat played at the table.",
        
        "The sleepy dog sat on the chair.", "The furry cat slept on the chair.", "The small bird watched from the chair.",
        "The playful mouse hid under the chair.", "The big dog jumped on the chair.", "The red toy was on the chair.",
        
        "The brown dog sat in the box.", "The white cat slept in the box.", "The small mouse hid in the box.",
        "The furry toy was in the box.", "The big bone was in the box.", "The playful cat played in the box.",
        
        "The dog ran on the green grass.", "The cat played on the green grass.", "The bird sat on the green grass.",
        "The big bone lay on the grass.", "The small toy lay on the grass.", "The furry dog slept on the grass.",
        
        # --- PREPOSITIONAL VARIATIONS (Breaking "On/Under" Bias) ---
        "The dog stood by the door.", "The cat sat by the door.", "The mouse ran by the door.",
        "The bird flew near the window.", "The cat watched the window.", "The dog sat near the window.",
        "The big cat hid behind the chair.", "The small dog hid behind the table.", "The mouse hid behind the box.",
        "The white dog stood near the bed.", "The brown cat stood near the rug.", "The sleepy bird stood near the food."
    ]
    # Pre-tokenize with preserve_line=True to treat list elements as individual sentences
    tokens = [word.lower() for sentence in corpus for word in word_tokenize(sentence, preserve_line=True) if word.isalpha()]
    
    # Create N-Grams (Context Windows)
    sequence_length = 3
    sequences_list, targets = [], []
    
    for i in range(len(tokens) - sequence_length):
        seq = tokens[i:i + sequence_length]
        target = tokens[i + sequence_length]
        sequences_list.append(" ".join(seq))
        targets.append(target)
        
    # Vectorization (Mathematizing the text)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sequences_list)
    y = np.array(targets)
    
    # The Algorithm: Linear Support Vector Machine
    # Chosen for its mathematical transparency compared to Neural Networks
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X, y)
    
    return svm_model, vectorizer, sequence_length, X, y, np.array(sequences_list), corpus

# --- 4. Logic: Inference & Provenance Tracking ---
def generate_and_audit(input_text, model, vectorizer, sequence_length, sequences_list, corpus):
    # 1. Prepare Input
    tokenized_input = word_tokenize(input_text.lower())
    if len(tokenized_input) < sequence_length:
        return None, None
    
    last_sequence = " ".join(tokenized_input[-sequence_length:])
    input_vector = vectorizer.transform([last_sequence])
    
    # 2. Mathematical Prediction
    prediction = model.predict(input_vector)[0]
    
    # 3. Provenance Audit
    # Identify the exact source in the training data that mandated this output
    search_phrase = f"{last_sequence} {prediction}"
    evidence = []
    
    for sentence in corpus:
        if search_phrase in sentence.lower():
            evidence.append(sentence)
            if len(evidence) >= 3: break 
            
    return prediction, evidence

# --- 5. Visualization Logic ---
def plot_mathematical_boundary(X, y, sequences, class1, class2):
    # Filters data to show only two words to demonstrate the geometric boundary
    class1_indices = np.where(y == class1)[0]
    class2_indices = np.where(y == class2)[0]
    indices = np.concatenate([class1_indices, class2_indices])
    
    if len(class1_indices) < 2 or len(class2_indices) < 2: return None
    
    X_filtered, y_filtered = X[indices].toarray(), y[indices]
    
    # Retrieve the text sequences corresponding to these vectors
    sequences_filtered = sequences[indices]
    
    # Dimensionality Reduction for visual representation
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_filtered)
    svm_2d = SVC(kernel='linear', C=1.0).fit(X_2d, y_filtered)

    # Grid for decision boundary
    x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
    y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_num = np.array([0 if l == class1 else 1 for l in Z]).reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Contour(x=xx[0], y=yy[:, 0], z=Z_num, colorscale='RdBu', opacity=0.3, showscale=False))
    
    colors = {class1: 'red', class2: 'blue'}
    for cls in [class1, class2]:
        mask = (y_filtered == cls)
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1], 
            mode='markers', 
            name=cls,
            marker=dict(color=colors[cls], size=12, line=dict(width=1, color='black')),
            # Injecting the text data into the chart
            customdata=sequences_filtered[mask],
            hovertemplate='<b>Sentence Segment:</b> "%{customdata}"<br><b>Forced Outcome:</b> '+cls+'<extra></extra>'
        ))
        
    fig.update_layout(
        title="Visualizing the 'Decision': A Geometric Boundary",
        xaxis_title="Vector Dimension A", yaxis_title="Vector Dimension B",
        template="plotly_white", height=400, margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# --- 6. Main Interface ---
st.markdown("## De-Mystifying Text Generation: An Inquiry into Provenance")
st.markdown("A demonstration for the purpose of analyzing algorithmic distinctiveness and source retention.")

with st.spinner("Initializing Vector Space..."):
    svm_model, vectorizer, SEQ_LENGTH, X, y, sequences, corpus = initialize_system()

# State Management
if 'text_buffer' not in st.session_state: st.session_state.text_buffer = "The happy dog sat"
if 'provenance_data' not in st.session_state: st.session_state.provenance_data = []

def perform_generation():
    pred, evidence = generate_and_audit(
        st.session_state.text_buffer, svm_model, vectorizer, SEQ_LENGTH, sequences, corpus
    )
    if pred:
        st.session_state.text_buffer += " " + pred
        st.session_state.provenance_data = evidence
    else:
        st.warning(f"The algorithm requires a context window of {SEQ_LENGTH} words.")

def reset_stream():
    st.session_state.text_buffer = "The happy dog sat"
    st.session_state.provenance_data = []

# Layout
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Text Generation Stream")
    st.text_area("Context Window (Input)", key="text_buffer", height=100)
    
    c1, c2 = st.columns(2)
    c1.button("Generate Next Token", on_click=perform_generation, type="primary", use_container_width=True)
    c2.button("Reset Stream", on_click=reset_stream, use_container_width=True)

    if st.session_state.provenance_data:
         st.success(f"Generated Token: **{st.session_state.text_buffer.split()[-1]}**")

    st.markdown("#### 3. Mathematical Visualization")
    # Added explicit instruction as requested
    st.info("ℹ️ **Instruction:** Please hover your mouse on a dot to see which sentence in the training corpus it represents.")
    st.caption("The decision to select 'rug' vs 'bed' is not cognitive; it is the result of a vector falling on a specific side of a mathematical line.")
    
    # Static visualization for simplicity
    fig = plot_mathematical_boundary(X, y, sequences, 'rug', 'bed')
    if fig: st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("2. Provenance Audit")
    
    if st.session_state.provenance_data:
        st.markdown(
            "The system generated a token. Does this represent 'learning' or 'retrieval'?"
        )
        st.markdown("**Forensic Analysis:**")
        st.markdown(
            "The output was mandated because the input sequence exists verbatim in the source corpus. "
            "The model is not 'creating'; it is completing a pattern found in the following copyrighted entries:"
        )
        for i, item in enumerate(st.session_state.provenance_data):
            st.code(f"Source Record {i+1}: {item}", language="text")
        
        st.error("Conclusion: The output is a derivative of the training data.")
    else:
        st.info("Awaiting generation to perform forensic audit...")
