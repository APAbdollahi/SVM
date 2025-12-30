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
        
        # --- LOCATION & OBJECT VARIATIONS ---
        "The happy dog sat at the table.", "The lazy cat slept under the table.", "The brown dog hid under the table.",
        "The small mouse ran under the table.", "The big bird sat on the table.", "The white cat played at the table.",
        "The sleepy dog sat on the chair.", "The furry cat slept on the chair.", "The small bird watched from the chair.",
        "The playful mouse hid under the chair.", "The big dog jumped on the chair.", "The red toy was on the chair.",
        "The brown dog sat in the box.", "The white cat slept in the box.", "The small mouse hid in the box.",
        "The furry toy was in the box.", "The big bone was in the box.", "The playful cat played in the box.",
        "The dog ran on the green grass.", "The cat played on the green grass.", "The bird sat on the green grass.",
        "The big bone lay on the grass.", "The small toy lay on the grass.", "The furry dog slept on the grass.",
        
        # --- PREPOSITIONAL VARIATIONS ---
        "The dog stood by the door.", "The cat sat by the door.", "The mouse ran by the door.",
        "The bird flew near the window.", "The cat watched the window.", "The dog sat near the window.",
        "The big cat hid behind the chair.", "The small dog hid behind the table.", "The mouse hid behind the box.",
        "The white dog stood near the bed.", "The brown cat stood near the rug.", "The sleepy bird stood near the food.","zero plus zero equals zero.", "zero plus one equals one.", "zero plus two equals two.", 
        "zero plus three equals three.", "zero plus four equals four.", "zero plus five equals five.", 
        "zero plus six equals six.", "zero plus seven equals seven.", "zero plus eight equals eight.", 
        "zero plus nine equals nine.", "zero plus ten equals ten.",
        
        "one plus zero equals one.", "one plus one equals two.", "one plus two equals three.", 
        "one plus three equals four.", "one plus four equals five.", "one plus five equals six.", 
        "one plus six equals seven.", "one plus seven equals eight.", "one plus eight equals nine.", 
        "one plus nine equals ten.",

        "two plus zero equals two.", "two plus one equals three.", "two plus two equals four.", 
        "two plus three equals five.", "two plus four equals six.", "two plus five equals seven.", 
        "two plus six equals eight.", "two plus seven equals nine.", "two plus eight equals ten.",

        "three plus zero equals three.", "three plus one equals four.", "three plus two equals five.", 
        "three plus three equals six.", "three plus four equals seven.", "three plus five equals eight.", 
        "three plus six equals nine.", "three plus seven equals ten.",

        "four plus zero equals four.", "four plus one equals five.", "four plus two equals six.", 
        "four plus three equals seven.", "four plus four equals eight.", "four plus five equals nine.", 
        "four plus six equals ten.",

        "five plus zero equals five.", "five plus one equals six.", "five plus two equals seven.", 
        "five plus three equals eight.", "five plus four equals nine.", "five plus five equals ten.",

        "six plus zero equals six.", "six plus one equals seven.", "six plus two equals eight.", 
        "six plus three equals nine.", "six plus four equals ten.",

        "seven plus zero equals seven.", "seven plus one equals eight.", "seven plus two equals nine.", 
        "seven plus three equals ten.",

        "eight plus zero equals eight.", "eight plus one equals nine.", "eight plus two equals ten.",

        "nine plus zero equals nine.", "nine plus one equals ten.",

        "ten plus zero equals ten.",

        # Symbolic forms (The model treats these as distinct tokens)
        "0 + 0 = 0.", "0 + 1 = 1.", "0 + 2 = 2.", "0 + 3 = 3.", "0 + 4 = 4.", 
        "0 + 5 = 5.", "0 + 6 = 6.", "0 + 7 = 7.", "0 + 8 = 8.", "0 + 9 = 9.", "0 + 10 = 10.",

        "1 + 0 = 1.", "1 + 1 = 2.", "1 + 2 = 3.", "1 + 3 = 4.", "1 + 4 = 5.", 
        "1 + 5 = 6.", "1 + 6 = 7.", "1 + 7 = 8.", "1 + 8 = 9.", "1 + 9 = 10.",

        "2 + 0 = 2.", "2 + 1 = 3.", "2 + 2 = 4.", "2 + 3 = 5.", "2 + 4 = 6.", 
        "2 + 5 = 7.", "2 + 6 = 8.", "2 + 7 = 9.", "2 + 8 = 10.",

        "3 + 0 = 3.", "3 + 1 = 4.", "3 + 2 = 5.", "3 + 3 = 6.", "3 + 4 = 7.", 
        "3 + 5 = 8.", "3 + 6 = 9.", "3 + 7 = 10.",

        "4 + 0 = 4.", "4 + 1 = 5.", "4 + 2 = 6.", "4 + 3 = 7.", "4 + 4 = 8.", 
        "4 + 5 = 9.", "4 + 6 = 10.",

        "5 + 0 = 5.", "5 + 1 = 6.", "5 + 2 = 7.", "5 + 3 = 8.", "5 + 4 = 9.", 
        "5 + 5 = 10.",

        "6 + 0 = 6.", "6 + 1 = 7.", "6 + 2 = 8.", "6 + 3 = 9.", "6 + 4 = 10.",

        "7 + 0 = 7.", "7 + 1 = 8.", "7 + 2 = 9.", "7 + 3 = 10.",

        "8 + 0 = 8.", "8 + 1 = 9.", "8 + 2 = 10.",

        "9 + 0 = 9.", "9 + 1 = 10.",

        "10 + 0 = 10."
    ]
    # Pre-tokenize
    tokens = [word.lower() for sentence in corpus for word in word_tokenize(sentence, preserve_line=True) if word.isalpha()]
    
    # Create N-Grams
    sequence_length = 3
    sequences_list, targets = [], []
    
    for i in range(len(tokens) - sequence_length):
        seq = tokens[i:i + sequence_length]
        target = tokens[i + sequence_length]
        sequences_list.append(" ".join(seq))
        targets.append(target)
        
    # Vectorization
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sequences_list)
    y = np.array(targets)
    
    # The Algorithm: Linear SVM
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X, y)
    
    return svm_model, vectorizer, sequence_length, X, y, np.array(sequences_list), corpus

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

# --- 6. Main Interface Execution ---

# Initialize Session State
if 'provenance_data' not in st.session_state:
    st.session_state.provenance_data = None
if 'generated_word' not in st.session_state:
    st.session_state.generated_word = None

# Initialize Model
model, vectorizer, sequence_length, X, y, sequences, corpus = initialize_system()

# --- DEFINE COLUMNS (The Fix) ---
col_left, col_right = st.columns([1, 1])

# --- LEFT COLUMN: Input & Simulation (Restored from previous context) ---
with col_left:
    st.subheader("1. The Simulation")
    st.markdown("Input a phrase (3+ words) to see how the SVM geometrically maps it to a completion.")
    
    # Default text helps users start immediately
    user_input = st.text_input("Input Sequence:", "The happy dog sat on the")
    
    if st.button("Generate Next Token", type="primary"):
        with st.spinner("Calculating vector trajectory..."):
            # Artificial delay for dramatic effect
            time.sleep(0.5) 
            pred, evidence = generate_and_audit(user_input, model, vectorizer, sequence_length, sequences, corpus)
            
            if pred:
                st.session_state.generated_word = pred
                st.session_state.provenance_data = evidence
                st.success(f"**Generated Token:** {pred}")
            else:
                st.error(f"Input too short. Please use at least {sequence_length} words.")

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
        for i, item in enumerate(st.session_state.provenance_data):
            st.code(f"Source Record {i+1}: {item}", language="text")
        
        st.error("Conclusion: The output is a derivative of the training data.")
    else:
        st.info("Awaiting generation to perform forensic audit...")
    
    st.divider()
    
    st.subheader("3. Mathematical Visualization")
    st.info("ℹ️ **Instruction:** Hover your mouse over any dot. You will see the specific phrase from the training corpus that creates that data point.")
    
    # Dynamic Selectboxes for Visualization
    all_classes = sorted(list(np.unique(y)))
    idx_a = all_classes.index('rug') if 'rug' in all_classes else 0
    idx_b = all_classes.index('table') if 'table' in all_classes else 1
    
    vc1, vc2 = st.columns(2)
    viz_class_1 = vc1.selectbox("Target Word A", all_classes, index=idx_a)
    remaining_classes = [c for c in all_classes if c != viz_class_1]
    viz_class_2 = vc2.selectbox("Target Word B", remaining_classes, index=remaining_classes.index('table') if 'table' in remaining_classes else 0)

    fig = plot_mathematical_boundary(X, y, sequences, viz_class_1, viz_class_2)
    if fig: 
        st.plotly_chart(fig, use_container_width=True)
        st.caption("The line represents the mathematical 'rule' the model learned to separate these two outcomes.")
    else:
        st.warning("Not enough data points to plot these specific words against each other.")
