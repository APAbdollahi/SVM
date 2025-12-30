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
def get_sentence_model(sequence_length=3, injected_corpus=None):
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
    
    # Live Injection
    if injected_corpus:
        corpus.extend(injected_corpus)
    
    return _train_pipeline(corpus, "Linguistic Model", sequence_length=sequence_length)
@st.cache_data
def get_arithmetic_model(sequence_length=4, injected_corpus=None):
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
                
    if injected_corpus:
        arithmetic_corpus.extend(injected_corpus)
    # Math needs longer context "2 + 2 =" is 4 tokens usually
    return _train_pipeline(arithmetic_corpus, "Arithmetic Model", sequence_length=sequence_length)

@st.cache_data
def get_everyday_model(sequence_length=3, injected_corpus=None):
    """Builds the Everyday Routine Model (Routine Actions)"""
    corpus = [
        # Personal Actions (Extended for N=3)
        "I sat on the soft sofa.", "I ordered a large pizza.", "She cooked a nice dinner.", "He drove the red car.",
        "I washed the dirty dishes.", "He read a long book.", "She watched the news on tv.", "I drank a glass of water.",
        "We walked in the city park.", "They played a fun game.", "I opened the front door.", "She closed the glass window.",
        
        # Emotions & States
        "He felt very happy today.", "She felt quite sad yesterday.", "I was feeling very tired.", "The man was very angry.",
        "The woman was quite calm.", "The child was very sleepy.", "I felt very cold outside.", "He felt quite hot inside.",
        
        # Daily Routine
        "The woman went to her work.", "The boy played in the garden.", "The girl went to her school.",
        "I woke up very early.", "He went to sleep late.", "She ate her breakfast alone.", "We had a big lunch together.",
        
        # Simple Social Interactions
        "She called her best friend.", "He met his boss today.", "I saw my neighbor outside.", "We visited our family members.",
        "They talked for two hours.", "I sent a long message."
    ]
    
    if injected_corpus:
        corpus.extend(injected_corpus)
        
    return _train_pipeline(corpus, "Everyday Model", sequence_length=sequence_length)
def _train_pipeline(corpus, model_name, sequence_length):
    """Shared training logic for any corpus."""
    sequences_list = []
    targets = []
    
    for sentence in corpus:
        # Tokenize per sentence
        sent_tokens = [word.lower() for word in word_tokenize(sentence, preserve_line=True) if word.isalpha() or word in ['+', '-', '='] or word.isdigit()]
        
        if len(sent_tokens) <= sequence_length:
            continue
            
        # Generate sliding windows ONLY within this sentence
        for i in range(len(sent_tokens) - sequence_length):
            seq = sent_tokens[i:i + sequence_length]
            target = sent_tokens[i + sequence_length]
            sequences_list.append(" ".join(seq))
            targets.append(target)
            
    # Vectorization
    # Adaptive N-gram range: Ensure we capture up to the full sequence length
    ngram_max = max(4, sequence_length)
    vectorizer = TfidfVectorizer(ngram_range=(1, ngram_max))
    
    if not sequences_list: 
        return None
    X = vectorizer.fit_transform(sequences_list)
    y = np.array(targets)
    
    # The Algorithm: Linear SVM
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
# --- 4. Logic: Inference & Provenance Tracking ---
def detect_intent(text):
    """Heuristic router to decide between Math and Sentences."""
    text_lower = text.lower()
    math_keywords = ['plus', 'minus', 'equals', 'sum', '+', '-', '=', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    
    score = sum(1 for k in math_keywords if k in text_lower)
    if score > 0:
        return "arithmetic"
    return "sentence"
def generate_and_audit(input_text, system_dict):
    if not system_dict: return None, None, []
    
    model = system_dict["model"]
    vectorizer = system_dict["vectorizer"]
    sequence_length = system_dict["sequence_length"]
    sequences_list = system_dict["sequences"]
    corpus = system_dict["corpus"]
    
    tokenized_input = word_tokenize(input_text.lower())
    if len(tokenized_input) < sequence_length:
        return None, None, []
    
    last_sequence = " ".join(tokenized_input[-sequence_length:])
    
    try:
        input_vector = vectorizer.transform([last_sequence])
        if input_vector.sum() == 0:
            return None, ["(Input contains no known n-grams from training data)"], []
            
        # Get raw decision values for ranking
        # SVC without probability=True returns signed distance to hyperplane
        decision_values = model.decision_function(input_vector)[0]
        
        # Determine classes
        classes = model.classes_
        
        # If binary classification, decision_function returns scalar (distance from separating plane)
        if len(classes) == 2:
            # Not handled for brevity in multi-class dominant case, but standard logic applies
            # We assume multi-class for this rich corpus
            max_idx = 0 if decision_values < 0 else 1 # Simplified, typically OvO for SVC
            # Actually standard SVC OvO is complex. Let's rely on .predict for primary
            prediction = model.predict(input_vector)[0]
            candidates = [] # Binary is simple
        else:
            # Multi-class (OvO usually) generates n_classes * (n_classes - 1) / 2 scores
            # BUT decision_function shape for OvR (which is default for LinearSVC but not SVC)
            # SVC 'ovr' shape is (n_samples, n_classes). Let's trust the indices.
            
            # Map scores to classes
            # Note: SVC with decision_function_shape='ovr' (default) gives (n_samples, n_classes)
            top_indices = np.argsort(decision_values)[::-1]
            
            prediction = classes[top_indices[0]]
            
            # Prepare candidates list: (Word, Score)
            candidates = []
            for idx in top_indices[:5]: # Top 5
                candidates.append((classes[idx], decision_values[idx]))
            
    except Exception as e:
        return None, [str(e)], []
    
    # Provenance Audit
    search_phrase = f"{last_sequence} {prediction}"
    evidence = []
    
    for sentence in corpus:
        if search_phrase in sentence.lower():
            evidence.append(sentence)
            if len(evidence) >= 3: break 
            
    return prediction, evidence, candidates
    
# --- 5. Visualization Logic ---
def plot_mathematical_boundary(system_dict, current_prediction=None, comparison_class=None):
    if not system_dict: return None
    
    X = system_dict["X"]
    y = system_dict["y"]
    sequences = system_dict["sequences"]
    model_name = system_dict["name"]
    
    # Adaptive Class Selection
    all_classes = sorted(list(np.unique(y)))
    if len(all_classes) < 2: return None
    
    class1 = current_prediction if current_prediction in all_classes else all_classes[0]
    
    if comparison_class and comparison_class in all_classes and comparison_class != class1:
        class2 = comparison_class
    else:
        # Default to the nearest neighbor (or just next in list)
        idx = all_classes.index(class1)
        class2 = all_classes[(idx + 1) % len(all_classes)]
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
        title=f"{model_name}<br>Boundary: '{class1}' vs. '{class2}'",
        xaxis_title="Principal Component 1", 
        yaxis_title="Principal Component 2",
        template="plotly_white", height=450,
        margin=dict(t=50, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
# --- 6. Event Callbacks ---
def on_generate_click():
    current_input = st.session_state.user_text
    
    # 1. Detect Intent -> Switch Model
    # Get parameters from UI (session state)
    user_n = st.session_state.get('ngram_slider', 3)
    injections = st.session_state.get('injected_corpus', [])
    universe_mode = st.session_state.get('universe_selector', 'Auto-Detect (Text/Math)')
    
    system_dict = None
    
    # Explicit Universe Selection
    if universe_mode == "The Mechanism (Philosophical)":
        system_dict = get_sentence_model(sequence_length=user_n, injected_corpus=injections)
    elif universe_mode == "The Human Routine (Everyday)":
        system_dict = get_everyday_model(sequence_length=user_n, injected_corpus=injections)
    elif universe_mode == "The Calculator (Arithmetic)":
        system_dict = get_arithmetic_model(sequence_length=4, injected_corpus=injections)
    else:
        # Auto-Detect Default
        intent = detect_intent(current_input)
        if intent == "arithmetic":
            # Keep math strict (N=4) unless we want to demo breakage, but injection applies.
            system_dict = get_arithmetic_model(sequence_length=4, injected_corpus=injections)
        else:
            # Sentence model uses the slider
            system_dict = get_sentence_model(sequence_length=user_n, injected_corpus=injections)
        
    st.session_state.active_model_name = system_dict["name"]
        
    # 2. Predict
    pred, evidence, candidates = generate_and_audit(current_input, system_dict)
    
    if pred:
        st.session_state.user_text = f"{current_input} {pred}"
        st.session_state.generated_word = pred
        st.session_state.provenance_data = evidence
        st.session_state.candidates = candidates # Store for UI
        st.session_state.last_error = None
        # Store for visualization
        st.session_state.viz_system = system_dict
    else:
        if evidence and evidence[0].startswith("(Input"):
            st.session_state.last_error = evidence[0]
        else:
            st.session_state.last_error = f"Input too short (Tokens: {len(word_tokenize(current_input.lower()))}). Need at least {system_dict['sequence_length']} known words/tokens."
# --- 7. Main Interface Execution ---
if __name__ == "__main__":
    # Initialize Session State
    if 'user_text' not in st.session_state:
        st.session_state.user_text = "happy dog sat" # User requested default
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
    if 'candidates' not in st.session_state:
        st.session_state.candidates = []
    if 'injected_corpus' not in st.session_state:
        st.session_state.injected_corpus = []
    # Sidebar Extras for Pedagogical Workbench
    with st.sidebar:
        st.divider()
        st.header("⚙️ Pedagogical Workbench")
        
        st.info("Manipulate the model's 'Brain' to see how fragile intelligence is.")
        
        # 1. Context Slider
        st.markdown("**1. Memory Window (N-Grams)**")
        ngram_val = st.slider("Words to look back:", min_value=2, max_value=5, value=3, key="ngram_slider")
        st.caption(f"The model only sees the last **{ngram_val}** words. It has no idea what happened before.")
        
        # 2. Live Injection
        st.markdown("**2. Live Injection (Poisoning)**")
        with st.form("inject_form"):
            new_fact = st.text_input("Teach a 'False' Fact:", placeholder="e.g., The sky is green.")
            submit_inject = st.form_submit_button("Inject & Retrain")
            
            if submit_inject and new_fact:
                st.session_state.injected_corpus.append(new_fact)
                st.success(f"Injected: '{new_fact}'")
                # Force reload by calling with new args happens naturally on next click/rerun
                st.rerun()
        if st.session_state.injected_corpus:
            with st.expander("Active Injections"):
                for item in st.session_state.injected_corpus:
                    st.code(item)
                if st.button("Clear All Injections"):
                    st.session_state.injected_corpus = []
                    st.rerun()
        
        # 3. Linguistic Universe Selector
        st.markdown("**3. Linguistic Universe**")
        st.selectbox(
            "Select the active 'reality':",
            ["Auto-Detect (Text/Math)", "The Mechanism (Philosophical)", "The Human Routine (Everyday)", "The Calculator (Arithmetic)"],
            index=0,
            key="universe_selector"
        )
    
    # Ensure models are loaded (Warmup)
    # We load all to ensure quick switching
    get_sentence_model(sequence_length=st.session_state.ngram_slider, injected_corpus=st.session_state.injected_corpus)
    get_arithmetic_model(injected_corpus=st.session_state.injected_corpus)
    get_everyday_model(sequence_length=st.session_state.ngram_slider, injected_corpus=st.session_state.injected_corpus)

    # --- DEFINE COLUMNS ---
    col_left, col_right = st.columns([1, 1], gap="large")
    # --- LEFT COLUMN: Input & Simulation ---
    with col_left:
        st.subheader("1. The Simulation")
        st.caption(f"Active System: **{st.session_state.active_model_name}**")
        
        st.markdown("Input a phrase. Note how text only needs **3 words** now, but math maintains structure.")
        
        st.text_input("Input Sequence:", key="user_text")
        st.button("Generate Next Token", type="primary", on_click=on_generate_click)
        
        if st.session_state.last_error:
            st.error(st.session_state.last_error)
            
        # --- WOW FACTOR: Confidence Metrics ---
        if st.session_state.candidates:
            st.divider()
            st.markdown("### 🧠 Internal Calculations")
            st.caption("The SVM calculates the signed distance of your input vector to the hyperplanes of every possible next word. The 'winner' is the one with the highest positive score.")
            
            top_cand = st.session_state.candidates[0]
            
            # Display the winner prominently
            st.metric(label="Top Prediction", value=top_cand[0], delta=f"{top_cand[1]:.2f} (Dist)")
            
            # Show "Runner Ups"
            if len(st.session_state.candidates) > 1:
                st.write("**Top Alternative Candidates:**")
                cand_df = pd.DataFrame(st.session_state.candidates[1:], columns=["Token", "Distance"])
                st.dataframe(cand_df, hide_index=True, use_container_width=True)
    # --- RIGHT COLUMN: Provenance & Visualization ---
    with col_right:
        st.subheader("2. Provenance Audit")
        
        if st.session_state.provenance_data:
            st.success(f"**Generated:** {st.session_state.generated_word}")
            st.markdown(f"**Source Matches:** Found {len(st.session_state.provenance_data)} verbatim record(s).")
            with st.expander("View Source Records", expanded=True):
                for i, item in enumerate(st.session_state.provenance_data):
                    st.code(f"{item}", language="text")
        else:
            st.info("Awaiting generation...")
        
        st.divider()
        
        st.subheader("3. Geometric Boundary")
        
        # Visualization is now fully adaptive
        if st.session_state.viz_system and st.session_state.generated_word:
            # Use the prediction to center the visualization
            pred = st.session_state.generated_word
            
            # Interactive Control: Choice of what to compare against
            # Get all classes for the current system/input
            # We can use the candidates list if available to filter relevant ones
            system_y = st.session_state.viz_system["y"]
            all_possible = sorted(list(np.unique(system_y)))
            
            # Default options should include the top runners-up
            if st.session_state.candidates and len(st.session_state.candidates) > 1:
                # Suggest the second best as default comparison
                default_ix = 0
                second_best = st.session_state.candidates[1][0]
                if second_best in all_possible:
                    default_ix = all_possible.index(second_best)
            else:
                default_ix = 0
                
            col_ctrl1, col_ctrl2 = st.columns([2, 1])
            with col_ctrl1:
                compare_class = st.selectbox(
                    f"Show boundary between **'{pred}'** and:", 
                    [c for c in all_possible if c != pred],
                    index=default_ix if default_ix < len([c for c in all_possible if c != pred]) else 0
                )
            fig = plot_mathematical_boundary(st.session_state.viz_system, current_prediction=pred, comparison_class=compare_class)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                st.caption(f"Visualizing the mathematical cut between predicting **'{pred}'** vs **'{compare_class}'**.")
            else:
                st.warning("Insufficient data to plot boundary.")
        else:
            st.info("Generate a word to see its geometric neighborhood.")
