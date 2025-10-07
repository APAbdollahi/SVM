import streamlit as st
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import nltk
from nltk.tokenize import word_tokenize
import plotly.graph_objects as go
import random

# --- 1. Model Training (with Caching) ---

@st.cache_data
def train_model():
    """
    Trains the SVM model and TfidfVectorizer on the sample corpus.
    Now also returns the original sequences list for inspection.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading NLTK 'punkt' model... Please wait.")
        nltk.download('punkt')
        st.success("Download complete!")

    corpus = [
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
        "The lazy cat played with the dog.", "The playful dog saw the moon.", "The sleepy cat ate the big bone."
    ]

    tokens = [word.lower() for sentence in corpus for word in word_tokenize(sentence, preserve_line=True) if word.isalpha()]
    sequence_length = 3
    sequences_list, targets = [], []
    for i in range(len(tokens) - sequence_length):
        seq = tokens[i:i + sequence_length]
        target = tokens[i + sequence_length]
        sequences_list.append(" ".join(seq))
        targets.append(target)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sequences_list)
    y = np.array(targets)
    svm_model = SVC(kernel='linear', C=1.0, probability=True)
    svm_model.fit(X, y)
    return svm_model, vectorizer, sequence_length, X, y, np.array(sequences_list)

# --- 2. Prediction Function --- (No changes)
def predict_next_word(input_text, model, vectorizer, sequence_length):
    tokenized_input = word_tokenize(input_text.lower(), preserve_line=True)
    if len(tokenized_input) < sequence_length:
        return f"(Input text must have at least {sequence_length} words.)"
    last_sequence = " ".join(tokenized_input[-sequence_length:])
    input_vector = vectorizer.transform([last_sequence])
    return model.predict(input_vector)[0]

# --- 3. Visualization Function ---
def create_interactive_visualization(X, y, sequences, class1, class2):
    class1_indices = np.where(y == class1)[0]
    class2_indices = np.where(y == class2)[0]
    indices = np.concatenate([class1_indices, class2_indices])
    if len(class1_indices) < 1 or len(class2_indices) < 1:
        return None
    X_filtered, y_filtered, sequences_filtered = X[indices].toarray(), y[indices], sequences[indices]
    
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_filtered)
    
    svm_2d = SVC(kernel='linear', C=1.0).fit(X_2d, y_filtered)

    x_min, x_max = X_2d[:, 0].min() - 0.1, X_2d[:, 0].max() + 0.1
    y_min, y_max = X_2d[:, 1].min() - 0.1, X_2d[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    
    Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z_numeric = np.array([0 if label == class1 else 1 for label in Z]).reshape(xx.shape)

    fig = go.Figure()

    # Add the contour plot for the decision boundary
    fig.add_trace(go.Contour(
        x=xx[0], y=yy[:, 0], z=Z_numeric,
        colorscale='RdBu_r', # <-- THE FIX IS HERE
        opacity=0.8, showscale=False
    ))

    for i, cls in enumerate([class1, class2]):
        mask = (y_filtered == cls)
        fig.add_trace(go.Scatter(
            x=X_2d[mask, 0], y=X_2d[mask, 1],
            mode='markers',
            name=cls,
            marker=dict(color=f'rgb({255*i}, 0, {255*(1-i)})', symbol='circle', size=8, line_width=1, line_color='black'),
            customdata=sequences_filtered[mask],
            hovertemplate='<b>Input</b>: "%{customdata}"<br><b>Class</b>: '+cls+'<extra></extra>'
        ))

    fig.update_layout(
        title=f"Interactive 2D SVM Boundary for '{class1}' vs. '{class2}'",
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title="Next Word",
        template="plotly_white"
    )

    return {
        "fig": fig, "variance": np.sum(pca.explained_variance_ratio_),
        "class1_samples": random.sample(list(sequences[class1_indices]), min(5, len(class1_indices))),
        "class2_samples": random.sample(list(sequences[class2_indices]), min(5, len(class2_indices)))
    }

# --- 4. Streamlit GUI ---

st.set_page_config(layout="wide")
st.title("Next-Word Predictor using SVM")

with st.spinner("Loading and training model..."):
    svm_model, vectorizer, SEQ_LENGTH, X, y, sequences = train_model()

st.subheader("Interactive Prediction")
if 'generated_text' not in st.session_state: st.session_state.generated_text = "The happy dog sat"

def handle_prediction():
    st.session_state.generated_text += " " + predict_next_word(st.session_state.generated_text, svm_model, vectorizer, SEQ_LENGTH)
def handle_reset():
    st.session_state.generated_text = "The happy dog sat"

st.text_input("Enter your starting phrase or edit the generated text:", key="generated_text")
col1, col2 = st.columns(2)
col1.button("Predict Next Word", on_click=handle_prediction, use_container_width=True)
col2.button("Reset", on_click=handle_reset, use_container_width=True)
st.markdown(f"**Generated Text:** {st.session_state.generated_text}")
st.divider()

st.subheader("Interactive Model Visualization")
st.markdown("Hover over the dots to see the exact input sequence they represent.")

unique_targets = sorted(list(np.unique(y)))
c1, c2 = st.columns(2)
class1 = c1.selectbox("Choose the first word:", unique_targets, index=unique_targets.index('sofa'))
filtered_targets = [word for word in unique_targets if word != class1]
class2 = c2.selectbox("Choose the second word:", filtered_targets, index=filtered_targets.index('bone'))

if st.button("Visualize Decision Boundary", use_container_width=True):
    with st.spinner("Creating interactive visualization..."):
        viz_data = create_interactive_visualization(X, y, sequences, class1, class2)
        if viz_data:
            st.plotly_chart(viz_data["fig"], use_container_width=True)
            st.info(f"**Approximation Quality:** The two Principal Components capture **{viz_data['variance']:.1%}** of the original data's variance.")
            v_col1, v_col2 = st.columns(2)
            with v_col1:
                st.write(f"**Example inputs for `{class1}`:**")
                for sample in viz_data["class1_samples"]: st.code(f'"{sample}"  -->  {class1}')
            with v_col2:
                st.write(f"**Example inputs for `{class2}`:**")
                for sample in viz_data["class2_samples"]: st.code(f'"{sample}"  -->  {class2}')
        else:
            st.error(f"Could not generate visualization. Not enough examples for '{class1}' or '{class2}'.")
