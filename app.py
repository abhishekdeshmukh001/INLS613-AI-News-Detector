"""INLS 613: compare four classifiers on the same news article."""
from pathlib import Path
from html import escape
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
METHODS = {
    'Char n-gram + LR': 'mixed_char_detector.joblib',
    'Word TF-IDF + LR': 'mixed_word_detector.joblib',
    'MiniLM + LR': 'mixed_minilm_detector.joblib',
    'Naive Bayes': 'mixed_nb_detector.joblib',
}
BEST = 'Char n-gram + LR'
ENCODER = 'sentence-transformers/all-MiniLM-L6-v2'
st.set_page_config(page_title='AI News Text Detector', page_icon='📰', layout='centered')
st.markdown('''<style>
.block-container {max-width: 950px; padding-top: 2.4rem;}
h1 {letter-spacing: -.025em;}
[data-testid="stMetric"] {border: 1px solid rgba(112,135,117,.4); padding: 14px; border-radius: 10px;}
.primary-result {border-left: 6px solid #688573; background: rgba(112,135,117,.13); border-radius: 8px; padding: 16px 20px; margin: 12px 0;}
.primary-result h2 {margin: 0; padding: 4px 0; font-size: 1.75rem;}
.comparison-heading {text-align:center; margin: 26px 0 4px;}
.comparison-caption {text-align:center; opacity:.75; font-size:.9rem; margin: 0 0 16px;}
.comparison-wrap {width:100%; overflow-x:auto; border:1px solid rgba(128,128,128,.32); border-radius:10px;}
.comparison-table {width:100%; min-width:610px; table-layout:fixed; border-collapse:collapse; margin:0 !important; font-size:.95rem;}
.comparison-table th, .comparison-table td {text-align:center !important; vertical-align:middle !important; padding:14px 8px !important; border:0 !important; border-bottom:1px solid rgba(128,128,128,.24) !important;}
.comparison-table th {background:rgba(128,128,128,.12); font-weight:600;}
.comparison-table tbody tr:last-child td {border-bottom:0 !important;}
.comparison-table .best-row {background:rgba(104,133,115,.22); font-weight:700;}
.comparison-table th:first-child {width:36%;}
.comparison-table th:nth-child(2) {width:26%;}
.comparison-table th:nth-child(3), .comparison-table th:nth-child(4) {width:19%;}
</style>''', unsafe_allow_html=True)
st.caption('INLS 613 | Text Mining')
st.title('AI-Generated News Text Detector')
st.write('Paste a news article and compare four classifiers trained on BBC news and outputs from 11 AI generators in RAID.')

@st.cache_resource
def load_model(name):
    return joblib.load(ROOT / 'models' / METHODS[name])

@st.cache_resource
def load_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(ENCODER, device='cpu')

@st.cache_data
def load_examples():
    # Optional: original local test file, or a permitted-to-share cloud example file.
    paths = [ROOT / 'examples.csv', ROOT.parent / 'data' / 'all_models' / 'test.csv']
    for p in paths:
        if p.is_file():
            return pd.read_csv(p, usecols=['text','label'])
    return pd.DataFrame()

if not (ROOT / 'models' / METHODS[BEST]).is_file():
    st.error('Model files are missing. Run export_models.py in the project environment first.')
    st.stop()

def score(name, text):
    model = load_model(name)
    X = load_encoder().encode([text], show_progress_bar=False) if name == 'MiniLM + LR' else [text]
    ai_col = list(model.classes_).index(1)
    return float(model.predict_proba(X)[0, ai_col]), int(model.predict(X)[0])

def label(n):
    return 'AI-generated' if n == 1 else 'Human-written'

def analyze(text, actual=None):
    words = text.split()
    if len(words) < 50:
        st.warning(f'Enter at least 50 words. You entered {len(words)}.')
        return
    used = ' '.join(words[:300])
    with st.spinner('Checking text with the main classifier...'):
        p, pred = score(BEST, used)
    st.markdown(f'<div class="primary-result"><strong>MAIN PREDICTION · CHARACTER N-GRAMS + LR</strong><h2>{label(pred)}</h2><div>Selected as the best classifier on the original mixed test dataset.</div></div>', unsafe_allow_html=True)
    a,b,c = st.columns(3)
    a.metric('Human score', f'{1-p:.1%}')
    b.metric('AI score', f'{p:.1%}')
    c.metric('Words analyzed', min(len(words),300))
    st.progress(p)
    if actual is not None:
        msg = f'Actual label: {label(actual)}'
        (st.success if pred == actual else st.error)(msg + (' | Correct prediction' if pred == actual else ' | Incorrect prediction'))
    st.caption('Only the first 300 words are analyzed.')
    st.markdown('<h3 class="comparison-heading">Predictions from all four classifiers</h3><p class="comparison-caption">All four classifiers analyze the same article.</p>', unsafe_allow_html=True)
    rows = []
    for name in METHODS:
        if name == BEST:
            q, y = p, pred
        else:
            try:
                with st.spinner(f'Running {name}...'):
                    q, y = score(name, used)
            except Exception as exc:
                st.warning(f'{name} could not run. {type(exc).__name__}: {exc}')
                rows.append({'Classifier': name, 'Prediction': 'Unavailable', 'Human score': 'N/A', 'AI score': 'N/A'})
                continue
        rows.append({'Classifier': name + (' (Best model)' if name == BEST else ''),
                     'Prediction': label(y), 'Human score': f'{1-q:.1%}', 'AI score': f'{q:.1%}'})
    headers = ['Classifier', 'Prediction', 'Human score', 'AI score']
    header_html = ''.join(f'<th scope="col">{escape(h)}</th>' for h in headers)
    body_html = []
    for row in rows:
        css = ' class="best-row"' if row['Classifier'].endswith('(Best model)') else ''
        cells = ''.join(f'<td>{escape(str(row[h]))}</td>' for h in headers)
        body_html.append(f'<tr{css}>{cells}</tr>')
    st.markdown('<div class="comparison-wrap"><table class="comparison-table" aria-label="Classifier predictions"><thead><tr>' + header_html + '</tr></thead><tbody>' + ''.join(body_html) + '</tbody></table></div>', unsafe_allow_html=True)
    st.markdown('<p class="comparison-caption" style="margin-top:10px">These percentages are model scores for this article, not proof of authorship.</p>', unsafe_allow_html=True)

examples = load_examples()
if examples.empty:
    tab = st.container()
else:
    tab, ex_tab = st.tabs(['Paste an article','Test dataset examples'])
with tab:
    with st.form('article_form'):
        text = st.text_area('News article', height=220, placeholder='Paste at least 50 words...')
        submit = st.form_submit_button('Analyze text', type='primary')
    if submit:
        analyze(text)
if not examples.empty:
    with ex_tab:
        choice = st.radio('Original label', ['Human-written','AI-generated'], horizontal=True)
        group = examples[examples.label == int(choice == 'AI-generated')].reset_index(drop=True)
        if not group.empty:
            idx = st.number_input('Example number', min_value=1, max_value=len(group), value=1)
            item = group.iloc[int(idx)-1]
            st.text_area('Example article', str(item.text), height=200, disabled=True)
            if st.button('Analyze example', type='primary'):
                analyze(str(item.text), int(item.label))
