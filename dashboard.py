# dashboard.py
import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# Optional flask helper for downloads
from flask import send_from_directory

# -------------------------
# Config / artifact paths
# -------------------------
BASE = Path('.')
ART_NPZ = BASE / 'dashboard_artifacts.npz'
MODELS_JOBLIB = BASE / 'trained_models.joblib'
MODELS_PKL = BASE / 'trained_models.pkl'
RESULTS_CSV_CANDIDATES = [
    BASE / 'dashboard_results.csv',
    BASE / 'dashboard_result.csv',
    BASE / 'dashboard_results.csv',
    BASE / 'dashboard_result.csv',
    BASE / 'dashboard_results.csv',
    BASE / 'dashboard_results.cvs',  # common typo
    BASE / 'dashboard_result.cvs'
]
DF_CSV_CANDIDATES = [
    BASE / 'dashboard_df.csv',
    BASE / 'dashboard_df.csv',
    BASE / 'dashboard_df.cvs',
    BASE / 'dashboard_df.cvs'
]
FEATURES_JSON = BASE / 'feature_names.json'
LABEL_CLASSES_JSON = BASE / 'label_encoder_classes.json'

# -------------------------
# Load artifacts if present
# -------------------------
art = {}
if ART_NPZ.exists():
    try:
        loaded = np.load(ART_NPZ, allow_pickle=True)
        for k in loaded.files:
            art[k] = loaded[k]
        print("Loaded:", list(art.keys()))
    except Exception as e:
        print("Failed to read dashboard_artifacts.npz:", e)
else:
    print("dashboard_artifacts.npz not found (some visuals disabled).")

# load trained models
models = None
if MODELS_JOBLIB.exists():
    try:
        import joblib
        models = joblib.load(MODELS_JOBLIB)
        print("Loaded joblib models.")
    except Exception as e:
        print("Failed loading joblib:", e)
elif MODELS_PKL.exists():
    try:
        with open(MODELS_PKL, 'rb') as f:
            models = pickle.load(f)
            print("Loaded pickle models.")
    except Exception as e:
        print("Failed loading pickle:", e)
else:
    print("No trained_models.joblib / .pkl found.")

# load results dataframe (try multiple candidate filenames)
results_df = None
for p in RESULTS_CSV_CANDIDATES:
    if p.exists():
        try:
            results_df = pd.read_csv(p)
            print("Loaded results from", p.name)
            break
        except Exception as e:
            print("Failed loading", p.name, e)

# load main dataframe (if provided)
df = None
for p in DF_CSV_CANDIDATES:
    if p.exists():
        try:
            df = pd.read_csv(p)
            print("Loaded df from", p.name)
            break
        except Exception as e:
            print("Failed loading", p.name, e)

# support loading feature names & label classes
feature_names = None
if FEATURES_JSON.exists():
    try:
        feature_names = json.loads(FEATURES_JSON.read_text())
        print("Loaded feature_names.json")
    except Exception as e:
        print("Failed loading feature_names.json:", e)

label_classes = None
if LABEL_CLASSES_JSON.exists():
    try:
        label_classes = json.loads(LABEL_CLASSES_JSON.read_text())
        print("Loaded label_encoder_classes.json")
    except Exception as e:
        print("Failed loading label_encoder_classes.json:", e)

# fallback populate from art
X_scaled = art.get('X_scaled', None)
X_pca = art.get('X_pca', None)
X_tsne = art.get('X_tsne', None)
y_encoded = art.get('y_encoded', None)
y_train = art.get('y_train', None)
y_test = art.get('y_test', None)
predictions = art.get('y_pred', None)  # optional

# if feature_names None but X_scaled present, synthesize
if feature_names is None and X_scaled is not None:
    feature_names = [f"F{i+1}" for i in range(X_scaled.shape[1])]

# -------------------------
# Utility: find metric columns robustly
# -------------------------
def find_column(df, candidates):
    if df is None:
        return None
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        low = cand.lower()
        if low in cols:
            return cols[low]
    # try substring match
    for c in df.columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None

def find_metric_cols(df):
    """Return a dict of discovered metric columns by canonical name"""
    if df is None:
        return {}
    mapping = {}
    mapping['accuracy'] = find_column(df, ['accuracy','acc','accu'])
    mapping['f1'] = find_column(df, ['f1','f1-score','f1_score'])
    mapping['precision'] = find_column(df, ['precision','prec'])
    mapping['recall'] = find_column(df, ['recall','rec'])
    mapping['cv'] = find_column(df, ['cv','cv_score','cv-score','cv mean','cv_mean'])
    return mapping

metric_cols = find_metric_cols(results_df)

# -------------------------
# App & layout
# -------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server
app.title = "EEG Mental State Dashboard"

PASTEL_COLORS = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA', '#FFD9BA', '#E0BBE4', '#FFDFD3']

# Helper: build confusion fig
def build_confusion_fig(y_true, y_pred, labels=None, title='Confusion Matrix'):
    try:
        cm = confusion_matrix(y_true, y_pred)
    except Exception:
        cm = np.array([[0]])
    if labels is None:
        labels = [str(i) for i in range(cm.shape[0])]
    fig = px.imshow(cm, text_auto=True, labels=dict(x='Predicted', y='Actual', color='Count'),
                    x=labels, y=labels, color_continuous_scale='Blues')
    fig.update_layout(title=title, height=450)
    return fig

# Helper: ROC
def build_roc_fig(model, X, y_true, labels):
    fig = go.Figure()
    try:
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X)
            n_classes = y_score.shape[1]
            y_true_bin = np.eye(n_classes)[y_true]
            for i, lab in enumerate(labels):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
                roc_auc = auc(fpr, tpr)
                fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{lab} (AUC={roc_auc:.3f})'))
            fig.update_layout(title='ROC Curves', xaxis_title='FPR', yaxis_title='TPR', height=450)
        else:
            fig.add_annotation(text='predict_proba not available for this model.', showarrow=False)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'ROC failed: {e}', showarrow=False)
    return fig

# Helper: PR
def build_pr_fig(model, X, y_true, labels):
    fig = go.Figure()
    try:
        if hasattr(model, 'predict_proba'):
            y_score = model.predict_proba(X)
            n_classes = y_score.shape[1]
            y_true_bin = np.eye(n_classes)[y_true]
            for i, lab in enumerate(labels):
                prec, rec, _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
                ap = np.trapz(prec, rec)
                fig.add_trace(go.Scatter(x=rec, y=prec, name=f'{lab} (AP={ap:.3f})'))
            fig.update_layout(title='Precision-Recall Curves', xaxis_title='Recall', yaxis_title='Precision', height=450)
        else:
            fig.add_annotation(text='predict_proba not available for this model.', showarrow=False)
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f'PR failed: {e}', showarrow=False)
    return fig

# compute best metrics for overview cards (safe)
def safe_best_value(df, col):
    if df is None or col is None:
        return None
    try:
        return df[col].max()
    except Exception:
        return None

best_acc = safe_best_value(results_df, metric_cols.get('accuracy'))
best_f1 = safe_best_value(results_df, metric_cols.get('f1'))

# create layout
cards = dbc.Row([
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("Best Accuracy", className="card-title"),
        html.H3(f"{best_acc*100:.2f}%" if best_acc is not None else "N/A", className="card-text")
    ]), color=PASTEL_COLORS[0], inverse=False), width=3),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("Best F1-Score", className="card-title"),
        html.H3(f"{best_f1:.4f}" if best_f1 is not None else "N/A", className="card-text")
    ]), color=PASTEL_COLORS[1], inverse=False), width=3),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("Total Samples", className="card-title"),
        html.H3(f"{len(X_scaled):,}" if X_scaled is not None else (f"{len(df):,}" if df is not None else "N/A"), className="card-text")
    ]), color=PASTEL_COLORS[2], inverse=False), width=3),
    dbc.Col(dbc.Card(dbc.CardBody([
        html.H5("Features", className="card-title"),
        html.H3(f"{len(feature_names):,}" if feature_names is not None else "N/A", className="card-text")
    ]), color=PASTEL_COLORS[3], inverse=False), width=3),
], className="mb-3")

tabs = dcc.Tabs(id='tabs', value='tab-overview', children=[
    dcc.Tab(label='Overview', value='tab-overview'),
    dcc.Tab(label='Model Comparison', value='tab-comparison'),
    dcc.Tab(label='Dimensionality Reduction', value='tab-dr'),
    dcc.Tab(label='Visualizations', value='tab-viz'),
    dcc.Tab(label='Model Details', value='tab-model'),
    dcc.Tab(label='Data & Download', value='tab-data')
])

app.layout = dbc.Container([
    html.Br(),
    html.H1("EEG Mental State Dashboard", style={'textAlign': 'center'}),
    html.P("Interactive analytics for EEG mental state classification", style={'textAlign': 'center', 'color':'#666'}),
    html.Hr(),
    cards,
    tabs,
    html.Div(id='tab-content', style={'marginTop': 20}),
    html.Hr(),
    html.Div("Artifacts folder: " + str(BASE.resolve()), style={'fontSize':10, 'color':'#999'})
], fluid=True)


# -------------------------
# Callbacks for tabs
# -------------------------
@app.callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-overview':
        return html.Div([
            html.H4("Summary"),
            html.P(f"Models loaded: {len(models) if models is not None else 0}"),
            html.P(f"Results table: {'loaded' if results_df is not None else 'not loaded'}"),
            html.P(f"PCA available: {'yes' if X_pca is not None else 'no'}"),
            html.P(f"t-SNE available: {'yes' if X_tsne is not None else 'no'}"),
        ])
    if tab == 'tab-comparison':
        if results_df is None:
            return html.Div([html.P("No results CSV found. Place a results CSV in working dir with model metrics.")])
        # detect numeric metrics to plot (exclude Model column)
        numeric_cols = [c for c in results_df.columns if c.lower() != 'model' and pd.api.types.is_numeric_dtype(results_df[c])]
        if not numeric_cols:
            return html.Div([html.P("No numeric metric columns in results table.")])
        fig = go.Figure()
        for i, col in enumerate(numeric_cols):
            fig.add_trace(go.Bar(x=results_df['Model'], y=results_df[col], name=col))
        fig.update_layout(barmode='group', title='Model comparison across metrics', height=600)
        table = dbc.Table.from_dataframe(results_df.round(4), striped=True, bordered=True, hover=True, responsive=True)
        return html.Div([dcc.Graph(figure=fig), html.H4("Results table"), table])
    if tab == 'tab-dr':
        children = []
        if X_pca is not None:
            try:
                pca_df = pd.DataFrame(X_pca[:, :3], columns=['PC1','PC2','PC3'])
                if y_encoded is not None:
                    pca_df['label'] = [str(int(x)) for x in y_encoded[:len(pca_df)]]
                fig_pca = px.scatter_3d(pca_df, x='PC1', y='PC2', z='PC3', color='label' if 'label' in pca_df.columns else None,
                                       title='3D PCA', opacity=0.7)
                children.append(dcc.Graph(figure=fig_pca))
            except Exception as e:
                children.append(html.P(f"PCA plotting failed: {e}"))
        else:
            children.append(html.P("PCA not available."))
        if X_tsne is not None:
            try:
                tsne_df = pd.DataFrame(X_tsne, columns=['Dim1','Dim2'])
                if y_encoded is not None:
                    tsne_df['label'] = [str(int(x)) for x in y_encoded[:len(tsne_df)]]
                fig_tsne = px.scatter(tsne_df, x='Dim1', y='Dim2', color='label' if 'label' in tsne_df.columns else None, title='t-SNE (2D)', opacity=0.7)
                children.append(dcc.Graph(figure=fig_tsne))
            except Exception as e:
                children.append(html.P(f"t-SNE plotting failed: {e}"))
        else:
            children.append(html.P("t-SNE not available."))
        return html.Div(children)
    if tab == 'tab-viz':
        # try to include any arrays or df-based visuals
        viz_children = []
        # If original df exists, show basic histograms for first few numeric columns
        if df is not None:
            numcols = df.select_dtypes(include='number').columns.tolist()[:6]
            for col in numcols:
                fig = px.histogram(df, x=col, nbins=40, title=f"Histogram: {col}")
                viz_children.append(dcc.Graph(figure=fig))
        else:
            viz_children.append(html.P("Original dataframe not found - cannot auto-generate histograms."))
        # Add PCA / t-SNE if present
        if X_pca is not None:
            try:
                sample = min(5000, X_pca.shape[0])
                fig = px.scatter(x=X_pca[:sample,0], y=X_pca[:sample,1], title='PCA 2D (PC1 vs PC2)')
                viz_children.append(dcc.Graph(figure=fig))
            except:
                pass
        if X_tsne is not None:
            fig = px.scatter(x=X_tsne[:,0], y=X_tsne[:,1], title='t-SNE 2D')
            viz_children.append(dcc.Graph(figure=fig))
        return html.Div(viz_children)
    if tab == 'tab-model':
        # pick model list from results_df Model column else models dict
        options = []
        if results_df is not None and 'Model' in results_df.columns:
            options = results_df['Model'].astype(str).tolist()
        elif models:
            options = list(models.keys())
        if not options:
            return html.Div([html.P("No model list available. Load models or results CSV.")])
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Dropdown(id='model-select', options=[{'label':m,'value':m} for m in options], value=options[0]), width=4),
                dbc.Col(html.Div(id='model-summary'), width=8)
            ]),
            html.Br(),
            dbc.Row([
                dbc.Col(dcc.Graph(id='confusion-graph'), width=6),
                dbc.Col(dcc.Graph(id='roc-graph'), width=6)
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id='pr-graph'), width=6),
                dbc.Col(dcc.Graph(id='fi-graph'), width=6)
            ]),
            html.Br(),
            dbc.Row(dbc.Col(dcc.Graph(id='conf-dist'), width=12))
        ])
    if tab == 'tab-data':
        links = []
        for p in [ART_NPZ, MODELS_JOBLIB, MODELS_PKL]:
            if p.exists():
                links.append(html.Li(html.A(p.name, href=f"/download/{p.name}")))
        for p in RESULTS_CSV_CANDIDATES:
            if p.exists():
                links.append(html.Li(html.A(p.name, href=f"/download/{p.name}")))
        for p in DF_CSV_CANDIDATES:
            if p.exists():
                links.append(html.Li(html.A(p.name, href=f"/download/{p.name}")))
        if not links:
            return html.Div([html.P("No downloadable artifacts found in working directory.")])
        return html.Div([html.H5("Available files for download"), html.Ul(links)])
    return html.Div([html.P("Unknown tab")])

# -------------------------
# Callbacks for model details
# -------------------------
@app.callback(
    Output('confusion-graph','figure'),
    Output('roc-graph','figure'),
    Output('pr-graph','figure'),
    Output('fi-graph','figure'),
    Output('conf-dist','figure'),
    Output('model-summary','children'),
    Input('model-select','value')
)
def update_model_panels(selected):
    empty = go.Figure()
    empty.add_annotation(text="No data", showarrow=False)
    if (models is None) or (X_scaled is None) or (y_encoded is None):
        return empty, empty, empty, empty, empty, html.P("Missing trained models or X / y arrays in artifacts.")
    model = models.get(selected) if isinstance(models, dict) else None
    if model is None:
        return empty, empty, empty, empty, empty, html.P("Model object not found in trained models.")
    # predictions
    try:
        y_pred = model.predict(X_scaled)
    except Exception as e:
        return empty, empty, empty, empty, empty, html.P(f"Model predict failed: {e}")
    # confusion
    labels = label_classes if label_classes is not None else [str(int(x)) for x in sorted(np.unique(y_encoded))]
    fig_cm = build_confusion_fig(y_encoded, y_pred, labels=labels, title=f'Confusion Matrix - {selected}')
    # roc / pr
    fig_roc = build_roc_fig(model, X_scaled, y_encoded, labels)
    fig_pr = build_pr_fig(model, X_scaled, y_encoded, labels)
    # feature importance
    if hasattr(model, 'feature_importances_'):
        try:
            importances = model.feature_importances_
            fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values('Importance', ascending=True).tail(25)
            fig_fi = px.bar(fi_df, x='Importance', y='Feature', orientation='h', title=f'Top features - {selected}', height=450)
        except Exception as e:
            fig_fi = empty
            fig_fi.add_annotation(text=f'FI failed: {e}', showarrow=False)
    else:
        fig_fi = empty
        fig_fi.add_annotation(text='Feature importance not available for this model.', showarrow=False)
    # confidence distribution
    if hasattr(model, 'predict_proba'):
        try:
            y_proba = model.predict_proba(X_scaled)
            max_conf = y_proba.max(axis=1)
            fig_conf = go.Figure()
            for i, lab in enumerate(labels):
                mask = (y_encoded == i)
                fig_conf.add_trace(go.Histogram(x=max_conf[mask], name=str(lab), opacity=0.6))
            fig_conf.update_layout(barmode='overlay', title='Prediction confidence by true class', height=400)
        except Exception as e:
            fig_conf = empty
            fig_conf.add_annotation(text=f'Confidence failed: {e}', showarrow=False)
    else:
        fig_conf = empty
        fig_conf.add_annotation(text='predict_proba not available for this model.', showarrow=False)
    # metrics summary
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        acc = accuracy_score(y_encoded, y_pred)
        f1 = f1_score(y_encoded, y_pred, average='weighted')
        prec = precision_score(y_encoded, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_encoded, y_pred, average='weighted')
        summary = html.Div([
            html.H5(f"Model: {selected}"),
            html.P(f"Accuracy: {acc:.4f}"),
            html.P(f"F1-score: {f1:.4f}"),
            html.P(f"Precision: {prec:.4f}"),
            html.P(f"Recall: {rec:.4f}")
        ])
    except Exception:
        summary = html.P("Failed to compute summary metrics.")
    return fig_cm, fig_roc, fig_pr, fig_fi, fig_conf, summary

# -------------------------
# Download route
# -------------------------
@server.route('/download/<path:filename>')
def download_file(filename):
    p = Path(filename)
    # search in working directory
    if p.exists():
        return send_from_directory(directory=str(p.parent.resolve()), filename=p.name, as_attachment=True)
    # else search known artifact names
    for cand in [ART_NPZ, MODELS_JOBLIB, MODELS_PKL] + RESULTS_CSV_CANDIDATES + DF_CSV_CANDIDATES:
        if cand.exists() and cand.name == filename:
            return send_from_directory(directory=str(cand.parent.resolve()), filename=cand.name, as_attachment=True)
    return "File not found", 404

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # run with new API (dash>=2.10)
    app.run(debug=True, port=8050)