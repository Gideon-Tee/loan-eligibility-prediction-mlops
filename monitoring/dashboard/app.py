import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from datetime import datetime, timedelta
import sys
import os

# Add monitoring modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data_collector.prediction_logger import PredictionLogger
from drift_detector.drift_analyzer import DriftAnalyzer

# Page config
st.set_page_config(
    page_title="LoanFlow Model Monitoring",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-success {
        background-color: #d4edda;
        border-color: #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-color: #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-danger {
        background-color: #f8d7da;
        border-color: #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_prediction_data(days_back=7):
    """Load prediction data with caching"""
    try:
        logger = PredictionLogger()
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        return logger.get_predictions_for_date_range(start_date, end_date)
    except Exception as e:
        st.error(f"Error loading prediction data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def run_drift_analysis(days_back=7):
    """Run drift analysis with caching"""
    try:
        analyzer = DriftAnalyzer()
        success, results = analyzer.analyze_drift(days_back)
        return success, results
    except Exception as e:
        st.error(f"Error running drift analysis: {e}")
        return False, {"error": str(e)}

def main():
    # Header
    st.markdown('<h1 class="main-header">🎯 LoanFlow Model Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    days_back = st.sidebar.slider("Days to analyze", 1, 30, 7)
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    
    if auto_refresh:
        st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Drift Analysis", "📊 Prediction Analytics", "⚠️ Alerts"])
    
    with tab1:
        overview_tab(days_back)
    
    with tab2:
        drift_analysis_tab(days_back)
    
    with tab3:
        prediction_analytics_tab(days_back)
    
    with tab4:
        alerts_tab(days_back)

def overview_tab(days_back):
    st.header("📈 Model Performance Overview")
    
    # Load data
    predictions_df = load_prediction_data(days_back)
    
    if predictions_df.empty:
        st.warning("No prediction data available for the selected time period.")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_predictions = len(predictions_df)
        st.metric("Total Predictions", total_predictions)
    
    with col2:
        approval_rate = (predictions_df['prediction'] == 1).mean() * 100
        st.metric("Approval Rate", f"{approval_rate:.1f}%")
    
    with col3:
        avg_confidence = predictions_df['confidence'].mean() if 'confidence' in predictions_df.columns else 0
        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
    
    with col4:
        unique_users = predictions_df['user_id'].nunique() if 'user_id' in predictions_df.columns else 0
        st.metric("Unique Users", unique_users)
    
    # Prediction trends
    st.subheader("📊 Prediction Trends")
    
    if 'timestamp' in predictions_df.columns:
        predictions_df['timestamp'] = pd.to_datetime(predictions_df['timestamp'])
        predictions_df['date'] = predictions_df['timestamp'].dt.date
        
        daily_stats = predictions_df.groupby('date').agg({
            'prediction': ['count', 'mean'],
            'confidence': 'mean'
        }).round(3)
        
        daily_stats.columns = ['Total_Predictions', 'Approval_Rate', 'Avg_Confidence']
        daily_stats = daily_stats.reset_index()
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Daily Predictions', 'Approval Rate', 'Average Confidence', 'Prediction Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # Daily predictions
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['Total_Predictions'],
                      mode='lines+markers', name='Predictions', line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        # Approval rate
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['Approval_Rate'],
                      mode='lines+markers', name='Approval Rate', line=dict(color='#ff7f0e')),
            row=1, col=2
        )
        
        # Average confidence
        fig.add_trace(
            go.Scatter(x=daily_stats['date'], y=daily_stats['Avg_Confidence'],
                      mode='lines+markers', name='Confidence', line=dict(color='#2ca02c')),
            row=2, col=1
        )
        
        # Prediction distribution
        pred_counts = predictions_df['prediction_label'].value_counts()
        fig.add_trace(
            go.Pie(labels=pred_counts.index, values=pred_counts.values, name="Predictions"),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Model Performance Metrics")
        st.plotly_chart(fig, use_container_width=True)

def drift_analysis_tab(days_back):
    st.header("🔍 Data Drift Analysis")
    
    with st.spinner("Running drift analysis..."):
        success, results = run_drift_analysis(days_back)
    
    if not success:
        st.error(f"Drift analysis failed: {results.get('error', 'Unknown error')}")
        return
    
    # Drift status
    drift_detected = results.get('drift_detected', False)
    
    if drift_detected:
        st.markdown('<div class="alert-danger">⚠️ <strong>Data Drift Detected!</strong> Model performance may be degraded.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">✅ <strong>No Significant Drift Detected</strong> Model is performing within expected parameters.</div>', unsafe_allow_html=True)
    
    # Drift metrics
    col1, col2, col3 = st.columns(3)
    
    drift_summary = results.get('drift_summary', {})
    
    with col1:
        drift_share = drift_summary.get('drift_share', 0) * 100
        st.metric("Drift Share", f"{drift_share:.1f}%")
    
    with col2:
        data_points = results.get('data_points', {})
        st.metric("Reference Points", data_points.get('reference', 0))
    
    with col3:
        st.metric("Current Points", data_points.get('current', 0))
    
    # Report link
    if results.get('report_url'):
        st.info(f"📄 Detailed report saved to: {results['report_url']}")
    
    # Feature drift visualization
    st.subheader("📊 Feature Drift Analysis")
    
    # Load current data for visualization
    predictions_df = load_prediction_data(days_back)
    
    if not predictions_df.empty and 'features' in predictions_df.columns:
        features_df = pd.json_normalize(predictions_df['features'])
        
        # Create feature distribution plots
        numerical_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
        
        for feature in numerical_features:
            if feature in features_df.columns:
                fig = px.histogram(features_df, x=feature, nbins=30, 
                                 title=f'{feature} Distribution',
                                 color_discrete_sequence=['#1f77b4'])
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

def prediction_analytics_tab(days_back):
    st.header("📊 Prediction Analytics")
    
    predictions_df = load_prediction_data(days_back)
    
    if predictions_df.empty:
        st.warning("No prediction data available.")
        return
    
    # Feature analysis
    if 'features' in predictions_df.columns:
        features_df = pd.json_normalize(predictions_df['features'])
        
        st.subheader("🎯 Feature Impact Analysis")
        
        # Correlation with predictions
        if not features_df.empty:
            features_df['prediction'] = predictions_df['prediction']
            
            # Calculate correlations
            numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']
            correlations = []
            
            for col in numerical_cols:
                if col in features_df.columns:
                    corr = features_df[col].corr(features_df['prediction'])
                    correlations.append({'Feature': col, 'Correlation': corr})
            
            if correlations:
                corr_df = pd.DataFrame(correlations)
                fig = px.bar(corr_df, x='Feature', y='Correlation',
                           title='Feature Correlation with Approval',
                           color='Correlation',
                           color_continuous_scale='RdYlBu')
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature distributions by prediction
        st.subheader("📈 Feature Distributions by Outcome")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'ApplicantIncome' in features_df.columns:
                fig = px.box(features_df, x='prediction', y='ApplicantIncome',
                           title='Applicant Income by Prediction',
                           color='prediction')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'LoanAmount' in features_df.columns:
                fig = px.box(features_df, x='prediction', y='LoanAmount',
                           title='Loan Amount by Prediction',
                           color='prediction')
                st.plotly_chart(fig, use_container_width=True)

def alerts_tab(days_back):
    st.header("⚠️ Model Alerts & Recommendations")
    
    predictions_df = load_prediction_data(days_back)
    success, drift_results = run_drift_analysis(days_back)
    
    alerts = []
    
    # Check for various alert conditions
    if not predictions_df.empty:
        # Low prediction volume
        daily_predictions = len(predictions_df) / days_back
        if daily_predictions < 10:
            alerts.append({
                'type': 'warning',
                'title': 'Low Prediction Volume',
                'message': f'Average {daily_predictions:.1f} predictions per day. Consider investigating usage patterns.',
                'recommendation': 'Monitor user engagement and system availability.'
            })
        
        # Extreme approval rates
        approval_rate = (predictions_df['prediction'] == 1).mean()
        if approval_rate > 0.9:
            alerts.append({
                'type': 'warning',
                'title': 'High Approval Rate',
                'message': f'Approval rate is {approval_rate:.1%}. This may indicate model bias.',
                'recommendation': 'Review model performance and consider retraining.'
            })
        elif approval_rate < 0.1:
            alerts.append({
                'type': 'warning',
                'title': 'Low Approval Rate',
                'message': f'Approval rate is {approval_rate:.1%}. This may indicate overly conservative model.',
                'recommendation': 'Review model thresholds and business requirements.'
            })
        
        # Low confidence predictions
        if 'confidence' in predictions_df.columns:
            low_confidence_rate = (predictions_df['confidence'] < 0.7).mean()
            if low_confidence_rate > 0.3:
                alerts.append({
                    'type': 'danger',
                    'title': 'High Low-Confidence Predictions',
                    'message': f'{low_confidence_rate:.1%} of predictions have confidence < 0.7.',
                    'recommendation': 'Consider model retraining or feature engineering.'
                })
    
    # Drift alerts
    if success and drift_results.get('drift_detected'):
        alerts.append({
            'type': 'danger',
            'title': 'Data Drift Detected',
            'message': 'Significant data drift detected in input features.',
            'recommendation': 'Immediate model retraining recommended.'
        })
    
    # Display alerts
    if alerts:
        for alert in alerts:
            alert_class = f"alert-{alert['type']}"
            icon = "⚠️" if alert['type'] == 'warning' else "🚨"
            
            st.markdown(f'''
            <div class="{alert_class}">
                {icon} <strong>{alert['title']}</strong><br>
                {alert['message']}<br>
                <em>Recommendation: {alert['recommendation']}</em>
            </div>
            ''', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-success">✅ <strong>All Systems Normal</strong> No alerts detected.</div>', unsafe_allow_html=True)
    
    # Model health score
    st.subheader("🏥 Model Health Score")
    
    health_score = 100
    health_factors = []
    
    if alerts:
        for alert in alerts:
            if alert['type'] == 'danger':
                health_score -= 30
                health_factors.append(f"❌ {alert['title']}")
            elif alert['type'] == 'warning':
                health_score -= 15
                health_factors.append(f"⚠️ {alert['title']}")
    
    health_score = max(0, health_score)
    
    # Health score gauge
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = health_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Model Health Score"},
        delta = {'reference': 100},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    if health_factors:
        st.write("**Health Impact Factors:**")
        for factor in health_factors:
            st.write(f"• {factor}")

if __name__ == "__main__":
    main()