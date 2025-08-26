import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import json

# Page configuration
st.set_page_config(
    page_title="TTC Subway Delay Predictor",
    page_icon="🚇",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .prediction-box {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the prediction data"""
    try:
        # Load enriched predictions
        predictions_df = pd.read_csv("src/routes/resources/enriched_predictions_full.csv")
        
        # Load station data JSON for additional info
        with open("src/routes/resources/station_data.json", "r") as f:
            station_data = json.load(f)
            
        return predictions_df, station_data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

def create_delay_map(predictions_df, selected_hour, selected_month=None, selected_day_num=None):
    """Create an interactive map showing delay predictions"""
    if predictions_df is None:
        return None
    
    # Filter data for selected parameters
    filtered_data = predictions_df[predictions_df['hour'] == selected_hour].copy()
    
    if selected_month is not None:
        filtered_data = filtered_data[filtered_data['month'] == selected_month]
    
    if selected_day_num is not None:
        filtered_data = filtered_data[filtered_data['day_of_week'] == selected_day_num]
    
    if filtered_data.empty:
        st.warning(f"No data available for the selected parameters")
        return None
    
    # Create the map
    title = f"TTC Subway Delay Predictions - {selected_hour:02d}:00"
    if selected_month is not None:
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        title += f" ({month_names[selected_month-1]})"
    if selected_day_num is not None:
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        title += f" ({day_names[selected_day_num]})"
    
    fig = px.scatter_map(
        filtered_data,
        lat='latitude',
        lon='longitude',
        size='likelihood_of_delay',
        color='likelihood_of_delay',
        hover_name='station',
        hover_data=['likelihood_of_delay', 'delay_severity', 'delay_length'],
        color_continuous_scale='RdYlGn_r',  # Red to Green (red = high delay likelihood)
        size_max=20,
        zoom=10,
        center={'lat': 43.6532, 'lon': -79.3832},  # Toronto center
        title=title
    )
    
    fig.update_layout(
        map_style='carto-positron',
        height=600,
        margin={'r': 0, 't': 30, 'l': 0, 'b': 0}
    )
    
    return fig

def create_delay_timeline(predictions_df, selected_station):
    """Create a timeline chart showing delay predictions throughout the day"""
    if predictions_df is None or selected_station is None:
        return None
    
    station_data = predictions_df[predictions_df['station'] == selected_station].copy()
    
    if station_data.empty:
        st.warning(f"No data available for station {selected_station}")
        return None
    
    # Sort by hour
    station_data = station_data.sort_values('hour')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=station_data['hour'],
        y=station_data['likelihood_of_delay'],
        mode='lines+markers',
        name='Delay Likelihood',
        line=dict(color='#1f77b4', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title=f"Delay Predictions for {selected_station}",
        xaxis_title="Hour of Day",
        yaxis_title="Likelihood of Delay",
        height=400,
        showlegend=False
    )
    
    fig.update_xaxes(tickmode='linear', tick0=0, dtick=2)
    fig.update_yaxes(range=[0, 1])
    
    return fig

def get_delay_prediction(predictions_df, station, hour, month=None, day_of_week=None):
    """Get specific delay prediction for a station and parameters"""
    if predictions_df is None:
        return None
    
    # Build filter conditions
    conditions = [
        (predictions_df['station'] == station),
        (predictions_df['hour'] == hour)
    ]
    
    if month is not None:
        conditions.append(predictions_df['month'] == month)
    
    if day_of_week is not None:
        conditions.append(predictions_df['day_of_week'] == day_of_week)
    
    # Apply all conditions
    prediction = predictions_df[np.logical_and.reduce(conditions)]
    
    if prediction.empty:
        return None
    
    return prediction.iloc[0]

def main():
    # Header
    st.markdown('<h1 class="main-header">🚇 TTC Subway Delay Predictor</h1>', unsafe_allow_html=True)
    
    # Load data
    predictions_df, station_data = load_data()
    
    if predictions_df is None:
        st.error("Failed to load data. Please check the data files.")
        return
    
    # Sidebar for controls
    st.sidebar.header("Controls")
    
    # Time selection
    selected_hour = st.sidebar.slider(
        "Select Hour of Day",
        min_value=0,
        max_value=23,
        value=8,
        help="Choose the hour to view delay predictions"
    )
    
    # Month selection
    selected_month = st.sidebar.selectbox(
        "Select Month",
        options=sorted(predictions_df['month'].unique()),
        index=0,  # January
        help="Choose the month to view predictions"
    )
    
    # Day of week selection
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    selected_day = st.sidebar.selectbox(
        "Select Day of Week",
        options=day_names,
        index=0,  # Monday
        help="Choose the day of week to view predictions"
    )
    selected_day_num = day_names.index(selected_day)
    
    # Station selection
    unique_stations = sorted(predictions_df['station'].unique())
    selected_station = st.sidebar.selectbox(
        "Select Station",
        options=unique_stations,
        index=0 if 'BLOOR-YONGE' in unique_stations else 0,
        help="Choose a station to view detailed predictions"
    )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📍 Interactive Delay Map")
        
        # Create and display the map
        map_fig = create_delay_map(predictions_df, selected_hour, selected_month, selected_day_num)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        
        # Map legend
        st.markdown("""
        **Map Legend:**
        - **Circle Size**: Likelihood of delay (larger = higher likelihood)
        - **Circle Color**: Delay likelihood (Red = High, Green = Low)
        """)
    
    with col2:
        st.subheader("📊 Station Details")
        
        # Get current prediction
        current_prediction = get_delay_prediction(predictions_df, selected_station, selected_hour, selected_month, selected_day_num)
        
        if current_prediction is not None:
            # Display prediction metrics
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Delay Likelihood",
                value=f"{current_prediction['likelihood_of_delay']:.1%}",
                delta=None
            )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction interpretation
            likelihood = current_prediction['likelihood_of_delay']
            if likelihood < 0.3:
                status = "🟢 Low Risk"
                color = "green"
            elif likelihood < 0.7:
                status = "🟡 Moderate Risk"
                color = "orange"
            else:
                status = "🔴 High Risk"
                color = "red"
            
            # Get month and day names
            month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                          'July', 'August', 'September', 'October', 'November', 'December']
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            month_name = month_names[selected_month - 1] if selected_month else "All months"
            day_name = day_names[selected_day_num] if selected_day_num is not None else "All days"
            
            st.markdown(f"""
            <div class="prediction-box">
                <h4>Prediction Details</h4>
                <p><strong>Time:</strong> {selected_hour:02d}:00</p>
                <p><strong>Date:</strong> {day_name}, {month_name}</p>
                <p><strong>Status:</strong> <span style="color: {color}">{status}</span></p>
                <p><strong>Station:</strong> {selected_station}</p>
                <p><strong>Severity:</strong> {current_prediction['delay_severity']}</p>
                <p><strong>Delay Length:</strong> {current_prediction['delay_length']:.1f} minutes</p>
                <p><strong>Coordinates:</strong> {current_prediction['latitude']:.4f}, {current_prediction['longitude']:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Station timeline
        st.subheader("📈 Daily Timeline")
        timeline_fig = create_delay_timeline(predictions_df, selected_station)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Additional analysis section
    st.markdown("---")
    st.subheader("🔍 Analysis & Insights")
    
    col3, col4, col5, col6 = st.columns(4)
    
    with col3:
        # Overall statistics
        st.markdown("**Overall Statistics**")
        avg_likelihood = predictions_df['likelihood_of_delay'].mean()
        max_likelihood = predictions_df['likelihood_of_delay'].max()
        min_likelihood = predictions_df['likelihood_of_delay'].min()
        
        st.metric("Average Delay Likelihood", f"{avg_likelihood:.1%}")
        st.metric("Highest Risk", f"{max_likelihood:.1%}")
        st.metric("Lowest Risk", f"{min_likelihood:.1%}")
    
    with col4:
        # Temporal analysis
        st.markdown("**Temporal Analysis**")
        
        # Weekend vs Weekday
        weekend_stats = predictions_df.groupby('is_weekend')['likelihood_of_delay'].mean()
        weekend_risk = weekend_stats.get(1, 0)
        weekday_risk = weekend_stats.get(0, 0)
        
        st.metric("Weekend Risk", f"{weekend_risk:.1%}")
        st.metric("Weekday Risk", f"{weekday_risk:.1%}")
        
        # Seasonal analysis
        monthly_avg = predictions_df.groupby('month')['likelihood_of_delay'].mean()
        worst_month = monthly_avg.idxmax()
        best_month = monthly_avg.idxmin()
        
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        st.metric("Worst Month", month_names[worst_month-1])
        st.metric("Best Month", month_names[best_month-1])
    
    with col5:
        # Peak hours analysis
        st.markdown("**Peak Hours Analysis**")
        hourly_avg = predictions_df.groupby('hour')['likelihood_of_delay'].mean().reset_index()
        peak_hour = hourly_avg.loc[hourly_avg['likelihood_of_delay'].idxmax()]
        
        st.metric("Peak Risk Hour", f"{int(peak_hour['hour']):02d}:00")
        st.metric("Peak Risk Level", f"{peak_hour['likelihood_of_delay']:.1%}")
        
        # Day of week analysis
        day_avg = predictions_df.groupby('day_of_week')['likelihood_of_delay'].mean()
        worst_day = day_avg.idxmax()
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        st.metric("Worst Day", day_names[worst_day])
    
    with col6:
        # Station rankings
        st.markdown("**Station Rankings**")
        station_avg = predictions_df.groupby('station')['likelihood_of_delay'].mean().sort_values(ascending=False)
        
        st.write("**Highest Risk Stations:**")
        for i, (station, risk) in enumerate(station_avg.head(3).items()):
            st.write(f"{i+1}. {station}: {risk:.1%}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit • TTC Subway Delay Prediction System</p>
        <p>Data based on neural network predictions from historical TTC data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
