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

# Custom CSS for better styling - Dark mode friendly
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .prediction-box {
        background-color: #2d3748;
        color: #e2e8f0;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #4CAF50;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .metric-card h4, .prediction-box h4 {
        color: #4CAF50;
        margin-bottom: 0.5rem;
    }
    .metric-card p, .prediction-box p {
        color: #e2e8f0;
        margin: 0.5rem 0;
    }
    .metric-card strong, .prediction-box strong {
        color: #81c784;
    }
    .sidebar .sidebar-content {
        background-color: #1a202c;
    }
    .stSelectbox > div > div {
        background-color: #2d3748;
        color: #e2e8f0;
    }
    .stSlider > div > div > div > div {
        background-color: #4CAF50;
    }
    /* Custom progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #4CAF50 !important;
    }
    .stProgress > div > div > div > div > div {
        background-color: #4CAF50 !important;
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
    
    # Create a much more visible size mapping by scaling the likelihood values
    # The values are between 0.26-0.51, so we'll scale them to be much more visible
    # Using a much larger multiplier to create dramatic size differences
    filtered_data['scaled_size'] = (filtered_data['likelihood_of_delay'] - 0.26) * 50
    
    # Debug: Print size range for verification
    if st.checkbox("Show size debugging info", key="debug_sizes"):
        st.write(f"Original likelihood range: {filtered_data['likelihood_of_delay'].min():.3f} to {filtered_data['likelihood_of_delay'].max():.3f}")
        st.write(f"Scaled size range: {filtered_data['scaled_size'].min():.1f} to {filtered_data['scaled_size'].max():.1f}")
        st.write(f"Sample stations with sizes:")
        sample_data = filtered_data[['station', 'likelihood_of_delay', 'scaled_size']].head(5)
        st.dataframe(sample_data)
    
    fig = px.scatter_map(
        filtered_data,
        lat='latitude',
        lon='longitude',
        size='scaled_size',
        color='likelihood_of_delay',
        hover_name='station',
        hover_data=['likelihood_of_delay', 'delay_severity', 'delay_length'],
        color_continuous_scale='RdYlGn_r',  # Red to Green (red = high delay likelihood)
        size_max=60,
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
    
    # Apply all conditions using pandas filtering
    mask = True
    for condition in conditions:
        mask = mask & condition
    prediction = predictions_df[mask]
    
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
        st.markdown('<h3 style="color: #4CAF50;">📍 Interactive Delay Map</h3>', unsafe_allow_html=True)
        
        # Create and display the map
        map_fig = create_delay_map(predictions_df, selected_hour, selected_month, selected_day_num)
        if map_fig:
            st.plotly_chart(map_fig, use_container_width=True)
        
        # Map legend with better spacing and styling
        st.markdown("""
        <div style="margin-top: 5rem; margin-bottom: 0rem; padding: 1rem; background-color: #2d3748; border-radius: 0.5rem; border-left: 4px solid #4CAF50;">
            <h4 style="color: #4CAF50; margin-bottom: 1rem;">Map Legend</h4>
            <ul style="color: #e2e8f0; margin: 0; padding-left: 1.5rem;">
                <li><strong>Circle Size:</strong> Likelihood of delay (larger = higher likelihood, dramatically scaled for visibility)</li>
                <li><strong>Circle Color:</strong> Delay likelihood (Red = High, Green = Low)</li>
                <li><strong>Size Range:</strong> Small circles (20px) = Low risk (~26%), Large circles (60px) = High risk (~51%)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<h3 style="color: #4CAF50;">Station Details</h3>', unsafe_allow_html=True)
        
        # Get current prediction
        current_prediction = get_delay_prediction(predictions_df, selected_station, selected_hour, selected_month, selected_day_num)
        
        # Debug: Show prediction details
        if st.checkbox("Show prediction debugging info", key="debug_prediction"):
            st.write(f"Selected parameters: Station={selected_station}, Hour={selected_hour}, Month={selected_month}, Day={selected_day_num}")
            st.write(f"Available data for station: {len(predictions_df[predictions_df['station'] == selected_station])} rows")
            st.write(f"Available data for hour {selected_hour}: {len(predictions_df[predictions_df['hour'] == selected_hour])} rows")
            if current_prediction is not None:
                st.write("Current prediction found:", current_prediction.to_dict())
            else:
                st.write("No prediction found for these parameters")
        
        if current_prediction is not None:
            # Display prediction metrics
            # st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric(
                label="Delay Likelihood",
                value=f"{current_prediction['likelihood_of_delay']:.1%}",
                delta=None
            )
            
            # Add a visual progress bar for the likelihood with color coding
            likelihood = current_prediction['likelihood_of_delay']
            
            # Determine color based on risk level
            if likelihood < 0.3:
                progress_color = "#4CAF50"  # Green for low risk
                risk_text = "Low Risk"
                text_color = "#4CAF50"
            elif likelihood < 0.7:
                progress_color = "#FF9800"  # Orange for moderate risk
                risk_text = "Moderate Risk"
                text_color = "#FF9800"
            else:
                progress_color = "#F44336"  # Red for high risk
                risk_text = "High Risk"
                text_color = "#F44336"
            
            # Create custom progress bar with dynamic color
            st.markdown(f"""
            <div style="margin: 1rem 0;">
                <div style="background-color: #2d3748; border-radius: 0.5rem; padding: 0.5rem;">
                    <div style="background-color: {progress_color}; height: 20px; border-radius: 0.5rem; width: {likelihood * 100}%; transition: width 0.3s ease;"></div>
                </div>
                <p style="color: {text_color}; text-align: center; margin-top: 0.5rem; font-weight: bold;">{risk_text}</p>
            </div>
            """, unsafe_allow_html=True)
            
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
        st.markdown('<h4 style="color: #4CAF50;">📈 Daily Timeline</h4>', unsafe_allow_html=True)
        timeline_fig = create_delay_timeline(predictions_df, selected_station)
        if timeline_fig:
            st.plotly_chart(timeline_fig, use_container_width=True)
    
    # Additional analysis section
    # st.markdown("---")
    # st.markdown('<h3 style="color: #4CAF50;">🔍 Analysis & Insights</h3>', unsafe_allow_html=True)
    
    # col3, col4, col5, col6 = st.columns(4)
    
    # with col3:
    #     # Overall statistics
    #     st.markdown('<h4 style="color: #4CAF50;">Overall Statistics</h4>', unsafe_allow_html=True)
    #     avg_likelihood = predictions_df['likelihood_of_delay'].mean()
    #     max_likelihood = predictions_df['likelihood_of_delay'].max()
    #     min_likelihood = predictions_df['likelihood_of_delay'].min()
        
    #     st.metric("Average Delay Likelihood", f"{avg_likelihood:.1%}")
    #     st.metric("Highest Risk", f"{max_likelihood:.1%}")
    #     st.metric("Lowest Risk", f"{min_likelihood:.1%}")
    
    # with col4:
    #     # Temporal analysis
    #     st.markdown('<h4 style="color: #4CAF50;">Temporal Analysis</h4>', unsafe_allow_html=True)
        
    #     # Weekend vs Weekday
    #     weekend_stats = predictions_df.groupby('is_weekend')['likelihood_of_delay'].mean()
    #     weekend_risk = weekend_stats.get(1, 0)
    #     weekday_risk = weekend_stats.get(0, 0)
        
    #     st.metric("Weekend Risk", f"{weekend_risk:.1%}")
    #     st.metric("Weekday Risk", f"{weekday_risk:.1%}")
        
    #     # Seasonal analysis
    #     monthly_avg = predictions_df.groupby('month')['likelihood_of_delay'].mean()
    #     worst_month = monthly_avg.idxmax()
    #     best_month = monthly_avg.idxmin()
        
    #     month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
    #                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    #     st.metric("Worst Month", month_names[worst_month-1])
    #     st.metric("Best Month", month_names[best_month-1])
    
    # with col5:
    #     # Peak hours analysis
    #     st.markdown('<h4 style="color: #4CAF50;">Peak Hours Analysis</h4>', unsafe_allow_html=True)
    #     hourly_avg = predictions_df.groupby('hour')['likelihood_of_delay'].mean().reset_index()
    #     peak_hour = hourly_avg.loc[hourly_avg['likelihood_of_delay'].idxmax()]
        
    #     st.metric("Peak Risk Hour", f"{int(peak_hour['hour']):02d}:00")
    #     st.metric("Peak Risk Level", f"{peak_hour['likelihood_of_delay']:.1%}")
        
    #     # Day of week analysis
    #     day_avg = predictions_df.groupby('day_of_week')['likelihood_of_delay'].mean()
    #     worst_day = day_avg.idxmax()
    #     day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    #     st.metric("Worst Day", day_names[worst_day])
    
    # with col6:
    #     # Station rankings
    #     st.markdown('<h4 style="color: #4CAF50;">Station Rankings</h4>', unsafe_allow_html=True)
    #     station_avg = predictions_df.groupby('station')['likelihood_of_delay'].mean().sort_values(ascending=False)
        
    #     st.write("**Highest Risk Stations:**")
    #     for i, (station, risk) in enumerate(station_avg.head(3).items()):
    #         st.write(f"{i+1}. {station}: {risk:.1%}")
    
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
