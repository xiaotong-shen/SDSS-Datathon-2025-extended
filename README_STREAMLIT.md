# TTC Subway Delay Predictor - Streamlit Version

A modern, interactive web application for predicting TTC subway delays using machine learning and neural networks.

## 🚇 Features

- **Interactive Map Visualization**: Real-time delay predictions displayed on an interactive map
- **Station-Specific Analysis**: Detailed predictions for individual stations
- **Time-based Filtering**: View predictions for any hour of the day
- **Risk Assessment**: Color-coded risk levels (Low, Moderate, High)
- **Daily Timeline Charts**: Track delay patterns throughout the day
- **Statistical Insights**: Overall system performance and peak hour analysis

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd SDSS-Datathon-2025-extended
   ```

2. **Set up virtual environment** (recommended):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements_streamlit.txt
   ```

## 🚀 Running the Application

1. **Activate your virtual environment** (if using one):
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Run the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

3. **Open your browser** and navigate to the URL shown in the terminal (typically `http://localhost:8501`)

## 📊 Data Sources

The application uses pre-computed predictions from a neural network model trained on historical TTC data:

- **Enriched Predictions**: `src/routes/resources/enriched_predictions.csv`
- **Station Data**: `src/routes/resources/station_data.json`

## 🎯 How to Use

### Main Interface
1. **Time Selection**: Use the sidebar slider to select the hour of day (0-23)
2. **Station Selection**: Choose a specific station from the dropdown menu
3. **Map Interaction**: Hover over stations on the map to see detailed information

### Understanding the Visualizations

#### Interactive Map
- **Circle Size**: Represents likelihood of delay (larger = higher likelihood)
- **Circle Color**: Color-coded risk levels (Red = High, Green = Low)
- **Hover Information**: Detailed station information and delay predictions

#### Station Timeline
- **Line Chart**: Shows delay likelihood throughout the day for selected station
- **Peak Hours**: Easily identify high-risk time periods
- **Pattern Analysis**: Understand daily delay patterns

#### Analysis Dashboard
- **Overall Statistics**: System-wide performance metrics
- **Peak Hours**: Identify the most problematic time periods
- **Station Rankings**: Compare risk levels across different stations

## 🔧 Technical Details

### Architecture
- **Frontend**: Streamlit web framework
- **Visualization**: Plotly for interactive charts and maps
- **Data Processing**: Pandas for data manipulation
- **Caching**: Streamlit's built-in caching for performance

### Key Components
- `app.py`: Main Streamlit application
- `load_data()`: Data loading and caching function
- `create_delay_map()`: Interactive map generation
- `create_delay_timeline()`: Timeline chart generation
- `get_delay_prediction()`: Individual prediction retrieval

### Performance Features
- **Data Caching**: Automatic caching of loaded data for faster subsequent loads
- **Responsive Design**: Adapts to different screen sizes
- **Real-time Updates**: Instant updates when changing parameters

## 📈 Model Information

The delay predictions are generated using a neural network model with the following features:

### Input Features
- **Temporal Features**: Time of day, day of week, month, seasonal patterns
- **Station Features**: Station-specific characteristics and historical patterns
- **Operational Features**: Rush hour indicators, weekend patterns, holiday seasons

### Output Predictions
- **Delay Likelihood**: Probability of experiencing a delay (0-1)
- **Risk Classification**: Low, Moderate, or High risk categories

## 🎨 Customization

### Styling
The application uses custom CSS for enhanced visual appeal:
- Modern color scheme with TTC branding
- Responsive design elements
- Custom metric cards and prediction boxes

### Adding New Features
To extend the application:
1. Add new visualization functions
2. Update the main interface in `app.py`
3. Include additional data sources as needed

## 🐛 Troubleshooting

### Common Issues

1. **Data Loading Errors**:
   - Ensure data files exist in the correct paths
   - Check file permissions
   - Verify CSV format compatibility

2. **Map Not Displaying**:
   - Check internet connection (required for map tiles)
   - Verify Plotly installation
   - Clear browser cache if needed

3. **Performance Issues**:
   - Use virtual environment for isolated dependencies
   - Ensure sufficient system memory
   - Consider data caching for large datasets

### Getting Help
- Check the console output for error messages
- Verify all dependencies are installed correctly
- Ensure Python version compatibility (3.8+ recommended)

## 📝 License

This project is part of the SDSS Datathon 2025 Extended Challenge.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For questions or issues:
- Check the troubleshooting section above
- Review the neural network implementation notes in `python notebooks/neuralnet_README.md`
- Contact the development team

---

**Built with ❤️ for the TTC and Toronto commuters**
