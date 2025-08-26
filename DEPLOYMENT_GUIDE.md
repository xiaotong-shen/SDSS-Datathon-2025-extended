# Deployment Guide - TTC Subway Delay Predictor

This guide covers various deployment options for the Streamlit-based TTC Subway Delay Predictor.

## 🏠 Local Deployment

### Quick Start
1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd SDSS-Datathon-2025-extended
   python setup_streamlit.py
   ```

2. **Run locally**:
   ```bash
   streamlit run app.py
   ```

3. **Access the app** at `http://localhost:8501`

### Production Local Setup
For production-like local deployment:

1. **Create production requirements**:
   ```bash
   pip freeze > requirements_production.txt
   ```

2. **Run with production settings**:
   ```bash
   streamlit run app.py --server.port 8501 --server.address 0.0.0.0
   ```

## ☁️ Cloud Deployment Options

### 1. Streamlit Cloud (Recommended)

**Pros**: Free tier, easy deployment, automatic updates
**Cons**: Limited resources on free tier

#### Setup Steps:
1. **Push to GitHub**: Ensure your code is in a public GitHub repository
2. **Connect to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
3. **Configure deployment**:
   - Repository: `your-username/your-repo`
   - Branch: `main`
   - Main file path: `app.py`
4. **Deploy**: Click "Deploy!"

#### Configuration:
Create `.streamlit/config.toml` for custom settings:
```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
```

### 2. Heroku

**Pros**: Good free tier, easy scaling
**Cons**: Requires credit card for verification

#### Setup Steps:
1. **Create Procfile**:
   ```
   web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. **Create runtime.txt**:
   ```
   python-3.9.18
   ```

3. **Deploy**:
   ```bash
   heroku create your-app-name
   git push heroku main
   ```

### 3. Google Cloud Platform (GCP)

**Pros**: Scalable, good integration with other Google services
**Cons**: More complex setup, costs money

#### Setup Steps:
1. **Create Dockerfile**:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   COPY requirements_streamlit.txt .
   RUN pip install -r requirements_streamlit.txt
   
   COPY . .
   EXPOSE 8501
   
   CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Deploy to Cloud Run**:
   ```bash
   gcloud builds submit --tag gcr.io/PROJECT_ID/ttc-delay-predictor
   gcloud run deploy --image gcr.io/PROJECT_ID/ttc-delay-predictor --platform managed
   ```

### 4. AWS

**Pros**: Highly scalable, many services
**Cons**: Complex setup, costs money

#### Setup Steps:
1. **Create Dockerfile** (same as GCP)
2. **Deploy to ECS or App Runner**:
   ```bash
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
   docker build -t ttc-delay-predictor .
   docker tag ttc-delay-predictor:latest ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ttc-delay-predictor:latest
   docker push ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/ttc-delay-predictor:latest
   ```

## 🔧 Environment Configuration

### Environment Variables
Create `.env` file for local development:
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
```

### Production Settings
For production deployments, consider these settings:

```python
# In app.py
import os

# Production settings
if os.getenv('ENVIRONMENT') == 'production':
    st.set_page_config(
        page_title="TTC Subway Delay Predictor",
        page_icon="🚇",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Disable debug features
    st.set_option('deprecation.showPyplotGlobalUse', False)
```

## 📊 Data Management

### Static Data Deployment
For cloud deployments, ensure data files are included:

1. **Verify data paths** in `app.py`:
   ```python
   predictions_df = pd.read_csv("src/routes/resources/enriched_predictions.csv")
   ```

2. **Include in repository**:
   ```bash
   git add src/routes/resources/
   git commit -m "Add prediction data files"
   ```

### Dynamic Data Updates
For real-time data updates:

1. **Set up data pipeline**:
   ```python
   # In app.py
   @st.cache_data(ttl=3600)  # Cache for 1 hour
   def load_data():
       # Load from external source or API
       pass
   ```

2. **Configure data refresh**:
   ```python
   # Auto-refresh every hour
   if st.button("Refresh Data"):
       st.cache_data.clear()
   ```

## 🔒 Security Considerations

### Production Security
1. **HTTPS**: Always use HTTPS in production
2. **Authentication**: Consider adding authentication for sensitive data
3. **Rate Limiting**: Implement rate limiting for API calls
4. **Input Validation**: Validate all user inputs

### Example Security Configuration
```python
# In app.py
import streamlit as st

# Security headers
st.markdown("""
<meta http-equiv="Content-Security-Policy" content="default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob:;">
""", unsafe_allow_html=True)

# Input validation
def validate_input(user_input):
    # Add validation logic
    return True
```

## 📈 Performance Optimization

### Caching Strategies
```python
# Cache expensive computations
@st.cache_data
def expensive_calculation(data):
    # Heavy computation
    return result

# Cache model predictions
@st.cache_data(ttl=3600)
def load_predictions():
    return pd.read_csv("predictions.csv")
```

### Memory Management
```python
# Clear cache when needed
if st.button("Clear Cache"):
    st.cache_data.clear()

# Optimize data loading
def load_optimized_data():
    # Load only necessary columns
    return pd.read_csv("data.csv", usecols=['needed', 'columns'])
```

## 🚀 Monitoring and Logging

### Basic Logging
```python
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log important events
logger.info("Application started")
logger.error("Error occurred: %s", error_message)
```

### Health Checks
```python
# Add health check endpoint
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }
```

## 📝 Deployment Checklist

Before deploying to production:

- [ ] All dependencies are in `requirements_streamlit.txt`
- [ ] Data files are included and accessible
- [ ] Environment variables are configured
- [ ] Security settings are applied
- [ ] Performance optimizations are implemented
- [ ] Monitoring and logging are set up
- [ ] Error handling is comprehensive
- [ ] Documentation is updated

## 🆘 Troubleshooting

### Common Deployment Issues

1. **Port conflicts**:
   ```bash
   # Use different port
   streamlit run app.py --server.port 8502
   ```

2. **Memory issues**:
   ```python
   # Optimize memory usage
   import gc
   gc.collect()
   ```

3. **Data loading errors**:
   ```python
   # Add error handling
   try:
       data = load_data()
   except Exception as e:
       st.error(f"Failed to load data: {e}")
   ```

### Getting Help
- Check Streamlit documentation: [docs.streamlit.io](https://docs.streamlit.io)
- Review deployment platform documentation
- Check application logs for error messages

---

**Happy Deploying! 🚀**
