# TTC Subway Delay Prediction Neural Network - Implementation Notes

## Overview
This document outlines the key design decisions, implementation choices, and learning outcomes from developing the TTC subway delay prediction neural network system.

---

## 1. Categorical vs Continuous Temporal Features

### **Key Decision: Categorical Approach**
**Question**: How do categorical patterns (weekends, rush hours, seasonal changes) impact TTC delays?

**Implementation Choice**: Categorical features instead of continuous datetime

### **Why Categorical Approach:**
- **Avoids artificial relationships**: Continuous datetime creates false "closeness" between December 31st and January 1st
- **Fundamental categorical patterns**: TTC delays follow distinct categorical patterns (Monday rush ≠ Friday rush)
- **Better neural network learning**: Categorical features are easier for neural networks to learn and interpret
- **Actionable insights**: "Weekends have 30% fewer delays" is more actionable than complex continuous relationships
- **Domain-specific patterns**: Captures real TTC operational realities

### **Features Implemented:**
```python
# Temporal Features (10 total)
- time_norm: Normalized time within TTC operating hours (0.0-1.0)
- month: Month of year (1-12) for seasonal patterns
- day_of_week: Day of week (0=Monday, 6=Sunday) for weekly patterns  
- is_weekend: Binary flag for weekend service patterns
- is_morning_rush: Binary flag for 7-9 AM rush hour
- is_evening_rush: Binary flag for 4-6 PM rush hour
- is_holiday_season: Binary flag for December/January patterns
- is_back_to_school: Binary flag for September patterns
```

---

## 2. Comprehensive Prediction Dataset

### **Full Coverage Implementation:**
- **All 12 months**: January through December
- **All 7 days**: Monday through Sunday
- **All 50 time points**: 6:00 AM to 1:30 AM (TTC operating hours)
- **All stations**: Every TTC subway station

### **Why Full Coverage:**
- **User expectations**: Users want predictions for any date/time they choose
- **No missing data**: Frontend can always find a prediction for user queries
- **Complete pattern analysis**: Captures all seasonal and weekly variations
- **Production ready**: Handles real-world user scenarios

### **Prediction Structure:**
```python
{
    'station': int,              # Station ID
    'station_name': str,         # "Bloor-Yonge Station"
    'month': int,                # 1-12
    'day_of_week': int,          # 0-6 (Mon-Sun)
    'is_weekend': int,           # 0 or 1
    'time_norm': float,          # 0.0-1.0
    'time_hhmm': str,            # "06:00" to "01:30"
    'delay_severity': str,       # "Minimal", "Minor", "Moderate", "Severe"
    'delay_length': float        # Minutes
}
```

---

## 3. Frontend Integration Design

### **Prediction Dataset as Reference:**
The prediction dataset serves as the **lookup table** for the frontend:

```
User Input: "Bloor-Yonge Station, February 15th, 8:30 AM"
    ↓
Frontend Query: month=2, day_of_week=2, time_norm=0.13
    ↓
Dataset Lookup: Find matching prediction
    ↓
Response: "Expected 8.5 minute moderate delay"
```

### **Key Benefits:**
- **Fast response**: No real-time model inference needed
- **Consistent results**: Same query always returns same prediction
- **Scalable**: Can handle thousands of concurrent users
- **Offline capable**: Predictions work without model server

---

## 4. Neural Network Architecture Changes

### **Model Enhancements:**
- **Input dimensions**: 3 → 10 temporal features
- **Variable naming**: `time_*` → `temporal_*` for clarity
- **Feature engineering**: Rich temporal context for better predictions

### **Architecture Details:**
```python
class MultiOutputModel(nn.Module):
    def __init__(self, temporal_features=10):  # Was 3
        # Input: 10 temporal + 16 station embedding = 26 features
        input_dim = temporal_features + embedding_dim
```

---

## 5. Critical Bug Fixes

### **Issues Resolved:**
1. **Variable consistency**: Fixed `time_batch` → `temporal_batch` throughout code
2. **Data type mismatch**: Added `.astype(np.float32)` to prevent Double/Float errors
3. **Indentation error**: Fixed prediction loop to process each time point individually
4. **Missing time variation**: Ensured predictions vary by actual time values

### **Impact:**
- **Correct timestamps**: Now shows 6:00, 6:30, 7:00, etc. instead of all 01:30
- **Varied predictions**: Different delay times for different times of day
- **Station-specific patterns**: Each station shows unique delay characteristics

---

## 6. Learning Outcomes

### **Technical Insights:**
- **Feature engineering matters**: Categorical features often outperform continuous for domain-specific problems
- **Data type consistency**: PyTorch is sensitive to dtype mismatches
- **Loop structure**: Indentation errors can cause subtle but critical bugs
- **Memory vs functionality**: Full coverage datasets are worth the memory cost for production systems

### **Domain Insights:**
- **TTC patterns**: Weekdays vs weekends, rush hours, seasonal effects are crucial
- **User needs**: Complete coverage is essential for real-world applications
- **Operational reality**: Transit systems follow categorical, not continuous, patterns

### **Product Insights:**
- **User expectations**: Users want predictions for any date/time, not just samples
- **Actionable data**: Categorical features provide clearer insights for operators
- **Scalability**: Pre-computed predictions enable fast, scalable frontend responses

---

## 7. File Format Changes

### **From Jupyter Notebook to Python Script:**
- **Better version control**: Easier to track changes in .py files
- **Production deployment**: Python scripts are easier to deploy
- **Code organization**: Cleaner structure without cell outputs
- **Reproducibility**: More reliable execution across environments

---

## 8. Future Enhancements

### **Potential Improvements:**
- **Weather integration**: Add weather data as additional features
- **Event data**: Include major events, construction, holidays
- **Real-time updates**: Periodic model retraining with new data
- **Confidence intervals**: Add uncertainty estimates to predictions
- **Multi-line support**: Extend to bus and streetcar predictions

---

## 9. Performance Metrics

### **Model Performance:**
- **Multi-task learning**: Simultaneous classification (severity) and regression (length)
- **Balanced loss**: 60% severity classification, 40% delay length regression
- **Early stopping**: Prevents overfitting with patience-based stopping
- **Learning rate scheduling**: Adaptive learning rate for better convergence

### **System Performance:**
- **Prediction coverage**: 100% of possible user queries
- **Response time**: Instant lookup from pre-computed dataset
- **Scalability**: Handles all TTC stations and temporal combinations
- **Reliability**: Robust error handling and data validation 