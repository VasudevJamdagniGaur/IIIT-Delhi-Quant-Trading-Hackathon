# Trading Strategy Submission - Team vasudevjamdagnigaur

**Team Members**: Vasudev Jamdagni Gaur

**Submission Date**: [Date]

---

## Team Name & Members

- **Team Name**: vasudevjamdagnigaur
- **Member 1**: Vasudev Jamdagni Gaur - [Role/Contribution]
- **Contact**: [Primary contact email]

---

## List of Strategies Submitted

### 1. Baseline SMA Crossover Strategy
- **File**: `strategy_vasudevjamdagnigaur_baseline.py`
- **Type**: Rule-based trend following
- **Description**: Simple Moving Average crossover strategy with adaptive window sizing based on data characteristics and volatility. Uses short-term (10-30 periods) and long-term (60-180 periods) SMAs to generate buy/sell signals when crossovers occur with sufficient trend strength.

### 2. Machine Learning RandomForest Strategy  
- **File**: `strategy_vasudevjamdagnigaur_ml.py`
- **Weights**: `weights_vasudevjamdagnigaur_ml.pkl`
- **Type**: Supervised machine learning
- **Description**: RandomForest classifier trained on engineered technical features to predict next-minute price direction. Uses probability thresholds and position tracking to generate trading signals with confidence filtering.

---

## Features / Indicators Used

### Baseline Strategy Features:
- **Simple Moving Averages**: Short-term (adaptive 5-30 periods) and long-term (adaptive 30-180 periods)
- **Trend Strength Filter**: Minimum price change threshold to confirm trend validity
- **Adaptive Window Sizing**: Windows adjust based on dataset length and historical volatility
- **Position Tracking**: Maintains current position state (cash/long) for realistic signal generation

### ML Strategy Features:
- **Price-based**: Multiple timeframe returns (1, 5, 15 minutes), rolling statistics, price z-scores
- **Technical Indicators**: RSI (14, 30 periods), Bollinger Bands (20, 40 periods), MACD-like features
- **Volume Features**: Volume ratios, volume-price trend relationships
- **Momentum**: Rate of change, acceleration, momentum across multiple timeframes
- **Volatility**: High-low spreads, rolling volatility measures
- **Position Features**: Price position within recent high-low ranges (10, 30 periods)

**Total Features**: 30+ engineered features with comprehensive technical analysis coverage

---

## Model / Logic

### Baseline Strategy Logic:
1. **Adaptive Parameter Selection**: Window sizes adjust based on:
   - Dataset length (longer data → longer windows for stability)
   - Historical volatility (higher volatility → shorter windows for responsiveness)
2. **Signal Generation**: 
   - BUY when short SMA crosses above long SMA with trend strength confirmation
   - SELL when short SMA crosses below long SMA with trend strength confirmation
   - HOLD otherwise
3. **Trend Strength Filter**: Uses 60th percentile of historical price changes as minimum threshold

### ML Strategy Logic:
1. **Feature Engineering**: Comprehensive technical feature extraction from OHLCV data
2. **Target Creation**: Binary classification predicting next-minute price direction (up/down)
3. **Model Training**: RandomForest with 200 estimators, balanced class weights, cross-validation
4. **Threshold Optimization**: Grid search to optimize buy/sell probability thresholds on validation set
5. **Signal Generation**:
   - BUY when P(up) ≥ optimized threshold and currently in cash
   - SELL when P(up) ≤ optimized threshold and currently holding position
   - HOLD otherwise, with confidence filtering

**Model Parameters**:
- RandomForest: 200 estimators, max_depth=10, balanced class weights
- Feature scaling: StandardScaler normalization
- Validation split: 80/20 train/validation with stratification

---

## Risk / Trade Controls

### Position Management:
- **Long-only strategy**: No short selling, only BUY/SELL/HOLD signals
- **All-in/All-out**: Full position sizing on each trade (no partial positions)
- **Position tracking**: Maintains state to prevent invalid signal sequences
- **Transaction costs**: 0.1% fee per trade side built into backtesting

### Signal Validation:
- **Format enforcement**: Strict validation of signal format (BUY/SELL/HOLD uppercase only)
- **Index alignment**: Ensures signals match test data index exactly
- **No lookahead**: All features use only past and current data, no future information

### ML-Specific Controls:
- **Confidence filtering**: Minimum confidence threshold for signal generation
- **Feature consistency**: Validation of feature alignment between training and prediction
- **Fallback mechanism**: HOLD-only mode if model loading fails
- **Probability thresholds**: Optimized thresholds prevent overtrading

---

## Overfitting Prevention

### Cross-Validation & Validation:
- **Time-series split**: 70/30 train/test split respecting temporal order
- **Validation set**: 20% of training data held out for threshold optimization
- **Stratified sampling**: Balanced target classes in train/validation splits

### Model Regularization:
- **RandomForest parameters**: Limited depth (10), minimum samples per leaf (10)
- **Feature engineering**: Focus on established technical indicators, not data-mined features
- **Balanced classes**: Class weights adjusted to handle imbalanced target distribution
- **Ensemble method**: RandomForest inherently reduces overfitting through bagging

### Strategy Design:
- **Simple baseline**: Rule-based strategy provides robust benchmark
- **Parameter adaptation**: Baseline adapts to data characteristics rather than optimizing on specific periods
- **Reproducibility**: Fixed random seeds ensure consistent results
- **Conservative thresholds**: ML thresholds optimized on validation, not test data

---

## Known Limitations

### Data Dependencies:
- **Minute-level data**: Strategies designed for minute-frequency data, may not generalize to other frequencies
- **Market regime**: Performance may vary significantly across different market conditions
- **Volume requirements**: Some features assume meaningful volume data availability

### Strategy Limitations:
- **Long-only**: Cannot profit from downward price movements
- **Transaction costs**: 0.1% fees may erode profits on high-frequency trading
- **Slippage**: Backtesting assumes perfect execution at open prices

### Technical Limitations:
- **Feature stability**: ML features may become less predictive over time
- **Model complexity**: RandomForest may still overfit despite regularization
- **Threshold sensitivity**: Performance sensitive to probability threshold selection

### Implementation Constraints:
- **No internet**: Cannot access real-time data or external APIs during execution
- **Library restrictions**: Limited to specified libraries (numpy, pandas, sklearn, etc.)
- **Memory usage**: Large feature matrices may cause memory issues on very large datasets

---

## How to Run Locally

### Prerequisites:
```bash
# Install required packages
pip install -r requirements.txt

# Install testing framework
pip install pytest
```

### Quick Start:
1. **Place training data**: Copy your `train.csv` file to the `data/` folder
2. **Run notebook**: Open and execute `main.ipynb` for complete analysis
3. **Run tests**: Validate implementation with `pytest -q`

### Testing:
```bash
# Run all tests quietly
pytest -q

# Run tests with verbose output
pytest -v

# Run specific test files
pytest tests/test_predict_api.py -v
pytest tests/test_no_internet.py -v
```

### Command Line Usage:
```python
# Import strategies
from strategies.strategy_vasudevjamdagnigaur_baseline import Strategy as BaselineStrategy
from strategies.strategy_vasudevjamdagnigaur_ml import Strategy as MLStrategy

# Load your data
import pandas as pd
df = pd.read_csv('data/train.csv', index_col=0, parse_dates=True)

# Split data
train_df = df.iloc[:int(0.7*len(df))]
test_df = df.iloc[int(0.7*len(df)):]

# Run baseline strategy
baseline = BaselineStrategy()
baseline.fit(train_df)
baseline_signals = baseline.predict(test_df)

# Run ML strategy
ml_strategy = MLStrategy()
ml_strategy.fit(train_df)  # This will save weights to weights/weights_teamname_ml.pkl
ml_signals = ml_strategy.predict(test_df)

# Backtest
from backtest import backtest_minute_sharpe
baseline_results = backtest_minute_sharpe(test_df, baseline_signals)
ml_results = backtest_minute_sharpe(test_df, ml_signals)
```

### File Structure:
```
hackathon-trading/
├── strategies/
│   ├── strategy_teamname_baseline.py
│   └── strategy_teamname_ml.py
├── weights/
│   └── weights_teamname_ml.pkl
├── data/
│   └── train.csv (place your data here)
├── tests/
├── main.ipynb
├── backtest.py
├── utils.py
└── requirements.txt
```

---

## Video Demonstration Checklist

**Recording Requirements** (for submission video):

- [ ] **Show file structure**: Display all required files in correct naming format
- [ ] **Demonstrate no internet**: Show `assert_no_network_imports()` passing
- [ ] **Show reproducibility**: Run same strategy twice, show identical results
- [ ] **Display random seeds**: Show `random_state=42` in ML strategy code
- [ ] **Validate signals**: Show signal format validation passing (BUY/SELL/HOLD only)
- [ ] **Show weights saving**: Demonstrate ML model saving to `weights_teamname_ml.pkl`
- [ ] **Run backtesting**: Execute backtest showing Sharpe ratio calculation
- [ ] **Show no lookahead**: Explain feature engineering uses only past/current data
- [ ] **Demonstrate fit/predict**: Show clear separation of training and testing phases
- [ ] **Performance summary**: Display final results table with both strategies

**Code Walkthrough Points**:
- Strategy class structure and required methods
- Feature engineering process (ML strategy)
- Signal generation logic
- Position tracking implementation
- Backtesting methodology

---

## Submission Checklist

**Files to Submit**:
- [ ] `strategy_vasudevjamdagnigaur_baseline.py`
- [ ] `strategy_vasudevjamdagnigaur_ml.py` 
- [ ] `weights_vasudevjamdagnigaur_ml.pkl`
- [ ] `README_vasudevjamdagnigaur.md` (this file)

**Validation Checklist**:
- [ ] All filenames follow required pattern
- [ ] Strategy class name is exactly "Strategy"
- [ ] predict() returns pandas Series with correct format
- [ ] No internet imports anywhere in code
- [ ] Random seeds set for reproducibility
- [ ] Weights file created and loadable
- [ ] Unit tests pass (`python -m pytest tests/`)
- [ ] README contains all required sections
- [ ] Video demonstration recorded

**Submission Portal**: [Insert hackathon submission URL here]

**Deadline**: [Insert submission deadline]

---

*This README was generated as part of the hackathon trading strategy development framework. Update all placeholder values with your actual team information and results.*

