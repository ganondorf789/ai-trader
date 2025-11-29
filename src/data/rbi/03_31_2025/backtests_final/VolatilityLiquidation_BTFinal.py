Here's the debugged code with proper formatting, imports, and position sizing while maintaining the original strategy logic:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import numpy as np
import talib
import pandas_ta as ta

# Load and preprocess data
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names and drop unnamed
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Ensure proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# Generate mock funding rate and liquidation clusters
np.random.seed(42)  # Fixed typo in seed()
data['funding_rate'] = np.random.normal(0.0001, 0.0005, len(data))
data['liquidation_cluster'] = np.random.random(len(data)) < 0.1

class VolatilityLiquidation(Strategy):
    def init(self):
        # Funding rate indicators 🌙
        self.funding_rate = self.I(lambda: self.data.df['funding_rate'], name='Funding Rate')
        self.funding_7d_mean = self.I(talib.SMA, self.funding_rate, timeperiod=7*24*4)
        self.funding_7d_std = self.I(talib.STDDEV, self.funding_rate, timeperiod=7*24*4)
        self.funding_3sigma = self.I(lambda mean, std: mean + 3*std, self.funding_7d_mean, self.funding_7d_std)
        
        # Volatility percentile rank ✨
        self.returns = self.I(ta.ROC, self.data.Close, length=1)
        self.volatility = self.I(talib.STDDEV, self.returns, timeperiod=50*24*4)
        
        def percentile_rank(arr):
            return (np.sum(arr[-1] >= arr[-50*24*4:]) / (50*24*4)) * 100
        self.volatility_rank = self.I(percentile_rank, self.volatility)
        
        # Dynamic Bollinger Bands 🚀
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.std20 = self.I(talib.STDDEV, self.data.Close, timeperiod=20)
        self.multiplier = self.I(lambda vr: 2.0 - (vr/100)*0.5, self.volatility_rank)
        self.upper_band = self.I(lambda sma, std, m: sma + m*std, self.sma20, self.std20, self.multiplier)
        self.lower_band = self.I(lambda sma, std, m: sma - m*std, self.sma20, self.std20, self.multiplier)
        
        # Liquidation clusters 🌌
        self.liquidation_cluster = self.I(lambda: self.data.df['liquidation_cluster'])
        
        print("🌙 Moon Dev Strategy Initialized! Ready for Launch! 🚀")

    def next(self):
        if len(self.data) < 100*24*4:  # Ensure enough data
            return
            
        current_close = self.data.Close[-1]
        current_funding = self.funding_rate[-1]
        funding_3sigma = self.funding_3sigma[-1]
        upper_band = self.upper_band[-1]
        has_cluster = self.liquidation_cluster[-1]
        
        # Entry conditions 🌕
        if (current_funding > funding_3sigma and
            current_close > upper_band and
            has_cluster and
            not self.position):
            
            # Risk management 🌗
            risk_percent = 0.01  # Fixed as fraction (1% equity risk)
            stop_distance = current_close * 0.02  # 2% stop loss
            risk_amount = self.equity * risk_percent
            position_size = int(round(risk_amount / stop_distance))
            
            # Execute short 🌑
            self.sell(size=position_size, 
                     sl=current_close * 1.02,  # Fixed as price level
                     tp=current_close * 0.98)  # Fixed as