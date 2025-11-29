Here's the complete implementation of the VolSqueezeReversion strategy for backtesting.py:

```python
import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
import talib

# Data preprocessing
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"

# Load and clean data
data = pd.read_csv(data_path)
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Validate required columns
required_cols = ['open', 'high', 'low', 'close', 'volume', 'funding_rate']
for col in required_cols:
    if col not in data.columns:
        raise ValueError(f"🌑 Missing required column: {col}")

# Proper column mapping
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'funding_rate': 'Funding_Rate'
}, inplace=True)

# Convert datetime
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolSqueezeReversion(Strategy):
    def init(self):
        # 🌙 Moon Dev Indicators 🌙
        # Bollinger Bands Components
        self.middle = self.I(talib.SMA, self.data.Close, 20, name='BB_middle')
        self.std = self.I(talib.STDDEV, self.data.Close, 20, name='BB_std')
        self.upper = self.I(lambda data: data['BB_middle'] + 2*data['BB_std'], name='BB_upper')
        self.lower = self.I(lambda data: data['BB_middle'] - 2*data['BB_std'], name='BB_lower')
        
        # Volatility Contraction
        self.bb_width = self.I(lambda data: (data['BB_upper'] - data['BB_lower'])/data['BB_middle'], name='BB_width')
        self.bb_pct = self.I(lambda data: data['BB_width'].rolling(2880).quantile(0.2), name='BB_pct')
        
        # Funding Rate Dynamics
        self.fund_sma7 = self.I(talib.SMA, self.data.Funding_Rate, 672, name='fund_sma7')
        self.fund_sma30 = self.I(talib.SMA, self.data.Funding_Rate, 2880, name='fund_sma30')
        self.fund_std30 = self.I(talib.STDDEV, self.data.Funding_Rate, 2880, name='fund_std30')
        
        # Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')
        
        self.entry_bar = None

    def next(self):
        # 🌕 Moon Dev Core Logic 🌕
        if len(self.data) < 2880:  # Wait for full data
            return
        
        # Current values
        price = self.data.Close[-1]
        upper = self.upper[-1]
        lower = self.lower[-1]
        bb_width = self.bb_width[-1]
        bb_pct = self.bb_pct[-1]
        fund = self.data.Funding_Rate[-1]
        fund_sma7 = self.fund_sma7[-1]
        fund_sma30 = self.fund_sma30[-1]
        fund_std30 = self.fund_std30[-1]
        atr = self.atr[-1]
        
        # 🚀 Entry Conditions 🚀
        if not self.position:
            # Long Setup
            long_cond = (price > upper) and (bb_width <= bb_pct) and (fund <= 2*fund_sma7) and (fund_sma7 < 0)
            
            # Short Setup
            short_cond = (price < lower) and (bb_width <= bb_pct) and (fund >= 2*fund_sma7) and (fund_sma7 > 0)
            
            if long_cond or short_cond:
                # 🌙 Risk Management Calculation 🌙
                risk_pct = 0.01  # 1% risk
                risk_amount = self.equity * risk_pct
                sl_dist = 1.5 * atr
                size = int(round(risk_amount / sl_dist))
                
                if size > 0: