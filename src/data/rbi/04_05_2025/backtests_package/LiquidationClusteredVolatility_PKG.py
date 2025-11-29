Here's the complete fixed code with all backtesting.lib imports removed and proper Moon Dev themed implementation:

```python
# 🌙 Moon Dev Backtest AI Implementation 🚀
import pandas as pd
import talib
import pandas_ta as pd_ta
from backtesting import Backtest, Strategy
import numpy as np

# =========================
# DATA PREPARATION 🌙✨
# =========================
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare data according to Moon Dev specs 🧹✨
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to backtesting.py requirements 📊
required_columns = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=required_columns, inplace=True)

# ============================
# STRATEGY IMPLEMENTATION 🌙🚀
# ============================
class LiquidationClusteredVolatility(Strategy):
    # Strategy parameters ⚙️
    bb_period = 20
    bb_dev = 2
    vwap_length = 2016  # 21 days in 15m intervals (21*24*4)
    atr_period = 14
    risk_pct = 0.03  # 3% max allocation
    imb_threshold = 0.85  # Order book imbalance threshold
    
    def init(self):
        # 🌙 Bollinger Band System 📉
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=self.bb_period, 
            nbdevup=self.bb_dev, nbdevdn=self.bb_dev
        )
        self.bb_width = self.I(
            lambda u, m, l: (u - l)/m, 
            self.bb_upper, self.bb_middle, self.bb_lower,
            name='BB_WIDTH 🌙'
        )
        
        # 🌙 VWAP System 📊
        self.vwap = self.I(
            pd_ta.vwap,
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume,
            length=self.vwap_length,
            name='VWAP 🚀'
        )
        
        # 🌙 Risk Management System ⚠️
        self.atr = self.I(
            talib.ATR,
            self.data.High,
            self.data.Low,
            self.data.Close,
            self.atr_period,
            name='ATR ⚠️'
        )
        
        # 🌙 Funding Rate Analysis 💹
        self.funding_rate = self.data['fundingrate']
        self.fund_upper = self.I(
            pd_ta.quantile,
            self.funding_rate,
            length=1008,  # 1 week in 15m intervals
            q=0.95,
            name='FU 🌙'
        )
        self.fund_lower = self.I(
            pd_ta.quantile,
            self.funding_rate,
            length=1008,
            q=0.05,
            name='FL 🌙'
        )
        
        # 🌙 Order Book Imbalance 📈
        self.imb = self.data['imb']
        self.vol_spike = self.I(
            talib.MAX,
            self.bb_width,
            timeperiod=24*4,  # 24 hours in 15m intervals
            name='VOL SPIKE 🚨'
        )

    def next(self):
        current_price = self.data.Close[-1]
        
        # 🌙 Exit Conditions ✨
        if self.position:
            # VWAP Reversion Exit - replaced crossover with manual check
            if (self.vwap[-2] < current_price and self.vwap[-1] > current_price):
                self.position.close()
                print(f"🌙 Moon Dev Exit: VWAP Reversion at {current_price:.2f} ✨")
            
            # Trailing ATR Stop
            if self.position.is_short:
                trail_stop = self.data.High[-1] + 1.5*self.atr[-1]
                self.position.stop = trail_stop
                
        # 🌙 Entry Conditions 🚀
        else:
            # Replaced all crossover conditions with manual checks