Here's the fixed and properly formatted backtest code with Moon Dev themed debug prints:

```python
# 🌙 Moon Dev's VolatilityFader Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# ================ DATA PREPARATION ================
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# 🌙 Cleanse the cosmic data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# 🚀 Align celestial coordinates (column names)
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)

# ⏳ Convert temporal dimensions
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# ================ STRATEGY CLASS ================
class VolatilityFader(Strategy):
    risk_percent = 0.01        # 🌑 Risk 1% per trade
    vwap_threshold = 0.02      # 🌕 2% divergence threshold (converted to fraction)
    rsi_period = 14            # 🌓 Cosmic rhythm period
    rsi_overbought = 70        # 🌔 Overbought threshold
    rsi_oversold = 30          # 🌒 Oversold threshold
    max_holding_bars = 5       # 🌗 Max bars before moon phase change

    def init(self):
        # ========== CELESTIAL INDICATORS ==========
        # 🌌 Keltner Channel (EMA 20 + 2xATR)
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA20')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20, name='ATR20')
        self.upper_keltner = self.I(lambda: self.ema20 + 2*self.atr20, name='Upper Keltner')
        self.lower_keltner = self.I(lambda: self.ema20 - 2*self.atr20, name='Lower Keltner')

        # 🌠 VWAP Calculation (Cosmic Balance Point)
        typical_price = (self.data.High + self.data.Low + self.data.Close)/3
        cumulative_volume = self.data.Volume.cumsum()
        vwap = (typical_price * self.data.Volume).cumsum() / cumulative_volume
        self.vwap = self.I(vwap, name='VWAP')
        
        # 🌖 Volume-Weighted RSI (Moon Gravity Index)
        close = self.data.Close
        volume = self.data.Volume
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = (gain * volume).rolling(14).sum() / volume.rolling(14).sum()
        avg_loss = (loss * volume).rolling(14).sum() / volume.rolling(14).sum()
        
        rs = avg_gain / avg_loss
        self.vw_rsi = self.I(100 - (100 / (1 + rs)), name='VW_RSI')

        print("🌙✨ Moon Dev Backtest Initialized! Ready to surf volatility waves! 🚀🌌")

    def next(self):
        # 🌑 Skip if already in cosmic trade
        if self.position:
            trade = self.trades[-1]
            
            # 🌗 Long Exit Conditions
            if trade.is_long:
                if (self.vw_rsi[-1] >= self.rsi_overbought or
                    self.data.Close[-1] > self.lower_keltner[-1] or
                    len(self.data) - trade.entry_bar >= self.max_holding_bars):
                    
                    trade.close()
                    print(f"🌕 Long Exit | RSI: {self.vw_rsi[-1]:.1f} | Moon phase shift detected!")
            
            # 🌒 Short Exit Conditions
            elif trade.is_short:
                if (self.vw_rsi[-1] <= self.rsi_oversold or
                    self.data.Close[-1] < self.upper_keltner[-1] or
                    len(self.data) - trade.entry_bar >= self.max_holding_bars