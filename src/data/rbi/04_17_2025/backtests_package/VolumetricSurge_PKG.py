I'll fix the backtesting.lib imports and ensure all indicators are properly implemented without any forbidden imports. Here's the corrected code with Moon Dev themed improvements:

```python
# 🌙 Moon Dev Backtest AI Implementation 🚀
# 🚨 STRICTLY NO backtesting.lib USAGE - PURE LUNAR POWER ONLY 🌕
import pandas as pd
from backtesting import Backtest, Strategy
import talib
import numpy as np

# ======================
# DATA PREPROCESSING 🌍
# ======================
def load_data():
    # Load lunar data from cosmic CSV 🌌
    data = pd.read_csv(
        '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv',
        parse_dates=['datetime'],
        index_col='datetime'
    )
    
    # Cleanse column names with moon gravity 🌕
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Align constellations with proper case 🌠
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    print("🌑🌒🌓🌔🌕 Data Successfully Aligned with Lunar Phases!")
    return data

# ======================
# CORE STRATEGY CLASS 🚀
# ======================
class VolumetricSurge(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    
    def init(self):
        # ======================
        # INDICATOR CALCULATIONS 🌗
        # ======================
        
        # Trend Strength Indicators 🌊
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.plus_di = self.I(talib.PLUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.minus_di = self.I(talib.MINUS_DI, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Volume Acceleration Signals 📈
        self.volume_sma5 = self.I(talib.SMA, self.data.Volume, timeperiod=5)
        self.volume_sma20 = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Bollinger Band Width Calculation 🌗
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close)
        self.bb_width_low = self.I(talib.MIN, self.bb_width, timeperiod=20)
        
        # Trend Alignment & Exit SMAs 🌙
        self.sma20 = self.I(talib.SMA, self.data.Close, timeperiod=20)
        self.sma10 = self.I(talib.SMA, self.data.Close, timeperiod=10)
        
        # Swing Low Detection for Stops 🌊
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=10)
        
        print("✨ Lunar Indicators Activated! Ready for Cosmic Analysis 🌌")

    def next(self):
        # ======================
        # ENTRY LOGIC 🚀
        # ======================
        if not self.position:
            # Trend confirmation 🌊
            adx_strong = self.adx[-1] > 25
            uptrend = self.plus_di[-1] > self.minus_di[-1]
            
            # Volume surge detection 📈
            vol_surge = (self.volume_sma5[-1] > self.volume_sma20[-1] * 1.2)
            
            # Price alignment 🌗
            price_above_sma20 = self.data.Close[-1] > self.sma20[-1]
            
            if all([adx_strong, uptrend, vol_surge, price_above_sma20]):
                # Calculate risk parameters 🌑
                sl_price = self.swing_low[-1]
                risk_per_share = self.data.Close[-1] - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round((self.equity * self.risk_percent) /