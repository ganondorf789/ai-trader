Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed debug prints added:

```python
# 🌙 Moon Dev's Volatility Band Reversal Backtest 🌙
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# ========== DATA PREPROCESSING ==========
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean columns like a Moon Dev wizard 🧙
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Magic column renaming spell ✨
column_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}
data.rename(columns=column_mapping, inplace=True)

# ========== STRATEGY IMPLEMENTATION ==========
class VolatilityBandReversal(Strategy):
    def init(self):
        # 🌙 VIX/VVIX Ratio Calculation
        self.vix_vvix_ratio = self.I(lambda x: x['vix']/x['vvix'], name='VIX/VVIX')
        
        # 📈 Bollinger Bands with TA-Lib Moon Power
        def bb_upper(series):
            upper, _, _ = talib.BBANDS(series, timeperiod=30, nbdevup=2)
            return upper
        self.bb_upper = self.I(bb_upper, self.vix_vvix_ratio, name='BB Upper')
        
        def bb_lower(series):
            _, _, lower = talib.BBANDS(series, timeperiod=30, nbdevdn=2)
            return lower
        self.bb_lower = self.I(bb_lower, self.vix_vvix_ratio, name='BB Lower')
        
        # 📊 Volume Analysis with Cosmic Precision
        self.spx_volume_ma10 = self.I(talib.SMA, self.data['spx_volume'], timeperiod=10, name='SPX Vol MA10')
        self.ratio_sma30 = self.I(talib.SMA, self.vix_vvix_ratio, timeperiod=30, name='Ratio SMA30')
        
        print("🌌 Moon Dev Strategy Activated! Ready for cosmic market patterns 🌠")

    def next(self):
        # 🛑 Ensure enough data for calculations
        if len(self.data) < 30:
            return
            
        current_ratio = self.vix_vvix_ratio[-1]
        current_upper = self.bb_upper[-1]
        current_lower = self.bb_lower[-1]
        current_spx_vol = self.data['spx_volume'][-1]
        current_vol_ma = self.spx_volume_ma10[-1]

        # 🚀 Long Entry: Volatility Spike Reversal
        if (current_ratio > current_upper and 
            current_spx_vol > 1.2 * current_vol_ma and 
            not self.position.is_long):
            
            risk_amount = self.equity * 0.02  # 2% risk
            entry_price = self.data.Open[-1]
            stop_price = entry_price * 0.95  # 5% stop loss
            
            if entry_price > stop_price > 0:
                position_size = int(round(risk_amount / (entry_price - stop_price)))
                self.buy(size=position_size, sl=stop_price)
                print(f"🌕 LONG ENTRY! Size: {position_size} @ {entry_price:.2f} | Cosmic Bullish Alignment Detected 🌙")

        # 🌑 Short Entry: Volatility Collapse
        elif (current_ratio < current_lower and 
              current_spx_vol < 0.8 * current_vol_ma and 
              not self.position.is_short):
            
            risk_amount = self.equity * 0.02  # 2% risk
            entry_price = self.data.Open[-1]
            stop_price = entry_price * 1.05  # 5% stop loss
            
            if stop_price > entry_price > 0:
                position_size = int(round(risk_amount / (stop_price - entry_price)))
                self.sell(size=position_size, sl=stop_price)
                print(f"🌑 SHORT ENTRY! Size: {position_size} @ {entry_price:.2f} | Bearish Quantum Shift Detected 🌙")

        # 💫 Exit Conditions
        current_sma