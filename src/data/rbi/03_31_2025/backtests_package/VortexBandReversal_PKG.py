I'll fix the code by removing all `backtesting.lib` imports and replacing the crossover function with the proper implementation. Here's the corrected version with Moon Dev themed improvements:

```python
# 🌙 Moon Dev's VortexBand Reversal Backtest 🌙
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# 🚀 Data Preparation Magic 🪄
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare cosmic data 🌌
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})
data['datetime'] = pd.to_datetime(data['datetime'])
data = data.set_index('datetime')

class VortexBandStrategy(Strategy):
    def init(self):
        # 🌗 Cosmic Indicator Configuration
        self.vi_length = 14
        self.bb_length = 20
        self.sma_period = 20
        self.atr_length = 14
        
        # 🌪️ Vortex Indicator Calculations
        vi_plus, vi_minus = ta.vi(self.data.High, self.data.Low, self.data.Close, length=self.vi_length)
        self.vi_plus = self.I(lambda: vi_plus, name='VI+')
        self.vi_minus = self.I(lambda: vi_minus, name='VI-')
        
        # 📉 Bollinger Bands from TA-Lib
        self.upper_bb, self.middle_bb, self.lower_bb = self.I(
            talib.BBANDS, self.data.Close, 
            timeperiod=self.bb_length, 
            nbdevup=2, nbdevdn=2, 
            matype=0,
            name=['Upper BB', 'Middle BB', 'Lower BB']
        )
        
        # 📊 Volume Analysis
        self.volume_sma = self.I(talib.SMA, self.data.Volume, self.bb_length, name='Volume SMA')
        
        # 🌙 Core SMA for Exit Signals
        self.sma20 = self.I(talib.SMA, self.data.Close, self.sma_period, name='SMA20')
        
        # 🛡️ ATR for Risk Management
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, self.atr_length, name='ATR')

    def next(self):
        # 🚦 Wait for full indicator formation
        if len(self.data.Close) < max(self.vi_length, self.bb_length, self.atr_length) + 1:
            return

        # 🌌 Entry Conditions Check
        if not self.position:
            # 🔍 Multi-dimensional Signal Verification
            price_above_bb = self.data.Close[-1] > self.upper_bb[-1]
            volume_below_sma = self.data.Volume[-1] < self.volume_sma[-1]
            vi_bearish_cross = (self.vi_plus[-2] > self.vi_minus[-2]) and (self.vi_plus[-1] < self.vi_minus[-1])
            
            if price_above_bb and volume_below_sma and vi_bearish_cross:
                # 💰 Moon-Sized Risk Management
                atr_value = self.atr[-1]
                entry_price = self.data.Close[-1]
                stop_loss = entry_price + 1.5 * atr_value
                risk_per_share = stop_loss - entry_price
                
                if risk_per_share <= 0:
                    print("🌙 Emergency Abort! Negative risk detected. Moon shields activated! 🛡️")
                    return
                
                risk_percent = 0.01  # 1% equity risk
                position_size = (self.equity * risk_percent) / risk_per_share
                position_size = int(round(position_size))
                
                if position_size > 0:
                    # 🚀 SHORT Entry with Cosmic Precision
                    self.sell(size=position_size, sl=stop_loss, tag='Vortex_Short')
                    print(f"🌑 BEARISH VORTEX DETECTED! Shorting {position_size} units at {entry_price:.2f}")
                    print(f"🛑 Cosmic Stop Loss set at {stop_loss:.2f} | ATR: {atr_value