```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class VortexFade(Strategy):
    risk_pct = 0.01  # 1% risk per trade
    
    def init(self):
        # Calculate indicators using self.I() wrapper
        high = self.data.High
        low = self.data.Low
        close = self.data.Close
        
        # Vortex Indicator (VI+ and VI-)
        vi = ta.vortex(high=high, low=low, close=close, length=14)
        self.vi_plus = self.I(lambda x: x, vi['VORTICSm_14'], name='VI+ 🌙')
        self.vi_minus = self.I(lambda x: x, vi['VORTICSs_14'], name='VI- 🌑')
        
        # ADX (14-period)
        self.adx = self.I(talib.ADX, high, low, close, timeperiod=14, name='ADX 🌀')
        
        # Volume SMA for confirmation
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='Vol SMA 📉')
        
        # Swing Highs/Lows for stop loss
        self.swing_high = self.I(talib.MAX, high, 20, name='Swing High 🚀')
        self.swing_low = self.I(talib.MIN, low, 20, name='Swing Low 🛑')

    def next(self):
        # Warmup period check
        if len(self.data) < 20:
            return
            
        # Current indicator values
        vi_plus, vi_minus = self.vi_plus[-1], self.vi_minus[-1]
        prev_vi_plus, prev_vi_minus = self.vi_plus[-2], self.vi_minus[-2]
        
        # Entry conditions
        vi_bullish = vi_plus > vi_minus and prev_vi_plus <= prev_vi_minus
        vi_bearish = vi_plus < vi_minus and prev_vi_plus >= prev_vi_minus
        volume_ok = self.data.Volume[-1] < self.volume_sma[-1]
        adx_ok = self.adx[-1] < 30
        
        # Moon Dev themed risk management 🌙
        if not self.position:
            # Long entry logic
            if vi_bullish and volume_ok and adx_ok:
                sl_price = self.swing_low[-1]
                risk_per_share = self.data.Close[-1] - sl_price
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    self.buy(size=size, sl=sl_price)
                    print(f"🌙✨ BULLISH VORTEX CROSS! Buying {size} @ {self.data.Close[-1]} ✨ SL: {sl_price}")
            
            # Short entry logic
            elif vi_bearish and volume_ok and adx_ok:
                sl_price = self.swing_high[-1]
                risk_per_share = sl_price - self.data.Close[-1]
                if risk_per_share > 0:
                    size = int(round((self.equity * self.risk_pct) / risk_per_share))
                    self.sell(size=size, sl=sl_price)
                    print(f"🌑🚀 BEARISH VORTEX CROSS! Selling {size} @ {self.data.Close[-1]} ✨ SL: {sl_price}")

        # Exit logic - ADX momentum drop
        elif self.adx[-1] < 25:
            self.position.close()
            print(f"🌀 ADX FADING! Closing position @ {self.data.Close[-1]}")

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

# Launch Moon Dev Backtest 🌙
bt = Backtest(data, VortexFade, cash=1_000_000