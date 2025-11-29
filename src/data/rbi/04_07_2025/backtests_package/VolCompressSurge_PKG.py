Here's the fixed code with all backtesting.lib imports removed and replaced with proper implementations, along with Moon Dev themed debug prints:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['datetime'] = pd.to_datetime(data['datetime'])
data.set_index('datetime', inplace=True)

class VolCompressSurge(Strategy):
    def init(self):
        # Bollinger Bands
        def bb_upper(close):
            upper, _, _ = talib.BBANDS(close, 20, 2, 2)
            return upper
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_upper')
        
        def bb_middle(close):
            _, middle, _ = talib.BBANDS(close, 20, 2, 2)
            return middle
        self.bb_middle = self.I(bb_middle, self.data.Close, name='BB_middle')
        
        def bb_lower(close):
            _, _, lower = talib.BBANDS(close, 20, 2, 2)
            return lower
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_lower')

        # Keltner Channel
        self.ema20 = self.I(talib.EMA, self.data.Close, timeperiod=20, name='EMA20')
        self.atr20 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=20, name='ATR20')
        
        def kc_upper(ema, atr):
            return ema + 2 * atr
        self.kc_upper = self.I(kc_upper, self.ema20, self.atr20, name='KC_upper')
        
        def kc_lower(ema, atr):
            return ema - 2 * atr
        self.kc_lower = self.I(kc_lower, self.ema20, self.atr20, name='KC_lower')

        # Volume MA
        self.volume_ma50 = self.I(talib.SMA, self.data.Volume, timeperiod=50, name='Volume_MA50')

    def next(self):
        if len(self.data) < 50:
            return

        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        volume_ma50 = self.volume_ma50[-1]
        bb_upper = self.bb_upper[-1]
        bb_lower = self.bb_lower[-1]
        kc_upper_current = self.kc_upper[-1]
        kc_upper_prev = self.kc_upper[-2] if len(self.kc_upper) > 1 else None

        kc_descending = kc_upper_current < kc_upper_prev if kc_upper_prev else False

        if not self.position:
            # Long Entry
            if (current_close > bb_upper and 
                current_volume > 1.2 * volume_ma50 and 
                kc_descending):
                
                bb_mid = self.bb_middle[-1]
                kc_low = self.kc_lower[-1]
                sl = max(bb_mid, kc_low)
                risk_per_share = current_close - sl
                
                if risk_per_share <= 0:
                    print("🌙✨ Warning: Risk per share <= 0 - skipping trade")
                    return
                
                risk_amount = 0.02 * self.equity
                size = int(round(risk_amount / risk_per_share))
                if size > 0:
                    tp = current_close + 2 * risk_per_share
                    self.buy(size=size, sl=sl, tp=tp)
                    print(f"🌙✨🚀 MOONSHOT LONG! Price: {current_close:.2f}, Size: {size}, SL: {sl:.2f}, TP: {tp:.2f}")

            # Short Entry