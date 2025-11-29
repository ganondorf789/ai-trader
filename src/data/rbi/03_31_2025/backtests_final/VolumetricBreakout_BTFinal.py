Here's the fully debugged code with Moon Dev themed improvements while preserving the original strategy logic:

```python
import pandas as pd
import talib as ta
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation Ritual 🌙
def prepare_data(filepath):
    data = pd.read_csv(filepath)
    # Cleanse column names
    data.columns = data.columns.str.strip().str.lower()
    # Banish unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    # Set temporal index
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    return data

class VolumetricBreakout(Strategy):
    def init(self):
        # Celestial Indicators 🌌 (Pure TA-Lib powered)
        close = self.data.Close
        self.ema50 = self.I(ta.EMA, close, timeperiod=50, name='EMA50')
        self.ema200 = self.I(ta.EMA, close, timeperiod=200, name='EMA200')
        self.volume_sma20 = self.I(ta.SMA, self.data.Volume, timeperiod=20, name='Volume SMA20')
        self.atr14 = self.I(ta.ATR, self.data.High, self.data.Low, close, timeperiod=14, name='ATR14')

    def next(self):
        # Cosmic Values Alignment ✨
        ema50 = self.ema50[-1]
        ema200 = self.ema200[-1]
        prev_ema50 = self.ema50[-2] if len(self.ema50) > 1 else ema50
        prev_ema200 = self.ema200[-2] if len(self.ema200) > 1 else ema200
        
        # Lunar Entry Conditions 🌕
        if not self.position:
            # EMA Crossover Constellation (Manual detection)
            golden_cross = (prev_ema50 <= prev_ema200) and (ema50 > ema200)
            # Volume Supernova
            volume_blast = self.data.Volume[-1] > self.volume_sma20[-1]
            
            if golden_cross and volume_blast:
                # Astral Risk Calculation 🌠
                risk_capital = self.equity * 0.01  # 1% risk
                atr_value = self.atr14[-1]
                if atr_value == 0:
                    print("🚨 Zero ATR - Cosmic Anomaly Detected!")
                    return
                
                # Meteor Position Sizing ☄️ (Fixed to whole units)
                position_size = risk_capital / (2 * atr_value)
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    entry_price = self.data.Close[-1]
                    # Initiate Stellar Stop Loss 🌑 (price level, not distance)
                    stop_price = entry_price - 2 * atr_value
                    self.position.sl = stop_price
                    self.position.peak = entry_price  # Track cosmic highs
                    print(f"🌕 MOON DEV ALERT: Entry at {entry_price:.2f}")
                    print(f"🛡️ Initial Stop: {stop_price:.2f} | Size: {position_size} units")

        # Galactic Exit Protocol 🚀
        elif self.position:
            # Update Cosmic Peak 🌟
            current_high = self.data.High[-1]
            if current_high > self.position.peak:
                self.position.peak = current_high
                # Trailing Stop Nebula 🌌 (price level)
                new_sl = self.position.peak - 1.5 * self.atr14[-1]
                if new_sl > self.position.sl:
                    self.position.sl = new_sl
                    print(f"✨ Trailing Stop Updated: {new_sl:.2f}")

            # Check for Supernova Collapse (Stop Loss) 💥
            if self.data.Low[-1] <= self.position.sl:
                self.position.close()
                print(f"🌑 STOP LOSS TRIGGERED: {self.position.sl:.2f}")

            # Death Cross Abort Sequence ☠️ (Manual detection)
            death_cross = (prev_ema50 >= prev_ema200) and (ema50 < ema200)
            if death_cross:
                self.position.close()
                print