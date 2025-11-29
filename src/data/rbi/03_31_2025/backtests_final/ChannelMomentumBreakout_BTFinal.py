Here's the complete fixed backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
# 🌙 Moon Dev's ChannelMomentumBreakout Backtest 🌙
from backtesting import Backtest, Strategy
import pandas as pd
import talib

# =====================
# DATA PREPARATION
# =====================
def load_data(path):
    # Load and clean data with Moon Dev precision 🌟
    data = pd.read_csv(path)
    
    # Clean column names
    data.columns = data.columns.str.strip().str.lower()
    
    # Remove unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Map to backtesting.py compatible columns
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Convert index to datetime
    data['datetime'] = pd.to_datetime(data['datetime'])
    data.set_index('datetime', inplace=True)
    
    print("🌌 MOON DEV DATA LOAD COMPLETE -", data.shape[0], "candles loaded ✨")
    return data

# =====================
# STRATEGY IMPLEMENTATION
# =====================
class ChannelMomentumBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # =====================
        # INDICATOR CALCULATIONS
        # =====================
        # Keltner Channels (EMA20 + 2*ATR20)
        def keltner_channels(high, low, close):
            ema20 = talib.EMA(close, 20)
            atr20 = talib.ATR(high, low, close, 20)
            return ema20 + 2*atr20, ema20 - 2*atr20
            
        self.upper_keltner, self.lower_keltner = self.I(
            keltner_channels,
            self.data.High,
            self.data.Low,
            self.data.Close,
            name=['Upper_KC', 'Lower_KC']
        )
        
        # Chande Momentum Oscillator with Signal Line
        self.cmo = self.I(talib.CMO, self.data.Close, 14, name='CMO_14')
        self.cmo_signal = self.I(talib.SMA, self.cmo, 9, name='CMO_Signal')
        
        # Volume SMA Filter
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='Volume_SMA20')
        
        # Keltner Width Analysis
        keltner_width = self.upper_keltner - self.lower_keltner
        self.min_width = self.I(talib.MIN, keltner_width, 20, name='KC_MinWidth20')
        
        print("🌠 MOON DEV INDICATORS INITIALIZED - Ready for launch! 🚀")

    def next(self):
        # =====================
        # ENTRY LOGIC
        # =====================
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Keltner Channel conditions
        keltner_narrowing = (self.upper_keltner[-1] - self.lower_keltner[-1]) <= self.min_width[-1]
        above_upper = current_close > self.upper_keltner[-1]
        below_lower = current_close < self.lower_keltner[-1]
        
        # Momentum conditions
        cmo_cross_up = (self.cmo[-2] < self.cmo_signal[-2]) and (self.cmo[-1] > self.cmo_signal[-1])
        cmo_cross_down = (self.cmo[-2] > self.cmo_signal[-2]) and (self.cmo[-1] < self.cmo_signal[-1])
        
        # Volume filter
        volume_ok = current_volume > self.volume_sma[-1]
        
        if not self.position:
            # 🌙 LONG ENTRY CONDITIONS 🌙
            if all([keltner_narrowing, cmo_cross_up, volume_ok, above_upper]):
                risk_amount = self.equity * self.risk_per_trade
                stop_price = self.lower_keltner[-1]
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    size = int(round(risk_amount / risk_per_share))
                    self.buy