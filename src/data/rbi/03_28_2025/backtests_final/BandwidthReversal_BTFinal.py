Here's the fully debugged and Moon Dev themed version of your BandwidthReversal strategy:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
import numpy as np

# Moon Dev Data Loading Sequence 🌙
print("🌌 INITIALIZING MOON DEV DATA SYSTEMS...")
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
print("🌠 DATA SYSTEMS ONLINE! MOON BASE OPERATIONAL")

class BandwidthReversal(Strategy):
    def init(self):
        # Moon Dev Indicator Initialization Sequence ✨
        print("🌙 INITIALIZING LUNAR INDICATORS...")
        close = self.data.Close
        
        # Bollinger Bands (20,2)
        upper, _, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.I(lambda: upper, name='BB_upper')
        self.I(lambda: lower, name='BB_lower')
        bb_width = upper - lower
        self.I(lambda: bb_width, name='BB_width')
        self.I(lambda: talib.MIN(bb_width, timeperiod=20), name='BB_width_low')

        # Stochastic RSI (14,3,3)
        stoch_rsi = ta.stochrsi(close, length=14, k=3, d=3)
        self.I(lambda: stoch_rsi.iloc[:,0], name='StochRSI_K')
        self.I(lambda: stoch_rsi.iloc[:,1], name='StochRSI_D')

        # ATR (14)
        self.I(lambda: talib.ATR(self.data.High, self.data.Low, 
                               self.data.Close, timeperiod=14), name='ATR')

        # Trade tracking variables
        self.trailing_stop = None
        self.highest_high = None
        print("✨ LUNAR INDICATORS ONLINE! READY FOR TRADING")

    def next(self):
        # Wait for enough data
        if len(self.data) < 20:
            return

        # Moon Dev Debugging Console 🌙
        current_bb_width = self.data.BB_width[-1]
        current_bb_low = self.data.BB_width_low[-1]
        stoch_k = self.data.StochRSI_K[-1]
        stoch_k_prev = self.data.StochRSI_K[-2] if len(self.data) > 1 else 0
        atr = self.data.ATR[-1]
        
        print(f"\n🌙 MOON DEV TRADING CYCLE {len(self.data)} 🌙")
        print(f"📊 BB Width: {current_bb_width:.2f} | BB Low: {current_bb_low:.2f}")
        print(f"✨ Stoch K: {stoch_k:.2f} (prev: {stoch_k_prev:.2f}) | ATR: {atr:.2f}")

        # Entry Logic
        if not self.position:
            # Check crossover using manual comparison
            stoch_cross = (stoch_k_prev < 20) and (stoch_k > 20)
            
            if (current_bb_width < current_bb_low) and stoch_cross:
                # Moon Dev Risk Management Protocol 🌙
                risk_percent = 0.01  # 1% risk per trade
                risk_amount = self.equity * risk_percent
                atr_value = self.data.ATR[-1]
                
                if atr_value == 0:
                    print("🚨 MOON WARNING: Zero ATR value detected!")
                    return
                
                position_size = risk_amount / (1.5 * atr_value)
                position_size = int(round(position_size))  # Ensure whole units
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"\n🌙✨ MOON DEV ALERT: LONG ENTERED! ✨🌙")
                    print(f"📈 Size