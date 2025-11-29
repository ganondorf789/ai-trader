```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

class VortexSqueeze(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # 🌙 Moon Dev Indicator Calculation Phase ✨
        print("🌙 Initializing VortexSqueeze strategy with Moon Dev power...")
        
        # 1. Vortex Indicator
        vortex = pandas_ta.vortex(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            length=14
        )
        self.vi_plus = self.I(lambda: vortex['VTX_14+'], name='VI+')
        self.vi_minus = self.I(lambda: vortex['VTX_14-'], name='VI-')

        # 2. Bollinger Bands Width
        def bb_width(close):
            upper, middle, lower = talib.BBANDS(close, 
                timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return (upper - lower) / middle
        self.bb_width = self.I(bb_width, self.data.Close, name='BB Width')
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, 10, name='BB Width SMA10')

        # 3. Volume Median
        self.volume_median = self.I(talib.MEDIAN, self.data.Volume, 20, name='Vol Median20')

        # 4. ATR for Stops
        self.atr = self.I(talib.ATR, 
            self.data.High, self.data.Low, self.data.Close, 14, name='ATR14')

    def next(self):
        # 🌙 Moon Dev Signal Processing 🌙
        current_close = self.data.Close[-1]
        
        # Entry Conditions
        vi_cross = (self.vi_plus[-1] > self.vi_minus[-1] and 
                    self.vi_plus[-2] <= self.vi_minus[-2])
        bb_squeeze = self.bb_width[-1] < self.bb_width_avg[-1]
        volume_spike = self.data.Volume[-1] > 2 * self.volume_median[-1]

        # Entry Logic
        if not self.position and vi_cross and bb_squeeze and volume_spike:
            atr_value = self.atr[-1]
            stop_loss = current_close - 1.5 * atr_value
            
            # 🚀 Moon Dev Position Sizing Algorithm ✨
            risk_amount = self.risk_per_trade * self.equity
            risk_per_share = current_close - stop_loss
            position_size = risk_amount / risk_per_share
            position_size = int(round(position_size))
            
            print(f"🌙 VortexSqueeze BUY TRIGGERED! ✨\n"
                  f"Entry: {current_close:.2f} | "
                  f"SL: {stop_loss:.2f} | "
                  f"Size: {position_size} contracts 🚀")
            
            self.buy(size=position_size, sl=stop_loss)

        # Exit Logic
        elif self.position.is_long:
            # VI- crosses above VI+
            if crossover(self.vi_minus, self.vi_plus):
                print(f"🌙 Vortex Reversal DETECTED! 🛑 Closing position at {current_close:.2f}")
                self.position.close()

# 🌙 Moon Dev Data Preparation Ritual ✨
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
data['DateTime'] = pd.to_datetime(data['datetime'])
data.set_index('DateTime', inplace=True)

print("🌙 Launching Backtest with 1,000,000 Moon Dev Bucks... 🚀")
bt = Backtest(data, VortexSqueeze, cash=1_000_000, commission=.002)
stats = bt.run()

print("\n🌙🌙🌙 FINAL BACKTEST RESULTS 🌙🌙🌙")
print(stats)
print