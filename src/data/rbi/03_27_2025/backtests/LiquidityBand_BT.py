```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Load and prepare data
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path)

# Clean data columns
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

class LiquidityBand(Strategy):
    def init(self):
        # Moon Dev Indicator Factory 🌙
        close = self.data.Close
        high = self.data.High
        low = self.data.Low
        volume = self.data.Volume

        # Bollinger Bands with Width Calculation
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.bb_upper = self.I(lambda: bb_upper, name='BB_Upper')
        self.bb_middle = self.I(lambda: bb_middle, name='BB_Middle')
        self.bb_lower = self.I(lambda: bb_lower, name='BB_Lower')
        bb_width = (bb_upper - bb_lower) / bb_middle
        self.bb_width = self.I(lambda: bb_width, name='BB_Width')

        # Liquidity Zones (Swing Highs/Lows)
        self.swing_high = self.I(talib.MAX, high, timeperiod=20, name='Swing_High')
        self.swing_low = self.I(talib.MIN, low, timeperiod=20, name='Swing_Low')

        # Volume Confirmation
        self.volume_ma = self.I(talib.SMA, volume, timeperiod=20, name='Volume_MA')

        # Volatility Expansion Threshold
        self.bb_width_50ma = self.I(talib.SMA, bb_width, timeperiod=50, name='BB_Width_50MA')

        # Risk Management Indicators
        self.atr = self.I(talib.ATR, high, low, close, timeperiod=14, name='ATR')

        print("🌙✨ Lunar indicators activated! Ready for cosmic gains 🚀")

    def next(self):
        # Galactic Price Action Analysis 🌌
        price = self.data.Close[-1]
        volatility = self.bb_width[-1]
        liquidity_high = self.swing_high[-1]
        liquidity_low = self.swing_low[-1]

        if not self.position:
            # Long Entry Constellation 🌠
            if (len(self.bb_width) > 20 and 
                volatility == min(self.bb_width[-20:]) and 
                price > liquidity_high and 
                self.data.Volume[-1] > self.volume_ma[-1]):
                
                risk_amount = 0.01 * self.equity
                atr = self.atr[-1]
                stop_loss = price - 2 * atr
                position_size = int(round(risk_amount / (price - stop_loss)))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss)
                    print(f"🌙🚀 BLAST OFF LONG! Entry: {price:.2f} | Size: {position_size} | Cosmic SL: {stop_loss:.2f} ✨")

            # Short Entry Meteor Shower 🌠
            elif (len(self.bb_width) > 20 and 
                  volatility == min(self.bb_width[-20:]) and 
                  price < liquidity_low and 
                  self.data.Volume[-1] > self.volume_ma[-1]):
                
                risk_amount = 0.01 * self.equity
                atr = self.atr[-1]
                stop_loss = price + 2 * atr
                position_size = int(round(risk_amount / (stop_loss - price)))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_loss)
                    print(f"🌙🌌 BLACK HOLE SHORT! Entry: {price:.2f} | Size: {position_size} | Singularity SL: {stop_loss:.2f} ✨")

        else:
            #