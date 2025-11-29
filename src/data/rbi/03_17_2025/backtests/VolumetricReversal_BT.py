```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import numpy as np

# Data preparation
data = pd.read_csv('/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv')

# Clean and prepare data
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
data.set_index('datetime', inplace=True)

class VolumetricReversal(Strategy):
    risk_percentage = 0.01
    atr_multiplier_sl = 1.5
    atr_multiplier_tp = 2
    
    def init(self):
        # Initialize indicators with self.I()
        self.so_slowk = self.I(talib.STOCH, self.data.High, self.data.Low, self.data.Close,
                              fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0]
        self.so_slowd = self.I(talib.STOCH, self.data.High, self.data.Low, self.data.Close,
                              fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1]
        
        self.upper_band, self.middle_band, self.lower_band = self.I(talib.BBANDS, self.data.Close, 
                                                                  timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Candlestick patterns
        self.bullish_engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High, self.data.Low, self.data.Close)
        
    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Moon Dev Debug Prints 🌙
        print(f"🌙 Current Close: {current_close:.2f} | Volume: {current_volume:.2f} | ATR: {self.atr[-1]:.2f} ✨")
        
        if not self.position:
            # Entry conditions
            so_overbought = self.so_slowk[-1] > 80
            price_above_bb = current_close > self.upper_band[-1]
            volume_spike = current_volume > 1.5 * self.volume_ma[-1]
            bullish_pattern = self.bullish_engulfing[-1] == 100 or self.hammer[-1] == 100
            
            if all([so_overbought, price_above_bb, volume_spike, bullish_pattern]):
                # Calculate position size
                risk_amount = self.equity * self.risk_percentage
                atr_value = self.atr[-1]
                stop_loss = current_close - self.atr_multiplier_sl * atr_value
                risk_per_share = current_close - stop_loss
                
                if risk_per_share <= 0:
                    print("⚠️ Risk per share <= 0! Skipping trade! 🌙")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                if position_size == 0:
                    print("⚠️ Position size 0! Not entering! 🌙")
                    return
                
                # Enter trade with Moon Dev flair 🌕
                self.buy(size=position_size, sl=stop_loss, 
                        tp=current_close + self.atr_multiplier_tp * atr_value)
                print(f"🌕 MOON DEV ALERT! Long entry at {current_close:.2f} 🚀")
                print(f"Size: {position_size} BTC | SL: {stop_loss:.2f} | TP: {self.atr_multiplier_tp