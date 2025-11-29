Here's the fixed and complete backtest code with Moon Dev themed debug prints and proper position sizing implementation:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class VolumeMACDDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Clean and prepare data
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        self.data.drop(columns=[col for col in self.data.columns if 'unnamed' in col], inplace=True)
        
        # Calculate indicators using TA-Lib with self.I()
        close = self.data.Close
        volume = self.data.Volume
        
        # MACD components
        self.macd = self.I(talib.MACD, close, fastperiod=12, slowperiod=26, signalperiod=9, name='MACD')
        self.macd_line = self.I(lambda: self.macd[0], name='MACD_LINE')
        self.signal_line = self.I(lambda: self.macd[1], name='SIGNAL_LINE')
        
        # Volume SMA
        self.volume_sma = self.I(talib.SMA, volume, timeperiod=20, name='VOLUME_SMA20')
        
        # Trend indicators
        self.ema200 = self.I(talib.EMA, close, timeperiod=200, name='EMA200')
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SWING_HIGH')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SWING_LOW')
        
        print("🌙 Moon Dev Indicators Activated! ✨")
        print("📊 MACD, Volume SMA, and Trend Indicators Initialized Successfully!")

    def next(self):
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Entry logic
        if not self.position:
            # Long entry conditions
            macd_above = (self.macd_line[-2] < self.signal_line[-2] and 
                        self.macd_line[-1] > self.signal_line[-1])
            volume_trigger = current_volume > 1.5 * self.volume_sma[-1]
            trend_confirmation = current_close > self.ema200[-1]
            
            if macd_above and volume_trigger and trend_confirmation:
                sl_price = self.swing_low[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = current_close - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        tp_price = current_close + 2 * risk_per_share
                        self.buy(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🚀🌕 MOON DEV LONG SIGNAL! Size: {position_size} | Entry: {current_close} | SL: {sl_price} | TP: {tp_price}")

            # Short entry conditions
            macd_below = (self.macd_line[-2] > self.signal_line[-2] and 
                         self.macd_line[-1] < self.signal_line[-1])
            price_downtrend = (self.data.High[-1] < self.data.High[-2] and 
                             self.data.Low[-1] < self.data.Low[-2])
            
            if macd_below and volume_trigger and price_downtrend:
                sl_price = self.swing_high[-1]
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = sl_price - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        tp_price = current_close - 2 * risk_per_share
                        self.sell(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🌑🚨 MOON DEV SHORT SIGNAL! Size: {position_size} | Entry: {current_close} | SL: {sl_price} | TP: {tp_price}")

        # Exit logic
        else:
            if self.position.is_long:
                # Long exit conditions
                bearish_divergence = (self.data.High[-1] > self.data.High[-2] and 
                                     self.macd_line[-1] < self.macd_line[-2