Here's the fixed code with all backtesting.lib references removed and Moon Dev themed debug prints added:

```python
from backtesting import Backtest, Strategy
import pandas as pd
import talib
import numpy as np

class SqueezeReversal(Strategy):
    ema_period = 20
    atr_period = 20
    keltner_multiplier = 1.5
    rsi_period = 3
    volume_ma_period = 20
    risk_per_trade = 0.02  # 2% risk per trade

    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col.lower()])

        # Calculate indicators with self.I()
        self.ema = self.I(talib.EMA, self.data.Close, timeperiod=self.ema_period, name='EMA')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=self.atr_period, name='ATR')
        self.upper_band = self.I(lambda ema, atr: ema + self.keltner_multiplier * atr, self.ema, self.atr, name='UpperBand')
        self.lower_band = self.I(lambda ema, atr: ema - self.keltner_multiplier * atr, self.ema, self.atr, name='LowerBand')
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=self.volume_ma_period, name='Volume_MA')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')
        self.bandwidth = self.I(lambda u, l: u - l, self.upper_band, self.lower_band, name='BandWidth')
        self.bandwidth_slope = self.I(talib.LINEARREG_SLOPE, self.bandwidth, timeperiod=5, name='BandWidth_Slope')

    def next(self):
        # Moon Dev debug prints for key values 🌙
        print(f"🌙 EMA: {self.ema[-1]:.2f} | Upper: {self.upper_band[-1]:.2f} | Lower: {self.lower_band[-1]:.2f} | RSI: {self.rsi[-1]:.2f}")
        print(f"🌙✨ Bandwidth Slope: {self.bandwidth_slope[-1]:.4f} | Volume Ratio: {self.data.Volume[-1]/self.volume_ma[-1]:.2f}x")

        if not self.position:
            # Entry logic
            in_keltner = (self.data.High[-1] < self.upper_band[-1]) and (self.data.Low[-1] > self.lower_band[-1])
            bandwidth_tightening = self.bandwidth_slope[-1] < 0
            volume_declining = self.data.Volume[-1] < self.volume_ma[-1]
            volume_spike = self.data.Volume[-1] > 1.5 * self.volume_ma[-1]

            if in_keltner and bandwidth_tightening and volume_declining:
                if volume_spike:
                    print("🚨🌙 Moon Dev Alert: Volume spike detected - avoiding crowded trade!")
                    return
                
                # Calculate stop loss and position size
                entry_price = self.data.Close[-1]
                stop_loss_price = self.lower_band[-1] - 0.5 * self.atr[-1]
                risk_amount = self.equity * self.risk_per_trade
                stop_loss_pips = entry_price - stop_loss_price
                
                if stop_loss_pips <= 0:
                    print("❌🌙 Moon Dev Alert: Invalid stop loss distance!")
                    return
                
                position_size = int(round(risk_amount / stop_loss_pips))
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss_price)
                    print(f"🚀🌕 MOON DEV LONG ENTRY @ {entry_price:.2f} | Size: {position_size} | Stop: {stop_loss_price:.2f}")

        else:
            # Exit logic for long positions
            price_outside = (self.data.Close[-1] > self.upper_band[-1]) or (self.data.Close[-1] < self.lower_band[-1])
            rsi_reversal = (