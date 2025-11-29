```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class BandMACDDivergence(Strategy):
    # Strategy parameters
    bb_period = 20
    bb_dev = 2
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    atr_period = 14
    slope_period = 5
    risk_pct = 0.01
    atr_multiplier = 1.5
    slope_threshold = 0.2

    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])

        # Calculate indicators with TA-Lib
        self.upper_band = self.I(talib.BBANDS, self.data.Close, timeperiod=self.bb_period, 
                                nbdevup=self.bb_dev, nbdevdn=self.bb_dev, matype=0, name='Upper BB', which=0)
        self.macd_line, self.macd_signal, _ = self.I(talib.MACD, self.data.Close, 
                                                    fastperiod=self.macd_fast, slowperiod=self.macd_slow, 
                                                    signalperiod=self.macd_signal, name='MACD')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.atr_period, name='ATR')
        
        # Calculate Bollinger Band slope using MAX (swing high approach)
        self.ub_slope = self.I(talib.LINEARREG_SLOPE, self.upper_band, timeperiod=self.slope_period, name='UB Slope')

    def next(self):
        # Moon Dev's cosmic checks 🌌
        if len(self.data) < 50:  # Ensure enough data
            return

        # Current cosmic readings 🌠
        price = self.data.Close[-1]
        ub_slope = self.ub_slope[-1]
        macd_diff = self.macd_line[-1] - self.macd_signal[-1]
        atr_value = self.atr[-1]

        # Universal risk management 🌙
        risk_amount = self.equity * self.risk_pct
        position_size = 0

        # Long entry supernova conditions 🌟
        if not self.position:
            if (ub_slope > self.slope_threshold and
                macd_diff > self.atr_multiplier * atr_value and
                atr_value > 10):  # Avoid low volatility
                
                # Calculate stardust position size 🌠
                stop_loss = price - 2 * atr_value
                risk_per_share = price - stop_loss
                position_size = int(round(risk_amount / risk_per_share))

                if position_size > 0:
                    self.buy(size=position_size, sl=stop_loss, 
                            tp=price + 3 * atr_value, 
                            tag="🌕 Moon Beam Long")
                    print(f"🚀🌙 LIFT OFF! Long {position_size} @ {price:.2f} | "
                          f"SL: {stop_loss:.2f} | Cosmic Momentum: {ub_slope:.2f}")

        # Galactic exit protocols 🛸
        elif self.position.is_long:
            if (ub_slope < 0 or 
                crossover(self.macd_signal, self.macd_line)):
                self.position.close()
                print(f"🌑🌠 COMET RETREAT! Closing long @ {price:.2f} | "
                      f"Profit: {self.position.pl_pct:.1f}%")

# Cosmic data preparation 🌍
data = pd.read_csv("/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv")
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

# Launch backtest rocket 🚀