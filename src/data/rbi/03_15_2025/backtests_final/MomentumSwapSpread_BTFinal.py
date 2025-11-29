```python
# 🌙 Moon Dev's MomentumSwapSpread Backtest Implementation 🚀
from backtesting import Backtest, Strategy
import talib
import pandas as pd
import numpy as np

class MomentumSwapSpread(Strategy):
    risk_per_trade = 0.02  # 2% risk per trade 🌕
    stop_loss_pct = 0.01   # 1% stop loss 🛑
    rsi_period = 14
    macd_fast = 12
    macd_slow = 26
    macd_signal = 9
    spread_short = 20
    spread_long = 50
    hist_spread_period = 50

    def init(self):
        # 🌀 Initialize indicators with TA-Lib
        close = self.data.Close
        
        # 🌗 Momentum Indicators
        self.rsi = self.I(talib.RSI, close, self.rsi_period)
        macd, macd_signal, _ = talib.MACD(close, 
                                         self.macd_fast,
                                         self.macd_slow,
                                         self.macd_signal)
        self.macd_line = self.I(lambda: macd, name='MACD')
        self.macd_signal = self.I(lambda: macd_signal, name='MACD Signal')
        
        # 🌌 Spread Calculation
        short_sma = talib.SMA(close, self.spread_short)
        long_sma = talib.SMA(close, self.spread_long)
        self.spread = self.I(lambda: short_sma - long_sma, name='Spread')
        self.hist_spread = self.I(talib.SMA, self.spread, self.hist_spread_period,
                                name='Hist Spread')

    def next(self):
        # 🌠 Current indicator values
        current_rsi = self.rsi[-1]
        current_spread = self.spread[-1]
        hist_spread = self.hist_spread[-1]
        price = self.data.Close[-1]

        # 🪐 Risk Management Calculations
        equity = self.equity
        position_size = int(round((equity * self.risk_per_trade) / 
                           (price * self.stop_loss_pct)))

        # 💫 Close positions first
        if self.position:
            if self.position.is_long:
                # Bullish to Bearish crossover detection
                if (current_rsi > 70 or 
                    current_spread >= hist_spread or 
                    (self.macd_signal[-2] > self.macd_line[-2] and 
                     self.macd_signal[-1] < self.macd_line[-1])):
                    self.position.close()
                    print(f"🌙 Closing LONG | RSI: {current_rsi:.1f} | "
                          f"Spread: {current_spread:.2f} → Hist: {hist_spread:.2f} ✨")

            elif self.position.is_short:
                # Bearish to Bullish crossover detection
                if (current_rsi < 30 or 
                    current_spread <= hist_spread or 
                    (self.macd_line[-2] > self.macd_signal[-2] and 
                     self.macd_line[-1] < self.macd_signal[-1])):
                    self.position.close()
                    print(f"🌑 Closing SHORT | RSI: {current_rsi:.1f} | "
                          f"Spread: {current_spread:.2f} → Hist: {hist_spread:.2f} 🌌")

        # 🌟 Open new positions
        if not self.position:
            # 🚀 Long Entry: RSI <30 + Spread < Hist + MACD Bullish
            if (current_rsi < 30 and 
                current_spread < hist_spread and 
                (self.macd_line[-2] < self.macd_signal[-2] and 
                 self.macd_line[-1] > self.macd_signal[-1])):
                sl = price * (1 - self.stop_loss_pct)
                self.buy(size=position_size, sl=sl)
                print(f"🚀 MOON BULLISH | Entry: {price:.2f} | "
                      f"RSI: {current_rsi:.1f} | Size: {position_size} 🌕")

            # 🌑 Short Entry: RSI >70 + Spread > Hist + MACD Bearish
            elif (current_rsi > 70 and 
                  current_spread > hist_spread and 
                  (self.macd_signal[-2] < self.macd_line[-2] and 
                   self.macd_signal[-1] > self.macd