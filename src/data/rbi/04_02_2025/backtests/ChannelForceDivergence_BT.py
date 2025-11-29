```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class ChannelForceDivergence(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    max_positions = 3

    def init(self):
        # Calculate indicators with TA-Lib using self.I()
        self.donchian_upper = self.I(talib.MAX, self.data.High, timeperiod=20, name='Donchian Upper 🌙')
        self.donchian_lower = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Donchian Lower 🌙')
        
        # Elder Force Index (13-period EMA of Force Index)
        mom = self.I(talib.MOM, self.data.Close, timeperiod=1, name='MOM_1')
        force_index_raw = mom * self.data.Volume
        self.efi = self.I(talib.EMA, force_index_raw, timeperiod=13, name='EFI 🚀')
        
        # 4-hour RSI with resampling
        self.rsi_4h = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI_4H 🔍', aspan='4H')
        
        # Divergence tracking
        self.prev_4h_rsi = []
        self.prev_4h_close = []
        self.bullish_divergence = False
        self.bearish_divergence = False

    def next(self):
        current_close = self.data.Close[-1]
        donchian_upper = self.donchian_upper[-1]
        donchian_lower = self.donchian_lower[-1]
        efi = self.efi[-1]

        # Detect 4H RSI divergence 🌗
        if len(self.rsi_4h) > 2 and self.rsi_4h[-1] != self.rsi_4h[-2]:
            new_rsi = self.rsi_4h[-1]
            new_close = self.data.Close[-1]
            
            if len(self.prev_4h_rsi) >= 2:
                rsi1, close1 = self.prev_4h_rsi[-2], self.prev_4h_close[-2]
                rsi2, close2 = self.prev_4h_rsi[-1], self.prev_4h_close[-1]
                
                self.bullish_divergence = (close2 < close1) and (rsi2 > rsi1)
                self.bearish_divergence = (close2 > close1) and (rsi2 < rsi1)
                
                if self.bullish_divergence:
                    print(f"🌙✨ Bullish Divergence! Close: {close2:.2f}<{close1:.2f}, RSI: {rsi2:.2f}>{rsi1:.2f}")
                if self.bearish_divergence:
                    print(f"🌙✨ Bearish Divergence! Close: {close2:.2f}>{close1:.2f}, RSI: {rsi2:.2f}<{rsi1:.2f}")

            self.prev_4h_rsi.append(new_rsi)
            self.prev_4h_close.append(new_close)
            self.prev_4h_rsi = self.prev_4h_rsi[-2:]
            self.prev_4h_close = self.prev_4h_close[-2:]

        # Entry Logic 🚀🌑
        if not self.position and len(self.trades) < self.max_positions:
            # Long Entry
            if current_close > donchian_upper and efi > 0 and self.bullish_divergence:
                sl = donchian_lower * 0.999
                risk = current_close - sl
                position_size = int(round((self.risk_per_trade * self.equity) / risk))
                if position_size > 0:
                    self.buy(size=position_size, sl=sl, tag="LONG-ENTRY 🌕")
                    print(f"🚀🌕 LONG | Size: {position_size} | SL: {sl:.2f}")

            # Short Entry
            elif current_close < donchian_lower and efi < 0 and self.bearish_divergence:
                sl = donchian_upper * 1.001
                risk = sl - current_close
                position_size = int(round((self.risk_per_trade * self.equity) / risk))
                if position_size > 0:
                    self.sell(size=position_size, sl=