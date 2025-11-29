Here's the fixed and complete code with Moon Dev themed debug prints! 🌙 ✨

```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

class FractalReversion(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    rr_ratio = 2  # Risk-reward ratio
    
    def init(self):
        # Clean data columns
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators
        # StochRSI
        stoch_rsi = self.data.df.ta.stochrsi(length=14, rsi_length=14, k=3, d=3)
        self.stoch_k = self.I(lambda: stoch_rsi['STOCHRSIk_14_14_3_3'].values, name='StochRSI K')
        self.stoch_d = self.I(lambda: stoch_rsi['STOCHRSId_14_14_3_3'].values, name='StochRSI D')
        
        # VDMA (KAMA implementation)
        self.vdma = self.I(talib.KAMA, self.data.Close, timeperiod=30, name='VDMA')
        
        # Fractals with 2-bar lag
        bearish, bullish = self.data.df.ta.fractals()
        self.bearish_fractal = self.I(lambda: bearish.shift(2).fillna(0).astype(float), name='Bearish Fractal')
        self.bullish_fractal = self.I(lambda: bullish.shift(2).fillna(0).astype(float), name='Bullish Fractal')
        
        print("🌙 Initialized FractalReversion strategy with cosmic indicators! 🚀")

    def next(self):
        if len(self.data) < 5:
            return
        
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        
        # Bullish setup check
        bull_div = False
        if current_low < self.data.Low[-2] and self.stoch_k[-1] > self.stoch_k[-2]:
            bull_div = True
            print(f"✨ Potential bullish divergence detected at {self.data.index[-1]}")

        # Bearish setup check
        bear_div = False
        if current_high > self.data.High[-2] and self.stoch_k[-1] < self.stoch_k[-2]:
            bear_div = True
            print(f"✨ Potential bearish divergence detected at {self.data.index[-1]}")

        # Entry logic
        if not self.position:
            # Bullish entry
            if bull_div and current_close > self.vdma[-1] and self.bullish_fractal[-1] == 1:
                fractal_high = self.data.High[-3]
                if current_close > fractal_high:
                    sl_price = self.data.Low[-3]  # Fractal low
                    risk_amount = self.equity * self.risk_per_trade
                    risk_per_share = current_close - sl_price
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount / risk_per_share))
                        tp_price = current_close + (current_close - sl_price) * self.rr_ratio
                        self.buy(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🚀🌕 BULLISH ENTRY at {current_close:.2f}")
                        print(f"    Size: {position_size}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")

            # Bearish entry
            elif bear_div and current_close < self.vdma[-1] and self.bearish_fractal[-1] == 1:
                fractal_low = self.data.Low[-3]
                if current_close < fractal_low:
                    sl_price = self.data.High[-3]  # Fractal high
                    risk_amount = self.equity * self.risk_per_trade
                    risk_per_share = sl_price - current_close
                    
                    if risk_per_share > 0:
                        position_size = int(round(risk_amount / risk_per_share))
                        tp_price = current_close - (sl_price - current_close) * self.rr_ratio
                        self.sell(size=position_size, sl=sl_price, tp=tp_price)
                        print(f"🌑🚀