The code you provided is already well-structured and avoids using any `backtesting.lib` imports or functions. However, I'll ensure that all indicators are properly wrapped in `self.I()` and that the code follows Moon Dev's best practices. Here's the complete and verified code:

```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class DivergentVolatility(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_multiplier_sl = 2  # Stop loss multiplier
    atr_threshold = 1.5  # ATR exit threshold
    swing_lookback = 20  # Lookback for swing highs/lows
    
    def init(self):
        # Calculate indicators with proper self.I() wrapper
        def calculate_macd(close):
            macd, _, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            return macd
        
        self.macd = self.I(calculate_macd, self.data.Close, name='MACD')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14, name='ATR')
        self.atr_ma = self.I(talib.SMA, self.atr, timeperiod=14, name='ATR MA')
        
        # Swing calculations using MAX/MIN
        self.price_highs = self.I(talib.MAX, self.data.High, timeperiod=self.swing_lookback, name='Price Highs')
        self.price_lows = self.I(talib.MIN, self.data.Low, timeperiod=self.swing_lookback, name='Price Lows')
        self.rsi_highs = self.I(talib.MAX, self.rsi, timeperiod=self.swing_lookback, name='RSI Highs')
        self.macd_lows = self.I(talib.MIN, self.macd, timeperiod=self.swing_lookback, name='MACD Lows')
        
        # Track swing points
        self.swing_low_prices = []
        self.swing_low_macd = []
        self.swing_high_prices = []
        self.swing_high_rsi = []

    def next(self):
        if len(self.data) < self.swing_lookback + 1:
            return
        
        # Update swing points 🌙
        current_price_low = self.price_lows[-1]
        if not self.swing_low_prices or current_price_low != self.swing_low_prices[-1]:
            self.swing_low_prices.append(current_price_low)
            if len(self.swing_low_prices) > 2:
                self.swing_low_prices.pop(0)
                
        current_macd_low = self.macd_lows[-1]
        if not self.swing_low_macd or current_macd_low != self.swing_low_macd[-1]:
            self.swing_low_macd.append(current_macd_low)
            if len(self.swing_low_macd) > 2:
                self.swing_low_macd.pop(0)
                
        current_price_high = self.price_highs[-1]
        if not self.swing_high_prices or current_price_high != self.swing_high_prices[-1]:
            self.swing_high_prices.append(current_price_high)
            if len(self.swing_high_prices) > 2:
                self.swing_high_prices.pop(0)
                
        current_rsi_high = self.rsi_highs[-1]
        if not self.swing_high_rsi or current_rsi_high != self.swing_high_rsi[-1]:
            self.swing_high_rsi.append(current_rsi_high)
            if len(self.swing_high_rsi) > 2:
                self.swing_high_rsi.pop(0)
        
        # Entry logic 🚀
        if (len(self.swing_low_prices) >= 2 and len(self.swing_low_macd) >= 2 and
            len(self.swing_high_prices) >= 2 and len(self.swing_high_rsi) >= 2):
            
            bearish_divergence = (
                self.swing_low_prices[-1] < self.swing_low_prices[-2] and  # Lower price lows
                self.swing_low_macd[-1] > self.swing_low_macd[-2] and      # Higher MACD lows
                self.swing_high_prices[-1] > self.swing_high_prices[-