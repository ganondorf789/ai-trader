```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class SqueezeAccelero(Strategy):
    def init(self):
        # 🌙 Initialize indicators with Moon Dev magic!
        close = self.data.Close
        
        # 🎢 Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        self.upper_band = self.I(lambda: upper, name='UpperBB')
        self.middle_band = self.I(lambda: middle, name='MiddleBB')
        self.lower_band = self.I(lambda: lower, name='LowerBB')
        
        # 📈 Volume Momentum
        self.volume_roc = self.I(talib.ROC, self.data.Volume, timeperiod=3, name='VolumeROC')
        
        # 🌪️ Trend Strength Filter
        self.adx = self.I(talib.ADX, self.data.High, self.data.Low, close, timeperiod=14, name='ADX')
        
        # ⚖️ Swing Points for Risk Management
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=20, name='SwingHigh')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='SwingLow')
        
        # 🛡️ ATR for Volatility Stop
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, close, timeperiod=14, name='ATR')

    def next(self):
        # 🌌 Moon Dev Signal Check
        price = self.data.Close[-1]
        upper = self.upper_band[-1]
        lower = self.lower_band[-1]
        volume_roc = self.volume_roc[-1]
        adx = self.adx[-1]
        
        # 🚫 No open positions
        if not self.position:
            # 🌕 Long Entry: Price > Upper BB + Volume Surge + Strong Trend
            if price > upper and volume_roc > 20 and adx > 25:
                sl_price = self.swing_low[-1] - 1.5 * self.atr[-1]
                self.calculate_risk('LONG', price, sl_price)
                
            # 🌑 Short Entry: Price < Lower BB + Volume Surge + Strong Trend
            elif price < lower and volume_roc > 20 and adx > 25:
                sl_price = self.swing_high[-1] + 1.5 * self.atr[-1]
                self.calculate_risk('SHORT', price, sl_price)
        
        # 🏃♂️ Open Position Management
        else:
            # ✨ Long Exit: Price crosses below Middle BB
            if self.position.is_long and crossunder(self.data.Close, self.middle_band):
                self.position.close()
                print(f"✨🌙 LONG EXIT | Price: {price:.2f} | Middle Band: {self.middle_band[-1]:.2f}")
                
            # ✨ Short Exit: Price crosses above Middle BB
            elif self.position.is_short and crossover(self.data.Close, self.middle_band):
                self.position.close()
                print(f"✨🌙 SHORT EXIT | Price: {price:.2f} | Middle Band: {self.middle_band[-1]:.2f}")

    def calculate_risk(self, direction, entry_price, sl_price):
        # 🌙 Risk Management Protocol
        risk_pct = 0.01  # 1% risk per trade
        risk_amount = self.equity * risk_pct
        
        if direction == 'LONG':
            price_diff = entry_price - sl_price
            if price_diff <= 0:
                print("🌙🚫 Invalid LONG stop loss! Moon Dev shields engaged!")
                return
        else:
            price_diff = sl_price - entry_price
            if price_diff <= 0:
                print("🌙🚫 Invalid SHORT stop loss! Moon Dev shields engaged!")
                return
        
        position_size = risk_amount / price_diff
        position_size = int(round(position_size))
        
        if position_size == 0:
            print(f"🌙💤 Zero size calculated! {direction} trade skipped")
            return
            
        # 🚀 Execute Trade with Moon Precision
        if direction == 'LONG':
            self.buy(size=position_size, sl=sl_price)
            print(f"🚀🌙 LONG LAUNCH | Size: {position_size} | Entry: {entry_price:.2f