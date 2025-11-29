I'll fix the code by removing the `backtesting.lib` import and replacing any usage of its functions. Here's the corrected version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import pandas_ta as ta
import numpy as np
from backtesting import Backtest, Strategy

class VolCompressShorts(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    atr_period = 20
    kc_period = 20
    atr_multiplier = 2.5
    
    def init(self):
        # Clean and prepare data
        self.data.columns = [col.strip().lower() for col in self.data.columns]
        self.data = self.data.drop(columns=[col for col in self.data.columns if 'unnamed' in col])
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate indicators using TA-Lib
        self.ema20 = self.I(talib.EMA, self.data.close, timeperiod=self.kc_period)
        self.atr20 = self.I(talib.ATR, self.data.high, self.data.low, self.data.close, timeperiod=self.atr_period)
        self.upper_kc = self.I(lambda: self.ema20 + self.atr_multiplier * self.atr20)
        self.lower_kc = self.I(lambda: self.ema20 - self.atr_multiplier * self.atr20)
        self.kc_width = self.I(lambda: self.upper_kc - self.lower_kc, name='KC_Width')
        
        # Funding rate indicators
        self.funding_30d = self.I(talib.SMA, self.data.funding_rate, timeperiod=30*96)  # 15m intervals
        self.funding_90th = self.I(lambda x: x.expanding().quantile(0.9), self.data.funding_rate)
        self.funding_75th = self.I(lambda x: x.expanding().quantile(0.75), self.data.funding_rate)
        
        # Liquidation move indicators
        self.liquidation_20d = self.I(talib.SMA, self.data.liquidation_move, timeperiod=20*96)
        
        print("🌙 Moon Dev Indicators Initialized! ✨")
        print("🚀 All indicators powered by TA-Lib and pure Python logic 🚀")

    def next(self):
        current_price = self.data.close[-1]
        
        # Entry conditions
        if not self.position:
            funding_condition = self.funding_30d[-1] > self.funding_90th[-1]
            kc_contraction = self.kc_width[-1] < np.min(self.kc_width[-288:-1])  # 3-day contraction
            kc_expansion = self.kc_width[-1] > self.kc_width[-2]
            
            if funding_condition and kc_contraction and kc_expansion:
                # Risk management calculations
                liquidation_move = self.liquidation_20d[-1]
                risk_amount = self.risk_percent * self.equity
                stop_loss = 2 * liquidation_move
                position_size = int(round(risk_amount / stop_loss))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=current_price + stop_loss, 
                             tp=current_price - 1.5*liquidation_move)
                    print(f"🚀 MOON DEV SHORT ENTRY 🚀 Size: {position_size} | SL: {stop_loss:.2f} | Entry: {current_price:.2f}")

        # Exit conditions
        for trade in self.trades:
            if trade.is_short:
                # Funding rate normalization
                if self.data.funding_rate[-1] < self.funding_75th[-1]:
                    trade.close()
                    print("🌑 Funding Normalization Exit! Funding < 75th percentile")
                    
                # Trailing stop logic
                current_profit = trade.entry_price - current_price
                if current_profit > 0:
                    new_sl = trade.entry_price - self.liquidation_20d[-1]
                    if new_sl < trade.sl:
                        trade.sl = new_sl
                        print(f"✨ Trailing SL Updated: {new_sl:.2f} | Moon Dev Protection Active ✨")

if __name__ == "__main__":
    # Load and prepare data
    data = pd.read_csv('/Users/md/Drop