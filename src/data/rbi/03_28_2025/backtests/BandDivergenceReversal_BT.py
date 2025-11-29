```python
import numpy as np
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

class BandDivergenceReversal(Strategy):
    def init(self):
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(self.data.Close, timeperiod=20, nbdevup=2, nbdevdn=2)
        self.upper_band = self.I(lambda: pd.Series(upper, index=self.data.index), name='Upper BB')
        self.middle_band = self.I(lambda: pd.Series(middle, index=self.data.index), name='Middle BB')
        self.lower_band = self.I(lambda: pd.Series(lower, index=self.data.index), name='Lower BB')
        
        # Calculate OBV
        obv_values = talib.OBV(self.data.Close, self.data.Volume)
        self.obv = self.I(lambda: pd.Series(obv_values, index=self.data.index), name='OBV')
        
        print("🌙✨ Moon Dev's BandDivergenceReversal strategy initialized! Ready to launch! 🚀")

    def next(self):
        if len(self.data) < 25:  # Ensure enough data for calculations
            return

        current_close = self.data.Close[-1]
        upper_band = self.upper_band[-1]
        lower_band = self.lower_band[-1]
        middle_band = self.middle_band[-1]

        # Long Entry Logic
        if not self.position.is_long and current_close <= lower_band:
            lookback = 5
            if len(self.data) > lookback + 1:
                price_lows = self.data.Low[-lookback-1:-1]
                obv_values = self.obv[-lookback-1:-1]
                
                if (current_close < price_lows.min() and 
                    self.obv[-1] > obv_values.max()):
                    self.calculate_risk('long')

        # Short Entry Logic
        if not self.position.is_short and current_close >= upper_band:
            lookback = 5
            if len(self.data) > lookback + 1:
                price_highs = self.data.High[-lookback-1:-1]
                obv_values = self.obv[-lookback-1:-1]
                
                if (current_close > price_highs.max() and 
                    self.obv[-1] < obv_values.min()):
                    self.calculate_risk('short')

        # Exit Logic
        for trade in self.trades:
            if trade.is_long and current_close >= middle_band:
                trade.close()
                print(f"🌙💰 Long exit at {current_close:.2f}! Riding middle band to profits! ✨")
            elif trade.is_short and current_close <= middle_band:
                trade.close()
                print(f"🌙🌟 Short exit at {current_close:.2f}! Surfing middle band waves! 🌊")

    def calculate_risk(self, trade_type):
        risk_pct = 0.01
        entry_price = self.data.Close[-1]
        
        if trade_type == 'long':
            swing_low = talib.MIN(self.data.Low, 20)[-1]
            stop_loss = min(swing_low, self.lower_band[-1]) * 0.995
            risk_per_share = entry_price - stop_loss
        else:
            swing_high = talib.MAX(self.data.High, 20)[-1]
            stop_loss = max(swing_high, self.upper_band[-1]) * 1.005
            risk_per_share = stop_loss - entry_price

        if risk_per_share <= 0:
            print(f"🌙⚠️ Risk too low for {trade_type} entry at {entry_price:.2f}")
            return

        position_size = int(round((self.equity * risk_pct) / risk_per_share))
        if position_size == 0:
            return

        if trade_type == 'long':
            self.buy(size=position_size, sl=stop_loss)
            print(f"🌙🚀 LONG signal! Buying {position_size} units at {entry_price:.2f} ✨")
        else:
            self.sell(size=position_size, sl=stop_loss)
            print(f"🌙🌊 SHORT signal! Selling {position_size} units at {entry_price:.2f} 🌊")

# Data preprocessing
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv