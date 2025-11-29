Here's the fixed code with all backtesting.lib imports removed and proper indicator implementations:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Moon Dev Data Preparation 🌙
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and prepare columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

# Moon Dev VolatilityHarmonic Strategy ✨
class VolatilityHarmonic(Strategy):
    window_3m = 3 * 30 * 24 * 4  # 3 months in 15m intervals
    
    def init(self):
        # VIX-VVIX Spread Calculation 🌊
        self.vix = self.data.df['vix']
        self.vvix = self.data.df['vvix']
        self.spread = self.I(lambda: self.vix - self.vvix, name='VIX-VVIX Spread')
        
        # Spread Z-Score Calculation 📈
        self.spread_mean = self.I(talib.SMA, self.spread, timeperiod=self.window_3m, name='Spread Mean')
        self.spread_std = self.I(talib.STDDEV, self.spread, timeperiod=self.window_3m, name='Spread Std')
        self.z_spread = self.I(lambda: (self.spread - self.spread_mean) / self.spread_std, 
                              name='Z-Spread')

        # S&P 500 Bollinger Band Width 📉
        self.spx_close = self.data.df['sp500_close']
        self.bb_upper, self.bb_mid, self.bb_lower = self.I(talib.BBANDS, self.spx_close, 
                                                          timeperiod=20, nbdevup=2, nbdevdn=2, 
                                                          matype=0, name=['BB Upper','BB Mid','BB Lower'])
        self.bb_width = self.I(lambda: self.bb_upper - self.bb_lower, name='BB Width')
        
        # BB Width Z-Score Calculation 📊
        self.bbw_mean = self.I(talib.SMA, self.bb_width, timeperiod=self.window_3m, name='BBW Mean')
        self.bbw_std = self.I(talib.STDDEV, self.bb_width, timeperiod=self.window_3m, name='BBW Std')
        self.z_bbw = self.I(lambda: (self.bb_width - self.bbw_mean) / self.bbw_std, 
                           name='Z-BBW')

        # ATR for Stop Loss Calculation 🛑
        self.spx_high = self.data.df['sp500_high']
        self.spx_low = self.data.df['sp500_low']
        self.atr = self.I(talib.ATR, self.spx_high, self.spx_low, self.spx_close, 
                         timeperiod=14, name='ATR')

        print("🌙✨ Moon Dev Indicators Initialized! Ready for Launch! 🚀")

    def next(self):
        current_close = self.data.Close[-1]
        z_spread = self.z_spread[-1]
        z_bbw = self.z_bbw[-1]
        atr = self.atr[-1]

        # Moon Dev Position Management 🌙
        if self.position:
            if self.position.is_long and (z_spread >= 0 or z_bbw >= 0):
                self.position.close()
                print(f"🌕💎 Moon Dev Profit Taken! Closed long at {current_close:.2f}")
                
            elif self.position.is_short and (z_spread <= 0 or z_bbw <= 0):
                self.position.close()
                print(f"🌕💎 Moon Dev Profit Taken! Closed short at {current_close:.2f}")
        else:
            # Long Entry: Volatility Expansion Signal 🌌
            if z_spread < -2 and z_bbw < -2:
                risk_amount = self.equity * 0.01
                stop_distance = 1.5 * atr
                position_size = int(round(risk_amount / stop