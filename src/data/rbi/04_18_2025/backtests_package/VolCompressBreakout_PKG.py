Here's the fixed code with all backtesting.lib references removed and replaced with proper Moon Dev implementations:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Backtest, Strategy

class VolCompressBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    atr_period = 14
    bb_period = 20
    bb_dev = 2
    rsi_period = 14
    bb_compress_lookback = 126  # 6-month lookback for volatility compression
    
    def init(self):
        # Calculate volatility components
        def compute_bb_width(close):
            upper, middle, lower = talib.BBANDS(close, 
                                                timeperiod=self.bb_period,
                                                nbdevup=self.bb_dev,
                                                nbdevdn=self.bb_dev)
            return (upper - lower) / middle
        
        self.bb_width = self.I(compute_bb_width, self.data.Close, name='BB_WIDTH')
        self.bb_width_min = self.I(talib.MIN, self.bb_width, timeperiod=self.bb_compress_lookback, name='BB_WIDTH_MIN')
        
        # Momentum components
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=self.rsi_period, name='RSI')
        
        # Risk management components
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 
                         timeperiod=self.atr_period, name='ATR')
        
        print("🌙 MOON DEV INIT COMPLETE 🌙 | Indicators Ready for Launch! 🚀")

    def next(self):
        # Skip initial warmup period
        if len(self.data) < max(self.bb_compress_lookback, self.atr_period, self.bb_period) + 1:
            return
        
        # Get indicator values
        current_bb_width = self.bb_width[-1]
        current_bb_min = self.bb_width_min[-1]
        prev_rsi = self.rsi[-2]
        current_rsi = self.rsi[-1]
        current_atr = self.atr[-1]

        # Entry Logic 🌙✨
        if not self.position:
            # Volatility compression condition
            bb_compress = current_bb_width <= current_bb_min
            
            # Momentum confirmation - replaced crossover with manual check
            rsi_confirm = (self.rsi[-4] < 50 and self.rsi[-3] < 50 and 
                           self.rsi[-2] < 50 and self.rsi[-1] > 50)  # Bullish crossover equivalent
            
            if bb_compress and rsi_confirm:
                # Calculate position size with Moon Dev risk management 🌙
                entry_price = self.data.Open[-1]
                stop_loss = entry_price - 1.5 * current_atr
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    position_size = min(position_size, int(self.equity // entry_price))
                    
                    if position_size > 0:
                        self.buy(size=position_size, 
                                sl=stop_loss,
                                tag={'entry_bb_width': current_bb_width})
                        print(f"\n🚀 MOON DEV LAUNCH 🚀 | LONG {position_size} @ {entry_price:.2f}")
                        print(f"   ░ SL: {stop_loss:.2f} | ATR: {current_atr:.2f}")
                        print(f"   ░ RSI: {current_rsi:.1f} | BB Width: {current_bb_width:.4f}")

        # Exit Logic ✨🌙
        for trade in self.trades:
            if trade.is_long:
                entry_width = trade.tag['entry_bb_width']
                volatility_expansion = current_bb_width >= 2 * entry_width
                
                if volatility_expansion:
                    self.sell(trade)
                    print(f"\n🎯 MOON DEV TARGET REACHED 🎯 | Profit: {trade.pl:.2f}")
                    print(f"   ░ Exit BB Width: {current_bb_width:.4f} vs Entry: {entry_width:.4f}")

# Data preparation moon magic 🌙
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path,