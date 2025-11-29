Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed indicators:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest

class LiquidityVolatilityBreakout(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    tp_multiplier = 1.5
    volatility_window = 15
    cluster_window = 20  # Swing high/low window for liquidity clusters
    
    def init(self):
        # Clean and prepare data
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        self.data.df = self.data.df.drop(columns=[col for col in self.data.df.columns if 'unnamed' in col])
        
        # Calculate indicators using TA-Lib through self.I()
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        self.volatility_ma = self.I(talib.SMA, self.atr, timeperiod=self.volatility_window)
        self.swing_high = self.I(talib.MAX, self.data.High, timeperiod=self.cluster_window)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=self.cluster_window)
        self.volume_ma = self.I(talib.SMA, self.data.Volume, timeperiod=20)
        
        # Track entry volatility level
        self.entry_volatility = None
        
    def next(self):
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_volume = self.data.Volume[-1]
        
        # Liquidity cluster detection
        upper_cluster = self.swing_high[-1]
        lower_cluster = self.swing_low[-1]
        
        # Volatility conditions
        volatility_spike = self.atr[-1] > 2 * self.volatility_ma[-1]
        volume_spike = current_volume > 2 * self.volume_ma[-1]
        
        # Moon Dev themed debug prints 🌙
        print(f"🌙 Moon Dev Status || Close: {current_close:.2f} | Cluster High: {upper_cluster:.2f} | Cluster Low: {lower_cluster:.2f}")
        print(f"✨ Volatility Check || ATR: {self.atr[-1]:.2f} vs MA: {self.volatility_ma[-1]:.2f}")
        print(f"🚀 Volume Alert || Current: {current_volume:.2f} vs MA: {self.volume_ma[-1]:.2f}")
        
        if not self.position:
            # Long entry conditions
            if (current_close > upper_cluster * 0.98 and 
                volatility_spike and 
                volume_spike):
                
                # Risk management calculations
                sl_price = lower_cluster * 0.99
                risk_per_share = current_close - sl_price
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    print(f"🌟🌟🌟 MOON SHOT LONG ENTRY 🌟🌟🌟 Size: {position_size} | Entry: {current_close:.2f} | SL: {sl_price:.2f}")
                    self.buy(size=position_size, sl=sl_price, 
                            tp=current_close + 2*(current_close - sl_price))
                    self.entry_volatility = self.atr[-1]
            
            # Short entry conditions
            elif (current_close < lower_cluster * 1.02 and 
                  volatility_spike and 
                  volume_spike):
                
                sl_price = upper_cluster * 1.01
                risk_per_share = sl_price - current_close
                risk_amount = self.risk_per_trade * self.equity
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    print(f"🌟🌟🌟 COMET SHORT ENTRY 🌟🌟🌟 Size: {position_size} | Entry: {current_close:.2f} | SL: {sl_price:.2f}")
                    self.sell(size=position_size, sl=sl_price, 
                             tp=current_close - 2*(sl_price - current_close))
                    self.entry_volatility = self.atr[-1]
        else:
            # Volatility exit condition
            if len(self.atr) > 3 and all(self.atr[-i] < self.entry_volatility for