Here's the fixed code with all backtesting.lib references removed and proper Moon Dev themed implementations:

```python
# 🌙 Moon Dev's VolSurge Breakout Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
import talib

def calculate_volume_percentile(volume, window, percentile):
    """Calculate rolling volume percentile with Moon Dev magic ✨"""
    return volume.rolling(window).quantile(percentile/100)

class VolSurgeBreakout(Strategy):
    # Strategy parameters
    vama_period = 20
    volume_lookback = 50
    volume_percentile = 90
    atr_period = 14
    atr_multiple = 2
    risk_per_trade = 0.01  # 1% risk per trade
    exit_on_vama_break = True
    
    def init(self):
        """Moon Dev Indicator Initialization 🌙"""
        # Volatility-Adjusted Moving Average (KAMA)
        self.vama = self.I(talib.KAMA, self.data.Close, self.vama_period, name='VAMA')
        
        # Volume Surge Indicator
        self.vol_percentile = self.I(calculate_volume_percentile, 
                                    self.data.Volume, 
                                    self.volume_lookback,
                                    self.volume_percentile,
                                    name='Volume_Percentile')
        
        # ATR for risk management
        self.atr = self.I(talib.ATR,
                         self.data.High,
                         self.data.Low,
                         self.data.Close,
                         self.atr_period,
                         name='ATR')
        
        print("🌕 Moon Dev Indicators Activated! VAMA, Volume Percentile, ATR Ready 🚀")

    def next(self):
        """Moon Dev Trading Logic Cycle 🌙➡️🌕"""
        # Skip first bars where indicators aren't calculated
        if len(self.data) < max(self.vama_period, self.volume_lookback, self.atr_period):
            return
        
        # Moon Dev Debug Prints
        current_close = self.data.Close[-1]
        if len(self.data) % 100 == 0:
            print(f"\n🌙 Moon Dev Status Update 🌙\n"
                  f"Bar: {len(self.data)} | Close: {current_close:.2f}\n"
                  f"VAMA: {self.vama[-1]:.2f} | Vol%: {self.vol_percentile[-1]:.2f}\n"
                  f"ATR: {self.atr[-1]:.2f} | Equity: {self.equity:.2f}\n"
                  "✨"*20)
        
        # Entry Logic 🌟
        if not self.position:
            # Check breakout conditions
            close_above_vama = current_close > self.vama[-1]
            volume_surge = self.data.Volume[-1] > self.vol_percentile[-1]
            vama_up_trend = self.vama[-1] > self.vama[-2]  # Replaced crossover with direct comparison
            
            if close_above_vama and volume_surge and vama_up_trend:
                # Calculate position size with Moon Dev precision 🌙
                risk_amount = self.risk_per_trade * self.equity
                risk_per_share = self.atr[-1] * self.atr_multiple
                
                if risk_per_share == 0:
                    print("⚠️ Moon Dev Warning: Zero risk detected, skipping trade!")
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size)
                    print(f"\n🚀 MOON DEV BREAKOUT ALERT 🚀\n"
                          f"Entry Price: {self.data.Close[-1]:.2f}\n"
                          f"Position Size: {position_size} units\n"
                          f"Risk/Share: {risk_per_share:.2f} | Equity: {self.equity:.2f}\n"
                          "🌙"*30)
                    
                    # Initialize trailing stop
                    self.trade_high = self.data.High[-1]
                    self.trail_stop = self.trade_high - (self.atr[-1] * self.atr_multiple)
        
        # Exit Logic 🌑
        else:
            # Update trailing stop
            self.trade_high = max(self.trade_high, self.data.High[-1])
            new_stop = self.trade_high - (self.atr[-1] * self.atr_multiple)
            self.trail_stop = max(self.trail_stop, new_stop)
            
            #