```python
# 🌙 Moon Dev's Liquidity Reversion Backtest 🌙
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class LiquidityReversion(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    bandwidth_threshold = 0.05  # Entry bandwidth threshold
    bandwidth_exit_threshold = 0.15  # Exit bandwidth threshold
    swing_period = 20  # Swing high/low period
    sl_multiplier = 1.5  # ATR multiplier for SL

    def init(self):
        # 🌙 Bollinger Band Components
        self.bb_middle = self.I(talib.SMA, self.data.Close, 20, name='BB_Middle')
        self.bb_stddev = self.I(talib.STDDEV, self.data.Close, 20, name='BB_StdDev')
        self.bb_upper = self.I(lambda: self.bb_middle + 2*self.bb_stddev, name='BB_Upper')
        self.bb_lower = self.I(lambda: self.bb_middle - 2*self.bb_stddev, name='BB_Lower')
        self.bb_bandwidth = self.I(lambda: (self.bb_upper - self.bb_lower)/self.bb_middle, name='BB_Bandwidth')
        
        # 🚀 Liquidation Zones (Swing Highs/Lows)
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period, name='Swing_High')
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period, name='Swing_Low')
        
        # ✨ Volatility Filter (ATR)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR')

    def next(self):
        current_close = self.data.Close[-1]
        current_bandwidth = self.bb_bandwidth[-1]
        swing_high = self.swing_high[-1]
        swing_low = self.swing_low[-1]
        atr_value = self.atr[-1]

        # 🌙 MOON DEV DEBUG PRINT 🌙
        print(f"🌙 Price: {current_close:.2f} | Bandwidth: {current_bandwidth:.4f} | Swing High: {swing_high:.2f} | Swing Low: {swing_low:.2f}")

        # 💰 Risk Management Calculations
        equity = self.equity
        position_size = 0

        # 🛑 Check for entries
        if not self.position:
            # Long Entry Condition
            if (current_bandwidth < self.bandwidth_threshold and
                current_close <= swing_low):
                
                sl_price = swing_low - (atr_value * self.sl_multiplier)
                risk_amount = equity * self.risk_percent
                risk_per_share = current_close - sl_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.buy(size=position_size, sl=sl_price, 
                            tp=swing_high, 
                            tag="🌙 LONG ENTRY")
                    print(f"🚀🌙 MOON DEV LONG ENTRY 🚀🌙\n"
                          f"Entry: {current_close:.2f} | "
                          f"Size: {position_size} | "
                          f"SL: {sl_price:.2f} | "
                          f"TP: {swing_high:.2f}")

            # Short Entry Condition    
            elif (current_bandwidth < self.bandwidth_threshold and
                  current_close >= swing_high):
                
                sl_price = swing_high + (atr_value * self.sl_multiplier)
                risk_amount = equity * self.risk_percent
                risk_per_share = sl_price - current_close
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    self.sell(size=position_size, sl=sl_price, 
                             tp=swing_low, 
                             tag="🌙 SHORT ENTRY")
                    print(f"🚀🌙 MOON DEV SHORT ENTRY 🚀🌙\n"
                          f"Entry: {current_close:.2f} | "
                          f"Size: {position_size} | "
                          f"SL: {sl_price:.2f} | "
                          f"TP: {swing