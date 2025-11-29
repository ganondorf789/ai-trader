Here's the debugged backtest code with proper position sizing and technical fixes while preserving the original strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

class LiquidationCluster(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade
    funding_rate_threshold = -0.001
    oi_drop_threshold = -10  # -10%
    stabilization_period = 2  # 2 candles
    exit_funding_rate = -0.0002
    max_trade_duration = 48  # 12 hours (48*15m)

    def init(self):
        # Data preprocessing checks
        if 'funding_rate' not in self.data.df.columns or 'open_interest' not in self.data.df.columns:
            raise ValueError("🌙 MOON ERROR: Missing required columns - funding_rate and open_interest must be in DataFrame")
            
        # Indicator calculations using talib
        self.oi_max = self.I(talib.MAX, self.data.df['open_interest'], timeperiod=4)
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=4)
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, timeperiod=14)
        
        # Track strategy state
        self.entry_triggered = False
        self.stabilization_count = 0
        self.lowest_oi = None
        self.entry_bar = 0  # Initialize entry_bar

    def next(self):
        # Skip if not enough data
        if len(self.data) < 4:
            return

        # Get current values
        current_funding = self.data.df['funding_rate'][-1]
        current_oi = self.data.df['open_interest'][-1]
        current_oi_max = self.oi_max[-1]
        oi_pct_drop = ((current_oi - current_oi_max) / current_oi_max * 100) if current_oi_max != 0 else 0

        # Entry logic
        if not self.position and not self.entry_triggered:
            if current_funding < self.funding_rate_threshold and oi_pct_drop < self.oi_drop_threshold:
                print(f"🌙 MOON ALERT! Liquidation cluster detected 🌑")
                print(f"   Funding Rate: {current_funding*100:.4f}% | OI Drop: {oi_pct_drop:.2f}%")
                self.entry_triggered = True
                self.lowest_oi = current_oi
                self.stabilization_count = 0

        # Stabilization monitoring
        if self.entry_triggered and not self.position:
            if current_oi < self.lowest_oi:
                self.lowest_oi = current_oi
                self.stabilization_count = 0
            else:
                self.stabilization_count += 1

            print(f"✨ Stabilization progress: {self.stabilization_count}/{self.stabilization_period} candles")

            if self.stabilization_count >= self.stabilization_period:
                # Calculate position size with risk management
                equity = self.equity
                entry_price = self.data.Close[-1]
                atr_value = self.atr[-1]
                stop_loss = min(self.swing_low[-1], entry_price - 1.5 * atr_value)
                
                risk_amount = equity * self.risk_per_trade
                risk_per_share = entry_price - stop_loss
                
                if risk_per_share <= 0:
                    print("⚠️ MOON WARNING: Aborting trade - invalid stop loss level")
                    self.entry_triggered = False
                    return
                
                position_size = int(round(risk_amount / risk_per_share))
                if position_size <= 0:
                    print("⚠️ MOON WARNING: Position size too small - skipping trade")
                    self.entry_triggered = False
                    return
                
                print(f"🚀 LAUNCHING POSITION: Size {position_size} units @ {entry_price:.2f}")
                
                self.buy(size=position_size, sl=stop_loss)
                self.entry_bar = len(self.data)
                self.entry_triggered = False

        # Exit logic
        if self.position:
            # Funding rate normalization
            if current_funding >= self.exit_funding_rate:
                print(f"🎯 TARGET HIT: Funding rate normalized to {current_funding*100:.4f}%")
                self.position.close