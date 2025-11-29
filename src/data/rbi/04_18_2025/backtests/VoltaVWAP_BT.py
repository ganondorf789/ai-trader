```python
# 🌙 Moon Dev Backtest AI Implementation for VoltaVWAP Strategy

# Required imports
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Strategy, Backtest
from backtesting.lib import crossover

class VoltaVWAP(Strategy):
    # Strategy parameters
    vwap_length = 96  # 24-hour VWAP (15m*96=24h)
    atr_period = 14
    rsi_period = 14
    risk_pct = 0.01  # 1% risk per trade
    use_rsi_filter = True
    volatility_threshold = 0.05  # 5% ATR filter
    
    def init(self):
        # Clean data columns and ensure proper case
        self.data.df.columns = self.data.df.columns.str.strip().str.lower()
        
        # Calculate indicators with proper TA-Lib/pandas_ta implementation
        self.vwap = self.I(ta.vwap,
                          high=self.data.High,
                          low=self.data.Low,
                          close=self.data.Close,
                          volume=self.data.Volume,
                          length=self.vwap_length)
        
        self.atr = self.I(talib.ATR,
                         self.data.High,
                         self.data.Low,
                         self.data.Close,
                         self.atr_period)
        
        self.rsi = self.I(talib.RSI,
                         self.data.Close,
                         self.rsi_period)
        
        # Track trade metadata
        self.entry_bar = 0
        self.highest_high = 0
        self.lowest_low = 0

    def next(self):
        current_close = self.data.Close[-1]
        current_vwap = self.vwap[-1] if len(self.vwap) > 0 else None
        current_atr = self.atr[-1] if len(self.atr) > 0 else None
        current_rsi = self.rsi[-1] if len(self.rsi) > 0 else None
        
        if not current_vwap or not current_atr:
            return  # Skip if indicators not ready
        
        # 🌙 Calculate volatility-adjusted bands
        upper_band = current_vwap + 2 * current_atr
        lower_band = current_vwap - 2 * current_atr
        
        # 🚨 Volatility filter check
        atr_pct = current_atr / current_close
        if atr_pct > self.volatility_threshold:
            print(f"🌪️ Moon Dev Volatility Alert! {atr_pct:.2%} > {self.volatility_threshold:.0%} - Trading Paused")
            return

        if not self.position:
            # Long entry logic
            if current_close < lower_band:
                if not self.use_rsi_filter or current_rsi < 30:
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    stop_distance = 2 * current_atr
                    
                    if stop_distance == 0:
                        print("⚠️ Zero stop distance detected!")
                        return
                        
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size <= 0:
                        print(f"🌙 Moonshot Aborted! Position size too small: {position_size}")
                        return
                        
                    # Execute long entry with risk management
                    entry_price = current_close
                    sl_price = entry_price - 3 * current_atr
                    self.buy(size=position_size, sl=sl_price)
                    self.entry_bar = len(self.data) - 1
                    self.highest_high = entry_price
                    print(f"🚀 Moon Dev LONG Launch! | Size: {position_size} | Entry: {entry_price} | SL: {sl_price}")

            # Short entry logic        
            elif current_close > upper_band:
                if not self.use_rsi_filter or current_rsi > 70:
                    equity = self.equity
                    risk_amount = equity * self.risk_pct
                    stop_distance = 2 * current_atr
                    
                    if stop_distance == 0:
                        return
                        
                    position_size = int(round(risk_amount / stop_distance))
                    if position_size <= 0:
                        return
                        
                    entry_price = current_close
                    sl_price = entry_price + 3 * current_atr
                    self.sell(size=position_size, sl=sl_price)
                    self.entry_bar = len(self.data) - 1
                    self.lowest_low = entry_price
                    print(f"🌒 Moon Dev SHORT Launch! | Size: {position_size} | Entry: {entry_price} | SL: {sl_price}")

        else:
            # Position management