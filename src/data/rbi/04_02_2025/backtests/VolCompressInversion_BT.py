```python
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

class VolCompressInversion(Strategy):
    bbw_lookback = 6 * 30 * 24 * 4  # 6 months in 15m intervals
    risk_percent = 0.01
    max_hold_bars = 10 * 24 * 4  # 10 days in 15m bars

    def init(self):
        # Calculate indicators using TA-Lib through self.I()
        # Bollinger Bands and Width
        def bb_bands(close):
            upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
            return (upper - lower) / middle, upper, lower

        self.bbw, self.bb_upper, self.bb_lower = self.I(bb_bands, self.data.Close, name=['BBW', 'BB_Upper', 'BB_Lower'])
        
        # VIX Term Structure inversion
        self.vix_front = self.I(talib.SMA, self.data.vix_front, timeperiod=1)
        self.vix_back = self.I(talib.SMA, self.data.vix_back, timeperiod=1)
        
        # 20-day SMA
        self.ma20 = self.I(talib.SMA, self.data.Close, 20)
        
        # Put/Call Ratio MA
        self.putcall_ma10 = self.I(talib.SMA, self.data.put_call_ratio, 10)
        
        # Volatility metrics
        self.atr14 = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.volume_ma20 = self.I(talib.SMA, self.data.Volume, 20)
        
        # Track trade parameters
        self.entry_bbw = None
        self.entry_bar = None

    def next(self):
        # Moon Dev progress tracking
        if len(self.data) % 1000 == 0:
            print(f"🌙 Moon Dev Tracking: Processing bar {len(self.data)} | Price: {self.data.Close[-1]:.2f} ✨")
            
        # Entry logic
        if not self.position:
            if (self.bbw[-1] == talib.MIN(self.bbw, self.bbw_lookback)[-1] and  # 6m BBW low
                self.vix_front[-1] > self.vix_back[-1] and  # VIX inversion
                self.data.Close[-1] < self.bb_lower[-1]):  # Price below BB
                
                # Risk management calculations
                equity = self.equity
                risk_amount = equity * self.risk_percent
                atr_value = self.atr14[-1]
                stop_loss = self.data.Close[-1] - 2 * atr_value
                risk_per_share = self.data.Close[-1] - stop_loss
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    if position_size > 0:
                        self.buy(size=position_size, sl=stop_loss)
                        self.entry_bbw = self.bbw[-1]
                        self.entry_bar = len(self.data)
                        print(f"🚀🌕 MOON DEV ENTRY: Short {position_size} units at {self.data.Close[-1]:.2f} | SL: {stop_loss:.2f} 🔥")

        # Exit logic
        else:
            current_bar = len(self.data)
            bars_held = current_bar - self.entry_bar
            
            # Main exit conditions
            exit_price = self.data.Close[-1]
            exit_conditions = [
                (exit_price > self.ma20[-1] and self.data.Volume[-1] > self.volume_ma20[-1], 
                 f"🌙💎 MA20 Breakout Exit at {exit_price:.2f}"),
                (self.data.put_call_ratio[-1] > 1.2 * self.putcall_ma10[-1], 
                 f"🌙📉 Put/Call Spike Exit at {exit_price:.2f}"),
                (self.bbw[-1] >= 2 * self.entry_bbw, 
                 f"🌪️🔥 Volatility Expansion Exit at {exit_price:.2f}"),
                (bars_held >= self.max_hold_bars, 
                 f"⏳🕒 Time Exit at {exit_price:.2f} after {