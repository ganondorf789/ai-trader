```python
# 🌙 Moon Dev's VolSqueezeBreakout Backtest 🌙
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
import pandas_ta as ta
import numpy as np

class VolSqueezeBreakout(Strategy):
    # Strategy parameters
    risk_per_trade = 0.01  # 1% risk per trade
    max_positions = 5
    bb_period = 20
    bb_dev = 2
    lookback_period = 100
    volume_multiplier = 1.5
    percentile_threshold = 20
    
    def init(self):
        # 🌙 Indicator Calculation Phase 🌙
        close = self.data.Close
        volume = self.data.Volume
        
        # Calculate Bollinger Bands
        upper, middle, lower = talib.BBANDS(close, 
                                           timeperiod=self.bb_period,
                                           nbdevup=self.bb_dev,
                                           nbdevdn=self.bb_dev)
        
        # Calculate Bollinger Band Width (BBW)
        bbw = (upper - lower) / middle
        self.bbw = self.I(lambda: bbw, name='BBW')
        
        # Calculate BBW percentile using pandas_ta
        self.bbw_percentile = self.I(lambda: ta.quantile(bbw, length=self.lookback_period, q=self.percentile_threshold/100),
                                     name='BBW_Percentile')
        
        # Calculate volume average
        self.vol_ma = self.I(talib.SMA, volume, timeperiod=20, name='Volume_MA')
        
        # Store bands as indicators
        self.upper_band = self.I(lambda: upper, name='UpperBB')
        self.lower_band = self.I(lambda: lower, name='LowerBB')
        
        # Track maximum equity for drawdown calculation
        self.max_equity = self._broker.starting_cash
        print("✨ Moon Dev Indicators Initialized! Ready for launch! 🚀")

    def next(self):
        # 🌙 Trading Logic Execution 🌙
        current_close = self.data.Close[-1]
        current_volume = self.data.Volume[-1]
        
        # Update max equity tracking
        self.max_equity = max(self.max_equity, self.equity)
        current_drawdown = (self.max_equity - self.equity) / self.max_equity
        allow_new_entries = current_drawdown <= 0.05
        
        # Check exit conditions first
        for trade in self.trades:
            # Time-based exit
            if len(self.data) - trade.entry_bar >= 5:
                trade.close()
                print(f"🌙 Lunar Time Exit! 🕒 Closed position at {current_close}")
            
            # Trailing stop logic
            if trade.is_long:
                risk = trade.entry_price - trade.sl
                if self.data.High[-1] >= trade.entry_price + risk:
                    trade.sl = max(trade.sl, trade.entry_price)
            else:
                risk = trade.sl - trade.entry_price
                if self.data.Low[-1] <= trade.entry_price - risk:
                    trade.sl = min(trade.sl, trade.entry_price)
        
        # Entry logic
        if len(self.trades) < self.max_positions and allow_new_entries:
            bbw_val = self.bbw[-1]
            bbw_pct = self.bbw_percentile[-1]
            vol_ma = self.vol_ma[-1]
            upper_band = self.upper_band[-1]
            lower_band = self.lower_band[-1]
            
            # Long entry conditions
            if (bbw_val < bbw_pct and
                current_close > upper_band and
                current_volume > self.volume_multiplier * vol_ma):
                
                # Position sizing
                risk_amount = self.equity * self.risk_per_trade
                stop_price = lower_band
                risk_per_share = current_close - stop_price
                
                if risk_per_share > 0:
                    position_size = int(round(risk_amount / risk_per_share))
                    tp_price = current_close + 2 * risk_per_share
                    
                    self.buy(size=position_size,
                            sl=stop_price,
                            tp=tp_price,
                            tag="VolSqueezeLong")
                    print(f"🚀 MOONSHOT LONG! 🌕 Size: {position_size} | Entry: {current_close} | Stop: {stop_price} | Target: {tp_price}")
            
            # Short entry conditions (mirror logic)