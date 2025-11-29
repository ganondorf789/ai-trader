Here's the fully debugged and Moon Dev-optimized version of your backtest code with all technical issues fixed while preserving the original strategy logic:

```python
import pandas as pd
import talib
import numpy as np
from backtesting import Backtest, Strategy

# Moon-powered data loader 🌙✨
def load_data(data_path):
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    
    # Lunar data cleansing
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Moon Dev cosmic column alignment ✨
    data = data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'funding_rate': 'Funding_Rate'
    })
    
    return data

class VolSqueezeLiquidity(Strategy):
    # Moon-tuned parameters 🌕
    bb_period = 20
    bb_dev = 2
    rsi_period = 14
    squeeze_threshold = 0.5
    volume_multiplier = 1.5
    risk_per_trade = 0.02  # 2% risk per trade
    profit_multiplier = 1.5

    def init(self):
        # Cosmic indicator initialization 🌌
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = self.I(talib.BBANDS, 
            self.data.Close, 
            timeperiod=self.bb_period,
            nbdevup=self.bb_dev,
            nbdevdn=self.bb_dev,
            matype=0
        )
        
        # RSI for overbought check
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        
        # Volume analysis
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20)
        
        # Squeeze detection
        self.bb_width = self.I(lambda: self.bb_upper - self.bb_lower)
        self.bb_width_avg = self.I(talib.SMA, self.bb_width, 50)

    def next(self):
        # Moonbase security check 🛰️
        if len(self.data) < 50:
            return
            
        # Current lunar market conditions
        price = self.data.Close[-1]
        funding = self.data.Funding_Rate[-1]
        
        # Moon entry checklist 🌙✅
        squeeze_active = self.bb_width[-1] < (self.bb_width_avg[-1] * self.squeeze_threshold)
        funding_negative = funding < -0.01
        price_breakout = price > self.bb_upper[-1]
        volume_surge = self.data.Volume[-1] > (self.volume_sma[-1] * self.volume_multiplier)
        rsi_safe = self.rsi[-1] < 70

        # Moon launch sequence! 🚀🌙
        if not self.position and all([
            squeeze_active,
            funding_negative,
            price_breakout,
            volume_surge,
            rsi_safe
        ]):
            # Calculate moon-sized position with proper rounding 🌕
            risk_amount = self.risk_per_trade * self.equity
            stop_loss = self.data.Low[-1]
            risk_per_unit = price - stop_loss
            
            if risk_per_unit > 0:
                position_size = int(round(risk_amount / risk_per_unit))
                take_profit = price + (self.bb_width[-1] * self.profit_multiplier)
                
                if position_size > 0:
                    self.buy(
                        size=position_size,
                        sl=stop_loss,
                        tp=take_profit
                    )
                    print(f"🚀🌙 MOON LAUNCH DETECTED! Long {position_size} units")
                    print(f"   Entry: {price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
                    print(f"   Current BB Width: {self.bb_width[-1]:.2f} | Volume: {self.data.Volume[-1]:.0f}")

    def notify_trade(self, trade):
        if trade.is_closed:
            profit = trade.pl_pct * 100
            duration = trade.exit_time - trade.entry_time
            print(f"🌑 Trade closed | Profit: {profit:.2f}% | Duration: {duration}")
            if profit >