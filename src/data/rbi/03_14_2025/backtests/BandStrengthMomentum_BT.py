Here's the complete backtesting implementation for the BandStrength Momentum strategy with Moon Dev themed logging and risk management:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover, crossunder

# Moon Dev Data Preparation Functions 🌙
def prepare_data(filepath):
    # Load and clean data
    data = pd.read_csv(filepath)
    data.columns = data.columns.str.strip().str.lower()
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
    
    # Proper column mapping
    data.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'datetime': 'DateTime'
    }, inplace=True)
    
    # Convert and set index
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index('DateTime', inplace=True)
    return data

# Moon Dev Strategy Implementation 🚀
class BandStrengthMomentum(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    rsi_period = 14
    bb_period = 20
    bb_dev = 2
    
    def init(self):
        # Calculate indicators using TA-Lib through self.I()
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period, name='RSI')
        
        # Bollinger Bands components
        def bb_upper(close):
            u, _, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return u
        def bb_middle(close):
            _, m, _ = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return m
        def bb_lower(close):
            _, _, l = talib.BBANDS(close, self.bb_period, self.bb_dev, self.bb_dev)
            return l
        
        self.bb_upper = self.I(bb_upper, self.data.Close, name='BB_upper')
        self.bb_middle = self.I(bb_middle, self.data.Close, name='BB_middle')
        self.bb_lower = self.I(bb_lower, self.data.Close, name='BB_lower')
        
        # Candlestick patterns
        self.engulfing = self.I(talib.CDLENGULFING, self.data.Open, self.data.High, 
                               self.data.Low, self.data.Close, name='Engulfing')
        self.hammer = self.I(talib.CDLHAMMER, self.data.Open, self.data.High,
                            self.data.Low, self.data.Close, name='Hammer')
        self.shooting_star = self.I(talib.CDLSHOOTINGSTAR, self.data.Open, self.data.High,
                                   self.data.Low, self.data.Close, name='ShootingStar')

    def next(self):
        current_close = self.data.Close[-1]
        current_rsi = self.rsi[-1]
        
        # Moon Dev Position Management 🌙
        for trade in self.trades:
            if trade.is_long and current_rsi < 40:
                trade.close()
                print(f"🌑 Moon Dev Exit! RSI fell below 40 at {current_rsi:.1f}")
            elif trade.is_short and current_rsi > 60:
                trade.close()
                print(f"🌑 Moon Dev Exit! RSI rose above 60 at {current_rsi:.1f}")
        
        if self.position:
            return  # Don't open new positions if already in one
            
        # Moon Dev Long Entry Conditions 🌟
        long_trigger = (
            crossover(self.data.Close[-2:], self.bb_lower[-2:]) and
            current_rsi > 50 and
            (self.engulfing[-1] == 100 or self.hammer[-1] == 100)
        )
        
        # Moon Dev Short Entry Conditions 🌪️
        short_trigger = (
            crossunder(self.data.Close[-2:], self.bb_upper[-2:]) and
            current_rsi < 50 and
            (self.engulfing[-1] == -100 or self.shooting_star[-1] == -100)
        )
        
        # Execute trades with Moon Dev Risk Management 🛡️
        if long_trigger:
            entry_price = self.data.Close[-1]
            sl_price = self.bb_lower[-1] * 0.995
            tp