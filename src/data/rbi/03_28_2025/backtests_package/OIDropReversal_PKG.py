I'll fix the backtesting.lib imports and usage in this code while maintaining all the Moon Dev theming. Here's the corrected version:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Backtest, Strategy

# 🌙 Moon Dev Data Preparation Magic
def prepare_data(data_path):
    # Load and clean data
    data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')
    data.columns = data.columns.str.strip().str.lower()
    
    # Drop any unnamed columns
    data = data.drop(columns=[col for col in data.columns if 'unnamed' in col])
    
    # Map columns to backtesting.py format
    column_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }
    data.rename(columns=column_mapping, inplace=True)
    
    # Check for required Open Interest data
    if 'open_interest' not in data.columns:
        raise ValueError("🚨 Moon Dev Alert: Open Interest data not found!")
    
    return data

# 🌙✨ OIDropReversal Strategy Class
class OIDropReversal(Strategy):
    def init(self):
        # 🚀 Moon Dev Indicator Setup
        self.oi_change = self.I(talib.ROC, self.data['Open_Interest'], timeperiod=96, name='OI 24h Change')
        self.price_change = self.I(talib.ROC, self.data.Close, timeperiod=96, name='Price 24h Change')
        self.rsi = self.I(talib.RSI, self.data.Close, timeperiod=14, name='RSI 14')
        self.sma10 = self.I(talib.SMA, self.data.Close, timeperiod=10, name='SMA 10')
        self.swing_low = self.I(talib.MIN, self.data.Low, timeperiod=20, name='Swing Low')

    def next(self):
        # 🌙 Early Return for Warmup Period
        if len(self.data) < 96:
            return

        # ✨ Moon Dev Debug Prints
        current_oi = self.oi_change[-1]
        current_price_chg = self.price_change[-1]
        current_rsi = self.rsi[-1]
        current_close = self.data.Close[-1]
        current_swing = self.swing_low[-1]
        
        print(f"🌙 Moon Dev Status | OI: {current_oi:.1f}% | PriceChg: {current_price_chg:.1f}% | RSI: {current_rsi:.1f} | SwingLow: {current_swing:.1f}")

        # 🚀 Entry Logic
        if not self.position:
            entry_conditions = [
                current_oi <= -20,         # OI drop >=20%
                current_price_chg <= -5,   # Price drop >=5%
                current_rsi < 30           # Oversold RSI
            ]
            
            if all(entry_conditions):
                # 🌙 Risk Management Calculation
                risk_percent = 0.01  # 1% risk per trade
                risk_amount = self.equity * risk_percent
                price_distance = current_close - current_swing
                
                if price_distance <= 0:
                    print("🚨 Moon Dev Warning: Invalid stop distance!")
                    return
                
                position_size = risk_amount / price_distance
                position_size = int(round(position_size))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=current_swing)
                    print(f"🚀🌕 BUY SIGNAL | Size: {position_size} | Entry: {current_close} | SL: {current_swing}")

        # 💫 Exit Logic
        else:
            # SMA Crossover Exit (replaced backtesting.lib.crossover)
            sma_cross = (self.data.Close[-2] < self.sma10[-2]) and (self.data.Close[-1] > self.sma10[-1])
            
            # RSI Threshold Exit
            rsi_cross = (self.rsi[-1] > 50) and (self.rsi[-2] <= 50)
            
            if sma_cross or rsi_cross:
                self.position.close()
                exit_type = "SMA Cross" if sma_cross else "RSI Cross"
                print(f"🌙✨ SELL SIGNAL | {exit_type} | Price: {current_close}")

# 🌙 Main Backtest Execution
if __name__ == "__main__":
    # Prepare lunar data
    data_path =