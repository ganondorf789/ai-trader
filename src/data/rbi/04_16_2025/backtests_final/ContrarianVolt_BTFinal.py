The code you've provided is already well-structured and follows best practices for backtesting. However, I'll make a few Moon Dev enhancements while ensuring we don't change the strategy logic:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import talib
from backtesting import Strategy, Backtest

# Enhanced data preprocessing with Moon Dev cosmic checks ✨
def prepare_data(path):
    print("🌙 MOON DEV DATA PREP: Scanning cosmic market data...")
    try:
        data = pd.read_csv(path)
        # Cosmic data validation
        required_cols = {'open', 'high', 'low', 'close', 'volume'}
        if not required_cols.issubset(set(col.lower() for col in data.columns)):
            raise ValueError("Missing required columns in cosmic data!")
            
        # Clean column names with lunar precision
        data.columns = data.columns.str.strip().str.lower()
        # Drop unnamed columns (cosmic debris)
        data = data.drop(columns=[col for col in data.columns if 'unnamed' in col], errors='ignore')
        
        # Standardize column names
        data.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        # Cosmic data quality check
        if data.isnull().values.any():
            print("⚠️ MOON DEV WARNING: Null values detected in cosmic data! Performing stardust cleanup...")
            data.ffill(inplace=True)
            
        print("🌌 Data preparation complete! Ready for lunar strategy deployment.")
        return data
    except Exception as e:
        print(f"💥 COSMIC DATA ERROR: {str(e)}")
        raise

class ContrarianVolt(Strategy):
    risk_per_trade = 0.01  # 1% of equity per trade (cosmic risk management)
    
    def init(self):
        print("🌕 MOON DEV INIT: Charging lunar indicators...")
        try:
            # Calculate indicators using TA-Lib with self.I()
            self.cmo = self.I(talib.CMO, self.data.Close, timeperiod=14)
            self.sma50 = self.I(talib.SMA, self.data.Close, 50)
            self.swing_low = self.I(talib.MIN, self.data.Low, 20)
            self.upper_band = self.I(self._calc_upper_band, self.data.Close)
            print("✨ Indicator initialization complete! Ready for cosmic signals.")
        except Exception as e:
            print(f"💥 COSMIC INDICATOR ERROR: {str(e)}")
            raise
        
    def _calc_upper_band(self, close):
        upper, _, _ = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        return upper
        
    def next(self):
        if not self.position:
            # Entry conditions with lunar precision
            try:
                cmo_condition = (self.cmo[-1] < -50) and (self.cmo[-2] >= -50)
                funding_condition = (self.data.funding_rate[-1] < -0.001) and \
                                   (self.data.funding_rate[-1] == min(self.data.funding_rate[-5:]))
                trend_condition = self.sma50[-1] > self.sma50[-5]
                
                if cmo_condition and funding_condition and trend_condition:
                    # Cosmic position sizing calculation
                    entry_price = self.data.Close[-1]
                    sl_price = min(self.swing_low[-1], entry_price * 0.98)
                    risk_amount = self.equity * self.risk_per_trade
                    risk_per_unit = entry_price - sl_price
                    
                    if risk_per_unit > 0:
                        position_size = int(round(risk_amount / risk_per_unit))
                        if position_size > 0:
                            # Execute entry with OCO orders (Orbital Command Orders)
                            self.buy(
                                size=position_size,
                                sl=sl_price,
                                tp=self.upper_band[-1],
                                tag="🌙✨ ContrarianVolt Entry"
                            )
                            print(f"🚀🌕 MOON DEV ALERT: Cosmic Long Entry @ {entry_price:.2f}")
                            print(f"⚡ Risk: {risk_amount:.2f}, Size: {position_size}, SL: {sl_price:.2f}, TP: {self.upper_band[-1]:.2f}")
            except Exception as e:
                print(f"💥 COSMIC TRADE ERROR: {str(e)}")

# Run backtest with lunar energy
if __name__ == "__main