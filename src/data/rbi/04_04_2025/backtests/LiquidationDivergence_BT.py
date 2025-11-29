```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import talib
import numpy as np

# Data Handling
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean column names
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])

# Map columns to required format
required_mapping = {
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'liquidations': 'Liquidations'  # Assuming liquidation data exists in CSV
}
data = data.rename(columns=required_mapping)

# Ensure required columns exist
required_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Liquidations']
for col in required_columns:
    if col not in data.columns:
        raise ValueError(f"Missing required column: {col}")

class LiquidationDivergence(Strategy):
    rsi_period = 14
    bb_period = 20
    bb_dev = 2
    liquidation_window = 96  # 24h in 15m intervals
    swing_low_period = 20
    risk_percent = 0.02  # 2% risk per trade

    def init(self):
        # Liquidation Spike Indicator
        self.liquidation_max = self.I(talib.MAX, self.data.Liquidations, self.liquidation_window)
        
        # RSI
        self.rsi = self.I(talib.RSI, self.data.Close, self.rsi_period)
        
        # OBV
        self.obv = self.I(talib.OBV, self.data.Close, self.data.Volume)
        self.obv_ma = self.I(talib.EMA, self.obv, 20)
        
        # Bollinger Bands Components
        self.sma = self.I(talib.SMA, self.data.Close, self.bb_period)
        self.stddev = self.I(talib.STDDEV, self.data.Close, self.bb_period)
        
        # Swing Low
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_low_period)
        
        self.entry_flag = False

    def next(self):
        if self.position:
            current_price = self.data.Close[-1]
            upper_band = self.sma[-1] + (self.stddev[-1] * self.bb_dev)
            band_width = upper_band - (self.sma[-1] - (self.stddev[-1] * self.bb_dev))
            profit_target = upper_band + 0.5 * band_width
            
            if current_price >= profit_target:
                self.position.close()
                print(f"🚀 Take Profit Reached! Moon Profit: {profit_target:.2f}")
            elif current_price <= self.position.sl:
                self.position.close()
                print(f"🌧️ Stop Loss Hit! Lunar Protection Activated")
        else:
            if self.entry_flag:
                risk_amount = self.equity * self.risk_percent
                entry_price = self.data.Open[-1]
                sl_price = self.swing_low[-1] * 0.995
                
                if entry_price <= sl_price:
                    print("⚠️ Invalid SL Price - Aborting Launch")
                    self.entry_flag = False
                    return
                
                risk_per_share = entry_price - sl_price
                position_size = int(round(risk_amount / risk_per_share))
                
                if position_size > 0:
                    upper_band = self.sma[-1] + (self.stddev[-1] * self.bb_dev)
                    band_width = upper_band - (self.sma[-1] - (self.stddev[-1] * self.bb_dev))
                    tp_price = upper_band + 0.5 * band_width
                    
                    self.buy(
                        size=position_size,
                        sl=sl_price,
                        tp=tp_price,
                        exectype=MarketOrder
                    )
                    print(f"🌙✨ LUNAR LIFTOFF! Long {position_size} units")
                    print(f"   Entry: {entry_price:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
                self.entry_flag = False
            else:
                # Check entry conditions
                liquidation_spike =