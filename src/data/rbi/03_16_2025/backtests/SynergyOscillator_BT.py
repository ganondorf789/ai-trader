```python
import pandas as pd
import talib
import pandas_ta as ta
from backtesting import Backtest, Strategy

# Data preparation
data_path = '/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv'
data = pd.read_csv(data_path, parse_dates=['datetime'], index_col='datetime')

# Clean and format columns
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data = data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
})

class SynergyOscillator(Strategy):
    risk_percent = 0.01  # 1% risk per trade
    risk_reward_ratio = 2  # 2:1
    atr_multiplier = 1.5  # Stop loss multiplier
    swing_period = 20
    
    def init(self):
        # Core indicators
        self.rsi = self.I(talib.RSI, self.data.Close, 14)
        
        # Volatility indicators
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14)
        self.atr_ma = self.I(talib.SMA, self.atr, 14)
        
        # Ichimoku Cloud
        ichimoku = ta.ichimoku(self.data.High, self.data.Low, self.data.Close)
        self.tenkan = self.I(ichimoku['ITS_9'], name='Tenkan')
        self.kijun = self.I(ichimoku['IKS_26'], name='Kijun')
        self.senkou_a = self.I(ichimoku['ISA_9'], name='Senkou A')
        self.senkou_b = self.I(ichimoku['ISB_26'], name='Senkou B')
        
        # Volume-weighted trend
        vwma = ta.vwma(self.data.Close, self.data.Volume, length=20)
        self.vwma = self.I(vwma, name='VWMA')
        
        # Swing levels
        self.swing_high = self.I(talib.MAX, self.data.High, self.swing_period)
        self.swing_low = self.I(talib.MIN, self.data.Low, self.swing_period)
    
    def next(self):
        # Moon Dev debug markers
        print(f"🌙 Processing bar {len(self.data)-1} | Price: {self.data.Close[-1]:.2f}")
        
        if self.position:
            current_rsi = self.rsi[-1]
            current_close = self.data.Close[-1]
            
            if self.position.is_long:
                if current_rsi >= 70:
                    print(f"🚨🌕 Overbought RSI {current_rsi:.2f}! Closing long")
                    self.position.close()
                elif current_close < self.senkou_a[-1] or current_close < self.senkou_b[-1]:
                    print(f"☁️🌑 Price entered cloud! Closing long")
                    self.position.close()
                    
            elif self.position.is_short:
                if current_rsi <= 30:
                    print(f"🚨🌑 Oversold RSI {current_rsi:.2f}! Closing short")
                    self.position.close()
                elif current_close > self.senkou_a[-1] or current_close > self.senkou_b[-1]:
                    print(f"☁️🌕 Price entered cloud! Closing short")
                    self.position.close()
        else:
            # Entry logic
            current_rsi = self.rsi[-1]
            prev_rsi = self.rsi[-2] if len(self.rsi) > 1 else None
            current_close = self.data.Close[-1]
            
            # Bullish conditions
            bull_rsi = prev_rsi and (prev_rsi <= 50) and (current_rsi > 50)
            bull_cloud = current_close > self.senkou_a[-1] and current_close > self.senkou_b[-1]
            bull_atr = self.atr[-1] > self.atr_ma[-1]
            bull_vwma = self.vwma[-1] > self.vwma[-2] if len(self.vwma) > 1 else False
            
            if bull_rsi and bull_cloud and bull_atr and bull_vwma:
                sl = self.swing_low[-1] - self.atr_multiplier * self.atr[-1]
                risk =