```python
# 🌙 Moon Dev's VoltaVWAP Backtest 🌙
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import talib

# 🚀 DATA PREPARATION 🚀
data_path = "/Users/md/Dropbox/dev/github/moon-dev-ai-agents-for-trading/src/data/rbi/BTC-USD-15m.csv"
data = pd.read_csv(data_path)

# Cleanse and prepare data
data.columns = data.columns.str.strip().str.lower()
data = data.drop(columns=[col for col in data.columns if 'unnamed' in col.lower()])
data.rename(columns={
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume'
}, inplace=True)
data['DateTime'] = pd.to_datetime(data['datetime'])
data.set_index('DateTime', inplace=True)

class VoltaVWAP(Strategy):
    risk_percent = 0.01  # 1% risk per trade 🌙
    
    def init(self):
        # 🌙 INDICATOR CALCULATION 🌙
        # VWAP (20-period)
        vwap_values = ta.vwap(
            high=self.data.High,
            low=self.data.Low,
            close=self.data.Close,
            volume=self.data.Volume,
            length=20
        )
        self.vwap = self.I(lambda: vwap_values, name='VWAP')
        
        # ATR (14-period)
        self.atr = self.I(talib.ATR, 
                         self.data.High, 
                         self.data.Low, 
                         self.data.Close, 
                         timeperiod=14, 
                         name='ATR')
        
        # RSI (14-period)
        self.rsi = self.I(talib.RSI, 
                         self.data.Close, 
                         timeperiod=14, 
                         name='RSI')
        
        print("🌙✨ Moon Dev Indicators Initialized! VWAP, ATR, RSI ready 🚀")

    def next(self):
        # 🌙 ENTRY LOGIC 🌙
        if not self.position:
            upper_band = self.vwap[-1] + 2 * self.atr[-1]
            lower_band = self.vwap[-1] - 2 * self.atr[-1]
            current_close = self.data.Close[-1]
            prev_close = self.data.Close[-2] if len(self.data.Close) > 1 else current_close

            # Long Entry Check 🌙
            if current_close > upper_band and prev_close <= upper_band:
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1] or 1  # Avoid division by zero
                position_size = int(round(risk_amount / atr_value))
                
                if position_size > 0:
                    self.buy(size=position_size, 
                            sl=current_close - atr_value,
                            tag="MoonDev_Long")
                    print(f"🌙🚀 LONG ENTRY! Size: {position_size} | Price: {current_close:.2f} | SL: {current_close - atr_value:.2f}")

            # Short Entry Check 🌙
            elif current_close < lower_band and prev_close >= lower_band:
                risk_amount = self.equity * self.risk_percent
                atr_value = self.atr[-1] or 1
                position_size = int(round(risk_amount / atr_value))
                
                if position_size > 0:
                    self.sell(size=position_size, 
                             sl=current_close + atr_value,
                             tag="MoonDev_Short")
                    print(f"🌙💫 SHORT ENTRY! Size: {position_size} | Price: {current_close:.2f} | SL: {current_close + atr_value:.2f}")

        # 🌙 EXIT LOGIC 🌙
        else:
            current_rsi = self.rsi[-1]
            prev_rsi = self.rsi[-2] if len(self.rsi) > 1 else current_rsi

            if self.position.is_long:
                # Profit Take (RSI <60)
                if prev_rsi >= 60 and current_rsi < 60:
                    self.position.close()
                    print(f"🌙✅ LONG EXIT (Profit) | RSI: {current_rsi:.2f}")
                # Stop Loss (RSI <40)
                elif prev_rsi >= 40 and current_rsi < 40:
                    self.position.close()
                    print(f"🌙🛑 LONG EXIT (Stop