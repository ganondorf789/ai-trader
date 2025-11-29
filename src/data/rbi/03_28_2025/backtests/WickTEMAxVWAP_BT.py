```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import pandas as pd
import pandas_ta as ta
import talib

class WickTEMAxVWAP(Strategy):
    risk_per_trade = 0.01  # 1% risk per trade
    
    def init(self):
        # Moon Dev initialization message
        print("🌙 INITIALIZING MOON DEV BACKTEST ENGINE 🌙")
        
        # Preprocess and resample data
        self.preprocess_data()
        
    def preprocess_data(self):
        # Clean and prepare base data
        df = self.data.df.copy()
        df.columns = [col.strip().lower() for col in df.columns]
        df = df.drop(columns=[col for col in df.columns if 'unnamed' in col])
        
        # Resample to 4H timeframe for Heikin-Ashi calculations
        resampled_4h = df.resample('4H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Calculate Heikin-Ashi candles
        self.ha_close = (resampled_4h['open'] + resampled_4h['high'] + 
                        resampled_4h['low'] + resampled_4h['close']) / 4
        self.ha_open = pd.Series(index=resampled_4h.index, dtype=float)
        self.ha_open.iloc[0] = (resampled_4h['open'].iloc[0] + resampled_4h['close'].iloc[0]) / 2
        
        for i in range(1, len(resampled_4h)):
            self.ha_open.iloc[i] = (self.ha_open.iloc[i-1] + self.ha_close.iloc[i-1]) / 2
            
        self.ha_high = resampled_4h[['high', 'open', 'close']].max(axis=1)
        self.ha_low = resampled_4h[['low', 'open', 'close']].min(axis=1)
        
        # Calculate TEMA and VWAP
        self.tema_20 = talib.TEMA(self.ha_close, timeperiod=20)
        self.vwap = (self.ha_close * resampled_4h['volume']).cumsum() / resampled_4h['volume'].cumsum()
        
        # Resample to 1H for Stochastic RSI
        resampled_1h = df.resample('1H').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last'
        })
        stoch_rsi = ta.stochrsi(resampled_1h['close'], length=14)
        self.stoch_k = stoch_rsi['STOCHRSIk_14_14_3_3']
        self.stoch_d = stoch_rsi['STOCHRSId_14_14_3_3']
        
    def next(self):
        current_idx = len(self.data) - 1
        i_4h = current_idx // 16
        i_1h = current_idx // 4
        
        # Entry Logic 🌙
        if not self.position and i_4h >= 3:
            # Check consecutive bullish HA candles
            ha_bullish = all(self.ha_close[i_4h - i] > self.ha_open[i_4h - i] for i in range(3))
            
            # Check decreasing upper wicks
            wicks = [
                self.ha_high[i_4h - i] - max(self.ha_open[i_4h - i], self.ha_close[i_4h - i])
                for i in range(3)
            ]
            decreasing_wicks = all(wicks[i] > wicks[i+1] for i in range(2))
            
            # Check TEMA crossover
            tema_cross = crossover(self.tema_20[i_4h-2:i_4h+1], self.vwap[i_4h-2:i_4h+1])
            
            if ha_bullish and decreasing_wicks and tema_cross[-1]:
                # Calculate stop loss
                stop_level = min(self.ha_low[i_4h-2:i_4h+1])
                risk_amount = self.equity * self.risk_per_trade
                risk_per_share = self.data.Close[-1] - stop_level
                position_size = int(round(risk_amount