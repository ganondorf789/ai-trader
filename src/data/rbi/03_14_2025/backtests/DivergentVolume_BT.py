```python
# 🌙 Moon Dev's DivergentVolume Backtest Implementation
import pandas as pd
import talib
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

class DivergentVolume(Strategy):
    def init(self):
        # 🌟 Cosmic Indicators
        self.rsi = self.I(talib.RSI, self.data.Close, 14, name='RSI_14')
        self.atr = self.I(talib.ATR, self.data.High, self.data.Low, self.data.Close, 14, name='ATR_14')
        self.volume_sma = self.I(talib.SMA, self.data.Volume, 20, name='Volume_SMA_20')
        
        # 🏔️ Peak Detection
        self.price_highs = self.I(talib.MAX, self.data.High, 20, name='Price_Highs_20')
        self.rsi_highs = self.I(talib.MAX, self.rsi, 20, name='RSI_Highs_20')
        self.volume_peaks = self.I(talib.MAX, self.data.Volume, 20, name='Volume_Peaks_20')
        
        self.exit_rsi_level = None
        self.trade_count = 0

    def next(self):
        # 🛑 Wait for indicators to stabilize
        if len(self.data) < 20 or len(self.rsi) < 20:
            return

        # 🌌 Current Cosmic Readings
        current_close = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_rsi = self.rsi[-1]
        current_volume = self.data.Volume[-1]
        atr_value = self.atr[-1]

        # Previous Cycle Measurements
        prev_price_high = self.price_highs[-2]
        prev_rsi_high = self.rsi_highs[-2]
        prev_volume_peak = self.volume_peaks[-2]
        volume_sma = self.volume_sma[-1]

        # 🕵️♂️ Detect Bearish Divergence
        price_higher_high = current_high > prev_price_high
        rsi_lower_high = current_rsi < prev_rsi_high
        bearish_divergence = price_higher_high and rsi_lower_high

        # 📉 Volume Confirmation
        volume_declining = (current_volume < prev_volume_peak) and (current_volume < volume_sma)

        # 🚀 Entry Conditions
        if (bearish_divergence and volume_declining and
            current_close > prev_price_high and
            not self.position and
            self.trade_count < 3):
            
            # 💰 Risk Management Calculations
            risk_amount = self.equity * 0.01
            stop_loss_distance = atr_value
            position_size = int(round(risk_amount / stop_loss_distance))
            
            # 🛡️ Max Exposure Check
            max_size = int((self.equity * 0.05) // current_close
            position_size = min(position_size, max_size)
            
            if position_size > 0:
                # 🌕 Moon Dev Entry
                self.buy(size=position_size)
                self.trade_count += 1
                print(f"\n🚀 MOON SHOT! Long {position_size} @ {current_close:.2f}")
                print(f"   🌗 RSI Divergence: {prev_rsi_high:.2f} -> {current_rsi:.2f}")
                print(f"   📉 Volume Below SMA: {volume_sma:.2f} vs {current_volume:.2f}")

                # 🎯 Exit Targets
                entry_price = current_close
                self.exit_rsi_level = prev_rsi_high
                stop_price = entry_price - atr_value
                profit_price = entry_price + (atr_value * 1.5)
                
                # ⚔️ Place Protection Orders
                self.sell(size=position_size, exectype=Order.Stop, price=stop_price)
                self.sell(size=position_size, exectype=Order.Limit, price=profit_price)
                print(f"   🔐 Stop Loss: {stop_price:.2f} | 🎯 Take Profit: {profit_price:.2f}")

        # 🌊 Exit Condition - RSI Convergence
        if self.position and current_rsi > self.exit_rsi_level:
            self.position.close()
            print(f"\n🌑 RSI CONVERGENCE! Closing position @ {current_close:.2f}")
            print(f"   🌈 Current RSI {current_rsi