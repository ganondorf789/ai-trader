I'll complete and debug the code while maintaining the original strategy logic. Here's the fixed version with Moon Dev themed debug prints:

```python
# -*- coding: utf-8 -*-
import pandas as pd
import talib
from backtesting import Strategy, Backtest

class LiquidationDivergence(Strategy):
    # Strategy parameters
    funding_threshold = -0.0005  # Negative funding rate threshold for longs
    positive_funding_threshold = 0.0007  # Positive funding rate threshold for shorts
    flow_threshold = 1000000  # Net flow threshold for accumulation/distribution
    risk_percent = 0.02  # 2% risk per trade
    stop_buffer = 0.005  # 0.5% buffer for stop loss
    volume_threshold = 1000000  # Minimum volume threshold
    
    def init(self):
        # Precompute indicators using TA-Lib wrapped in self.I()
        close = self.data.Close.values
        
        # Stochastic RSI (14,3,3)
        stoch_k, stoch_d = talib.STOCHRSI(close, timeperiod=14, fastk_period=3, fastd_period=3)
        self.I(lambda: stoch_k, name='StochRSI_K')
        self.I(lambda: stoch_d, name='StochRSI_D')
        
        # Debug prints for indicator confirmation
        print("🌙✨ Moon Dev Indicators Activated! ✨🌙")
        print(f"Stochastic RSI K/D Length: {len(stoch_k)}/{len(stoch_d)}")
        
    def next(self):
        # Skip low liquidity periods
        if self.data.Volume[-1] < self.volume_threshold:
            return
        
        # Moon Dev debug information
        moon_dev_debug = f"🌙 Bar {len(self.data)} | Price: {self.data.Close[-1]} | "
        moon_dev_debug += f"Funding: {self.data.funding_rate[-1]:.6f} | Flow: {self.data.net_flow[-1]:.2f} 🌙"
        print(moon_dev_debug)
        
        if not self.position:
            # Long entry conditions
            if (self.data.funding_rate[-1] < self.funding_threshold and
                self.data.net_flow[-1] > self.flow_threshold and
                self.data.High[-1] > self.data.liquidation_cluster_high[-1]):
                
                # Risk management calculations
                stop_price = self.data.liquidation_cluster_low[-1] * (1 - self.stop_buffer)
                risk_per_share = self.data.Close[-1] - stop_price
                position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                
                if position_size > 0:
                    self.buy(size=position_size, sl=stop_price)
                    print(f"🚀🌕 LONG ENTRY! Size: {position_size} | SL: {stop_price:.2f} 🚀")
            
            # Short entry conditions
            elif (self.data.funding_rate[-1] > self.positive_funding_threshold and
                  self.data.net_flow[-1] < -self.flow_threshold and
                  self.data.Low[-1] < self.data.liquidation_cluster_low[-1]):
                
                # Risk management calculations
                stop_price = self.data.liquidation_cluster_high[-1] * (1 + self.stop_buffer)
                risk_per_share = stop_price - self.data.Close[-1]
                position_size = int(round((self.equity * self.risk_percent) / risk_per_share))
                
                if position_size > 0:
                    self.sell(size=position_size, sl=stop_price)
                    print(f"🌑🛸 SHORT ENTRY! Size: {position_size} | SL: {stop_price:.2f} 🌌")
        else:
            # Exit conditions
            current_k = self.data.StochRSI_K[-1]
            prev_k = self.data.StochRSI_K[-2] if len(self.data) > 1 else current_k
            
            # Exit long when StochRSI crosses below 80
            if self.position.is_long and prev_k >= 80 and current_k < 80:
                self.position.close()
                print(f"🌙💎 LONG EXIT: StochRSI_K {prev_k:.1f}->{current_k:.1f} 💎")
                
            # Exit short when StochRSI crosses above 20
            elif self.position.is_short and prev_k <= 20 and current_k > 20:
                self.position.close()
                print(f"🌙🔥 SHORT EXIT: St