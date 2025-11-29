.is_short:
                # Short exit conditions
                exit_cond = (
                    (self.stoch_k[-1] > 20 and self.stoch_k[-2] <= 20) or
                    (self.data.Close[-1] > self.lower_band[-1])
                )
                if exit_cond:
                    self.position.close()
                    print(f"✅ MOON DEV SHORT EXIT | Price: {self.data.Close[-1]} | StochRSI: {self.stoch_k[-1]}")

# Backtest setup
bt = Backtest(data, SqueezeSurge, commission=.002, exclusive_orders=True)

# Run backtest
stats = bt.run()
print("🌙✨ MOON DEV BACKTEST COMPLETE ✨🌙")
print(stats)

# Plot results
bt.plot(filename='moon_dev_squeeze_surge')