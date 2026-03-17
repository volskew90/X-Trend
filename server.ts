import express from "express";
import { createServer as createViteServer } from "vite";
import path from "path";

async function startServer() {
  const app = express();
  const PORT = 3000;

  app.use(express.json());

  // API routes
  app.post("/api/backtest", (req, res) => {
    const { ticker, startDate, endDate, lr, alpha } = req.body;
    
    // Seed random walk based on ticker length/chars just to make it vary slightly
    let seed = (ticker || 'AAPL').split('').reduce((acc: number, char: string) => acc + char.charCodeAt(0), 0);
    const random = () => {
      const x = Math.sin(seed++) * 10000;
      return x - Math.floor(x);
    };

    const data = [];
    let price = 100 + random() * 100; // Random starting price
    const start = new Date(startDate || "2023-01-01");
    const end = new Date(endDate || "2024-01-01");
    
    let current = new Date(start);
    let inPosition = false;
    let buyPrice = 0;
    
    let cash = 10000;
    let shares = 0;
    let peakValue = 10000;
    let maxDrawdown = 0;
    
    let winningTrades = 0;
    let totalTrades = 0;
    
    const dailyReturns = [];
    let lastValue = 10000;

    // Adjust volatility based on learning rate/alpha just to make UI reactive
    const vol = 0.015 + (parseFloat(lr) || 0.001) * 2; 
    const trendProb = 0.5 + (parseFloat(alpha) || 0.1) * 0.05;

    while (current <= end) {
      // Random walk with slight momentum
      const change = (random() - (1 - trendProb)) * vol;
      price = price * (1 + change);
      
      let signal = "HOLD";
      
      // Trading logic
      if (!inPosition && random() > 0.90) {
        signal = "BUY";
        inPosition = true;
        buyPrice = price;
        shares = cash / price;
        cash = 0;
      } else if (inPosition && (price > buyPrice * 1.05 || price < buyPrice * 0.95 || random() > 0.95)) {
        signal = "SELL";
        inPosition = false;
        cash = shares * price;
        shares = 0;
        totalTrades++;
        if (price > buyPrice) winningTrades++;
      }
      
      const currentValue = cash + shares * price;
      if (currentValue > peakValue) peakValue = currentValue;
      const drawdown = (peakValue - currentValue) / peakValue;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
      
      const dailyReturn = (currentValue - lastValue) / lastValue;
      dailyReturns.push(dailyReturn);
      lastValue = currentValue;

      data.push({
        date: current.toISOString().split('T')[0],
        price: parseFloat(price.toFixed(2)),
        signal: signal,
        buyPrice: signal === "BUY" ? parseFloat(price.toFixed(2)) : null,
        sellPrice: signal === "SELL" ? parseFloat(price.toFixed(2)) : null,
        portfolioValue: parseFloat(currentValue.toFixed(2))
      });
      
      current.setDate(current.getDate() + 1);
    }

    // Force sell at the end if in position
    if (inPosition) {
      cash = shares * price;
      shares = 0;
      totalTrades++;
      if (price > buyPrice) winningTrades++;
      data[data.length - 1].signal = "SELL";
      data[data.length - 1].sellPrice = parseFloat(price.toFixed(2));
      data[data.length - 1].portfolioValue = parseFloat(cash.toFixed(2));
    }

    const totalReturn = ((cash - 10000) / 10000) * 100;
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;
    
    const meanReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
    const stdReturn = Math.sqrt(dailyReturns.reduce((a, b) => a + Math.pow(b - meanReturn, 2), 0) / dailyReturns.length) || 1e-6;
    const sharpeRatio = (meanReturn / stdReturn) * Math.sqrt(252);

    const metrics = {
      totalReturn: (totalReturn >= 0 ? "+" : "") + totalReturn.toFixed(2) + "%",
      sharpeRatio: sharpeRatio.toFixed(2),
      maxDrawdown: "-" + (maxDrawdown * 100).toFixed(2) + "%",
      winRate: winRate.toFixed(1) + "%",
      tradesExecuted: totalTrades
    };

    // Simulate model training/inference delay
    setTimeout(() => {
      res.json({ data, metrics });
    }, 1500);
  });

  // Vite middleware for development
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
