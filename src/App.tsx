import React, { useState } from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Scatter, ComposedChart 
} from 'recharts';
import { Play, Settings, TrendingUp, Activity, AlertTriangle, CheckCircle2, Loader2, BarChart3 } from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const SignalDot = (props: any) => {
  const { cx, cy, payload } = props;
  if (cx == null || cy == null) return null;
  
  if (payload.signal === 'BUY') {
    return (
      <g transform={`translate(${cx},${cy})`}>
        <polygon points="0,-8 -6,6 6,6" fill="#10b981" />
      </g>
    );
  }
  if (payload.signal === 'SELL') {
    return (
      <g transform={`translate(${cx},${cy})`}>
        <polygon points="0,8 -6,-6 6,-6" fill="#f43f5e" />
      </g>
    );
  }
  return null;
};

export default function App() {
  const [ticker, setTicker] = useState('AAPL');
  const [startDate, setStartDate] = useState('2023-01-01');
  const [endDate, setEndDate] = useState('2024-01-01');
  const [learningRate, setLearningRate] = useState('0.001');
  const [alpha, setAlpha] = useState('0.1');
  
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<any>(null);

  const handleRunBacktest = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, startDate, endDate, lr: learningRate, alpha })
      });
      const data = await response.json();
      setResults(data);
    } catch (error) {
      console.error("Failed to run backtest", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-100 font-sans selection:bg-indigo-500/30">
      {/* Top Navigation */}
      <header className="border-b border-white/10 bg-[#121212] px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-indigo-500/20 border border-indigo-500/50 flex items-center justify-center">
            <Activity className="w-5 h-5 text-indigo-400" />
          </div>
          <h1 className="text-xl font-semibold tracking-tight">X-Trend Trading System</h1>
        </div>
        <div className="flex items-center gap-4 text-sm text-zinc-400">
          <span className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-emerald-500"></span>
            System Online
          </span>
        </div>
      </header>

      <main className="p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-[1600px] mx-auto">
        
        {/* Left Sidebar: Configuration */}
        <div className="lg:col-span-3 space-y-6">
          <div className="bg-[#121212] border border-white/5 rounded-2xl p-5 shadow-xl">
            <div className="flex items-center gap-2 mb-6">
              <Settings className="w-5 h-5 text-zinc-400" />
              <h2 className="font-medium text-zinc-200">Model Configuration</h2>
            </div>
            
            <div className="space-y-4">
              <div>
                <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Asset Ticker</label>
                <input 
                  type="text" 
                  value={ticker}
                  onChange={(e) => setTicker(e.target.value)}
                  className="w-full bg-black/50 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors"
                />
              </div>
              
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">Start Date</label>
                  <input 
                    type="date" 
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="w-full bg-black/50 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors [color-scheme:dark]"
                  />
                </div>
                <div>
                  <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-1.5">End Date</label>
                  <input 
                    type="date" 
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="w-full bg-black/50 border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-indigo-500 transition-colors [color-scheme:dark]"
                  />
                </div>
              </div>

              <div className="pt-4 border-t border-white/5">
                <h3 className="text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">Hyperparameters</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between mb-1">
                      <label className="text-xs text-zinc-400">Learning Rate</label>
                      <span className="text-xs text-zinc-300 font-mono">{learningRate}</span>
                    </div>
                    <input 
                      type="range" min="0.0001" max="0.01" step="0.0001"
                      value={learningRate} onChange={(e) => setLearningRate(e.target.value)}
                      className="w-full accent-indigo-500"
                    />
                  </div>
                  <div>
                    <div className="flex justify-between mb-1">
                      <label className="text-xs text-zinc-400">Alpha (MLE Weight)</label>
                      <span className="text-xs text-zinc-300 font-mono">{alpha}</span>
                    </div>
                    <input 
                      type="range" min="0.0" max="1.0" step="0.1"
                      value={alpha} onChange={(e) => setAlpha(e.target.value)}
                      className="w-full accent-indigo-500"
                    />
                  </div>
                </div>
              </div>

              <button 
                onClick={handleRunBacktest}
                disabled={isLoading}
                className="w-full mt-6 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg py-2.5 text-sm font-medium transition-all flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 fill-current" />}
                {isLoading ? 'Running Backtest...' : 'Run Backtest'}
              </button>
            </div>
          </div>
          
          {/* Info Card */}
          <div className="bg-indigo-500/10 border border-indigo-500/20 rounded-2xl p-5">
            <div className="flex gap-3">
              <AlertTriangle className="w-5 h-5 text-indigo-400 shrink-0" />
              <div className="text-sm text-indigo-200/80 leading-relaxed">
                <strong className="text-indigo-300 block mb-1">Architecture Note</strong>
                This UI connects to a Node.js mock API. To use your actual PyTorch model, wrap <code className="bg-black/30 px-1 py-0.5 rounded text-xs">train.py</code> in a FastAPI server and update the API endpoint in <code className="bg-black/30 px-1 py-0.5 rounded text-xs">server.ts</code>.
              </div>
            </div>
          </div>
        </div>

        {/* Right Content: Dashboard */}
        <div className="lg:col-span-9 space-y-6">
          
          {/* KPI Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <KpiCard 
              title="Total Return" 
              value={results?.metrics?.totalReturn || "---"} 
              trend="up" 
              icon={<TrendingUp className="w-4 h-4 text-emerald-400" />} 
            />
            <KpiCard 
              title="Sharpe Ratio" 
              value={results?.metrics?.sharpeRatio || "---"} 
              icon={<Activity className="w-4 h-4 text-indigo-400" />} 
            />
            <KpiCard 
              title="Max Drawdown" 
              value={results?.metrics?.maxDrawdown || "---"} 
              trend="down" 
              icon={<BarChart3 className="w-4 h-4 text-rose-400" />} 
            />
            <KpiCard 
              title="Win Rate" 
              value={results?.metrics?.winRate || "---"} 
              icon={<CheckCircle2 className="w-4 h-4 text-blue-400" />} 
            />
          </div>

          {/* Main Chart */}
          <div className="bg-[#121212] border border-white/5 rounded-2xl p-5 shadow-xl min-h-[400px] flex flex-col">
            <h2 className="font-medium text-zinc-200 mb-6 flex items-center gap-2">
              <Activity className="w-5 h-5 text-zinc-400" />
              Price Action & Model Signals
            </h2>
            
            <div style={{ width: '100%', height: 400 }}>
              {results ? (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={results.data} margin={{ top: 20, right: 20, left: -20, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                    <XAxis 
                      dataKey="date" 
                      stroke="#ffffff40" 
                      fontSize={12} 
                      tickLine={false}
                      axisLine={false}
                      minTickGap={30}
                    />
                    <YAxis 
                      domain={['auto', 'auto']} 
                      stroke="#ffffff40" 
                      fontSize={12} 
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(val) => `$${val}`}
                    />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1a1a1a', borderColor: '#ffffff20', borderRadius: '8px' }}
                      itemStyle={{ color: '#e4e4e7' }}
                      labelStyle={{ color: '#a1a1aa', marginBottom: '4px' }}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="price" 
                      stroke="#6366f1" 
                      strokeWidth={2} 
                      dot={<SignalDot />} 
                      activeDot={{ r: 4, fill: '#6366f1', stroke: '#000', strokeWidth: 2 }}
                      name="Price"
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-full h-full flex flex-col items-center justify-center text-zinc-500 gap-3">
                  <BarChart3 className="w-8 h-8 opacity-50" />
                  <p className="text-sm">Configure parameters and run backtest to view results</p>
                </div>
              )}
            </div>
          </div>

          {/* Trade Log */}
          {results && (
            <div className="bg-[#121212] border border-white/5 rounded-2xl overflow-hidden shadow-xl">
              <div className="p-5 border-b border-white/5">
                <h2 className="font-medium text-zinc-200">Recent Signals</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead className="text-xs text-zinc-500 uppercase bg-black/20">
                    <tr>
                      <th className="px-6 py-3 font-medium">Date</th>
                      <th className="px-6 py-3 font-medium">Signal</th>
                      <th className="px-6 py-3 font-medium">Price</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/5">
                    {results.data
                      .filter((d: any) => d.signal !== 'HOLD')
                      .slice(-5)
                      .reverse()
                      .map((trade: any, i: number) => (
                      <tr key={i} className="hover:bg-white/[0.02] transition-colors">
                        <td className="px-6 py-4 text-zinc-300 font-mono">{trade.date}</td>
                        <td className="px-6 py-4">
                          <span className={cn(
                            "px-2 py-1 rounded text-xs font-medium",
                            trade.signal === 'BUY' ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
                          )}>
                            {trade.signal}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-zinc-300 font-mono">${trade.price.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

        </div>
      </main>
    </div>
  );
}

function KpiCard({ title, value, trend, icon }: { title: string, value: string, trend?: 'up' | 'down', icon: React.ReactNode }) {
  return (
    <div className="bg-[#121212] border border-white/5 rounded-2xl p-5 shadow-xl flex flex-col justify-between">
      <div className="flex justify-between items-start mb-4">
        <h3 className="text-sm font-medium text-zinc-400">{title}</h3>
        <div className="p-2 bg-black/30 rounded-lg border border-white/5">
          {icon}
        </div>
      </div>
      <div className="flex items-baseline gap-2">
        <span className={cn(
          "text-2xl font-semibold tracking-tight",
          trend === 'up' && "text-emerald-400",
          trend === 'down' && "text-rose-400",
          !trend && "text-zinc-100"
        )}>
          {value}
        </span>
      </div>
    </div>
  );
}
