import numpy as np
from collections import deque

class Starter(QCAlgorithm):
    def Initialize(self):
        # Environment setup.
        self.SetStartDate(2014, 11, 1)
        self.SetCash(1000000)
        self.SetBrokerageModel(AlphaStreamsBrokerageModel())
        self.SetExecution(ImmediateExecutionModel())
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel())
        self.UniverseSettings.Resolution = Resolution.Minute
        self.symbols = {}
        
        # Parameters.
        self.barResolution = Resolution.Hour # Use 1H tradebars.
        self.insightFrequency = self.TimeRules.At(10, 30) # Emit insights daily at 10:30am.
        self.warmupBars = 100 # Initialize with 100 historical tradebars.
        self.isLongOnly = True # Only use long positions.

        # Selection.           
        # self.SetUniverseSelection(LiquidETFUniverse()) # Or, self.AddEquities(['QQQ', 'QTEC'])
        self.AddEquities(['SPY', 'QQQ', 'QTEC'])
        self.SetBenchmark('SPY')
        
        # Main insight loop.
        self.Schedule.On(self.DateRules.EveryDay(), self.insightFrequency, self.PublishInsights)
                         
    def Logger(self, msg):
        self.Debug('[{}] {}'.format(self.Time, msg))
        
    def ActiveSymbols(self):
        # Returns a tuple list of active symbols, ready for trading.
        return [(str(x.Symbol), self.symbols[str(x.Symbol)]) for x in self.ActiveSecurities.Values \
                 if self.IsMarketOpen(x.Symbol) and \
                 str(x.Symbol) in self.symbols.keys() and \
                 self.Securities[x.Symbol].Price > 0 and \
                 self.Status != Status.Wait]

    def OnSecuritiesChanged(self, changes):
        symbolsAdded   = [x.Symbol for x in changes.AddedSecurities]
        symbolsRemoved = [x.Symbol for x in changes.RemovedSecurities]

        # Warm up new symbols.
        self.AddSymbols(symbolsAdded)

        # Discard removed symbols.
        self.RemoveSymbols(symbolsRemoved)
        
    def AddEquities(self, equities):
        # Adds a list of equities to the strategy.
        self.AddSymbols([self.AddEquity(equity).Symbol for equity in equities])

    def AddSymbols(self, symbols):
        # Warms up a list of symbols and adds them to the symbols dict.
        df = self.History(symbols, self.warmupBars, self.barResolution)
        for symbol in symbols:
            self.symbols[str(symbol)] = Symbol(str(symbol), symbol.ID, self)
            if str(symbol.ID) not in df.index.get_level_values(0): continue
            self.symbols[str(symbol)].HistoricalDataWarmup(df.loc[symbol], self)

    def RemoveSymbols(self, symbols):
        # Removes a list of symbols and unsubscribes their tradebar consolidators.
        for symbol in symbols:
            self.SubscriptionManager.RemoveConsolidator(symbol, self.symbols[symbol].bar)
            self.symbols[symbol].pop()
            
    def PublishInsights(self):
        # Emit insights from active symbols and update status.
        insights = []

        for symbol, symbolData in self.ActiveSymbols():
            status = symbolData.status
            duration = symbolData.exp_max - self.Time if symbolData.exp_max != None else Time.EndOfTimeTimeSpan

            isInvested = self.Portfolio[symbol].Invested
            isLong = self.Portfolio[symbol].IsLong
            isExpired = symbolData.exp_max != None and self.Time > symbolData.exp_max
            isClose = (status == Status.Sell and isLong) or (status == Status.Buy and not isLong) or status == Status.Close

            if not isInvested:
                if isExpired:
                    # Update the expired symbols from the portfolio environment.
                    symbolData.Reset(Status.Ready)
                    continue
                if status == Status.Buy:
                    # Push buy orders.
                    insights.append(Insight.Price(symbol, duration, InsightDirection.Up, symbolData.magnitude, None, None))
                    symbolData.status = Status.Bought
                    symbolData.orderAt = self.Time
                elif status == Status.Sell and not self.isLongOnly:
                    # Push sell orders.
                    insights.append(Insight.Price(symbol, duration, InsightDirection.Down, symbolData.magnitude, None, None))
                    symbolData.status = Status.Sold
                    symbolData.orderAt = self.Time
            else:
                # Updated expired (closed) positions and process close events for current holdings.
                if isExpired or isClose:
                    insights.append(Insight.Price(symbol, timedelta(1), InsightDirection.Flat))
                    symbolData.Reset(Status.Ready)

        self.EmitInsights(insights)

class Symbol:
    def __init__(self, symbol, symbolId, alg):
        # Setup.
        self.symbol = symbol
        self.id = symbolId
        self.Reset(Status.Wait)
        self.Initialize(alg)
        
        # Subscribe to tradebars.
        self.bar = TradeBarConsolidator(Extensions.ToTimeSpan(alg.barResolution))
        self.bar.DataConsolidated += lambda sender, tradebar: self.HandleBars(tradebar, alg)
        alg.SubscriptionManager.AddConsolidator(symbol, self.bar)

    def Initialize(self, alg):
        self.barQ = deque(maxlen=alg.warmupBars)
        self.rsi = alg.RSI(self.symbol, 30, alg.barResolution)

    def HandleBars(self, bar, alg, isWarmup=False):
        # Main event loop.
        self.rsi.Update(bar.Time, bar.Close)
        self.barQ.append(bar)
        
        if isWarmup or len(self.barQ) < self.barQ.maxlen: return
        
        rsiValue = self.rsi.Current.Value
        close = np.array([bar.Close for bar in self.barQ])
        stdDev = np.std(close)
        
        if self.status == Status.Ready:
            if rsiValue < 30:
                self.status = Status.Buy
                self.magnitude = stdDev / close[-1]
                self.exp_max = alg.Time + timedelta(weeks=4) # Close orders open longer than 4 weeks.
                self.exp_min = alg.Time + timedelta(weeks=1) # Keep orders open at least 1 week.
        elif self.status == Status.Bought:
            if self.exp_min != None and alg.Time < self.exp_min: return
            
            if rsiValue > 70:
                self.status = Status.Sell # Immediately close if RSI crosses upper limit.

    def HistoricalDataWarmup(self, history, alg):
        # Stream historical warm up data.
        for t, r in history.iterrows():
            o, h, l, c, v = r['open'], r['high'], r['low'], r['close'], r['volume']
            bar = TradeBar(t, self.symbol, o, h, l, c, v)
            self.HandleBars(bar, alg, isWarmup=True)
        self.status = Status.Ready
        
    def Reset(self, status):
        self.exp_max = None
        self.exp_min = None
        self.magnitude = None
        self.orderAt = None
        self.status = status
        
class Status(Enum):
    Wait   = 0 # Warming up or not ready for trading.
    Ready  = 1 # Ready to trade.
    Buy    = 2 # Signal a buy recommendation (close if sold.)
    Sell   = 3 # Signal a sell recommendation (close if bought.)
    Close  = 4 # Signal a close recommendation.
    Bought = 5 # Symbol holds a long position.
    Sold   = 6 # Symbol holds a short position.
