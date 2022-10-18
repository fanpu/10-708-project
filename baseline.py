import os
import pickle

import ipdb
import numpy as np
import yfinance as yf
from tqdm import tqdm

from sp500_tickers import sp500_tickers

CACHE_DIR = "cache"
EXPECTED_TRADING_DAYS = 3021


class Ticker():
    def __init__(self, symbol, start_date, end_date):
        """
        Returns:
            whether this ticker should be used in computing the correlation coefficient
            based on whether it has been trading for all the days expected
        """
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.ticker = yf.Ticker(symbol)

        try:
            self.hist = pickle.load(open(self.hist_pkl_filename(
                self.symbol, self.start_date, self.end_date), "rb"))
        except (OSError, IOError) as e:
            self.hist = self.ticker.history(start=start_date, end=end_date)
            pickle.dump(self.hist, open(self.hist_pkl_filename(
                self.symbol, self.start_date, self.end_date), "wb"))

        self.trading_days = len(self.hist["Close"])

    def hist_pkl_filename(self, ticker, start_date, end_date):
        return f"{CACHE_DIR}/{ticker}_{start_date}_{end_date}.pkl"

    def has_expected_trading_days(self):
        return self.trading_days == EXPECTED_TRADING_DAYS


class Baseline():
    def __init__(self, start_date="2010-01-01", end_date="2022-01-01"):
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = {}
        self.all_sp500_tickers = sp500_tickers()
        self.tickers_to_analyze = []
        print("Retrieved current S&P500 tickers", self.all_sp500_tickers)
        for symbol in tqdm(self.all_sp500_tickers):
            print("Setting up", symbol)
            ticker = Ticker(symbol, start_date, end_date)
            self.tickers[symbol] = ticker
            if ticker.has_expected_trading_days():
                self.tickers_to_analyze.append(ticker)

    def compute_correlation(self):
        """Computes the Pearson product-moment correlation coefficients of all tickers
        """
        ticker_names = [ticker.symbol for ticker in self.tickers_to_analyze]
        print(
            f"Computing correlation for the following {len(ticker_names)} tickers", ticker_names)
        M = np.zeros(len(self.tickers_to_analyze), EXPECTED_TRADING_DAYS)
        for ticker in tqdm(self.tickers_to_analyze):
            ipdb.set_trace()
            pass


if __name__ == '__main__':
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    baseline = Baseline()
    baseline.compute_correlation()
