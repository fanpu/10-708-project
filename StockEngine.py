import pickle
import networkx as nx
import pandas as pd
import pyarrow
import os
import numpy as np
import pickle
from wordcloud import WordCloud
from tqdm import tqdm
from pprint import pprint
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import ipdb

import yfinance as yf
from tqdm import tqdm

class StockEngine():
    def __init__(self, tickers, start_date="2007-01-01", end_date="2022-12-31"):
        # Raw Data From yFinance
        self.raw_data = yf.download(' '.join(tickers), 
                                    start=start_date,
                                    end=end_date,
                                    group_by='ticker', 
                                    interval = "1d",
                                    actions=False)
        # Return Data
        ret_df = pd.DataFrame()
        for ticker in tqdm(tickers):
            ret_df[ticker] = (self.raw_data[ticker]['Adj Close']/self.raw_data[ticker]['Adj Close'].shift(1)-1)
        ret_df = ret_df[sorted(list(ret_df))]
        ## temporary solution
        # self.ret_df = ret_df.drop(['DVA', 'ULTA'],axis=1).dropna()
        self.ret_df = ret_df.dropna()
        self.tickers = list(self.ret_df)

        # Year Month
        year_month_list = []
        for year in np.arange(int(start_date[:4]),int(end_date[:4])+1):
            for month in np.arange(1,13):
              year_month_list.append(str(year)+'-'+str(month).zfill(2))
        self.year_month_list = year_month_list
        self.trading_day_list = list(self.ret_df.index.strftime('%Y-%m-%d'))
    
    def get_shift_month(self, year_month, diff):
        """
        Shift year month
        """
        idx = self.year_month_list.index(year_month) + diff
        assert(idx>=0)
        return self.year_month_list[idx]

    def get_shift_tday(self, tday, diff):
        """
        Shift trading days
        """
        idx = self.trading_day_list.index(tday) + diff
        assert(idx>=0)
        return self.trading_day_list[idx]

    def get_HAR_DRD(self, tday):
        """
        Get HAR DRD for the 21 days ending on given trading day
        """
        r = self.ret_df.loc[self.get_shift_tday(tday,-21*1+1):self.get_shift_tday(tday,0)]
        # print('Last Trading Day', self.get_shift_tday(tday,-21*1+1))
        # Ht = np.cov(r.T)
        # Rt = np.corrcoef(r.T)
        # RVt = np.diag(Ht)
        # Dt = np.diag(np.sqrt(RVt))

        Ht = np.zeros((r.shape[1], r.shape[1]))
        for i in range(r.shape[0]):
            Ht+=np.outer(r.iloc[i].values, r.iloc[i].values)
        RVt = np.diag(Ht)
        Dt = np.diag(np.sqrt(RVt))
        Dt_inverse = np.diag(1/np.sqrt(RVt))
        Rt = Dt_inverse @ Ht @ Dt_inverse
        
        return Ht, Rt, Dt, RVt

    def get_HAR_DRD_data(self, tday):
        """
        Get HAR DRD training data for the trading month (21 days) ending on a given trading day
        """
        Ht, Rt, Dt, RVt = self.get_HAR_DRD(tday)
        Ht1, Rt1, Dt1, RVt1 = self.get_HAR_DRD(self.get_shift_tday(tday, -21))

        il = np.tril_indices(Rt.shape[0], -1)
        xt = Rt[il]
        xt1 = Rt1[il]

        RVt25 = np.zeros_like(RVt)
        xt25 = np.zeros_like(xt)

        for i in range(2, 6):
            prev_tday = self.get_shift_tday(tday, -i*21)
            # print('Previous Trading Day:',prev_tday)
            Htp, Rtp, Dtp, RVtp = self.get_HAR_DRD(prev_tday)
            # print(prev_year_month)
            il = np.tril_indices(Rtp.shape[0], -1)
            xtp = Rtp[il]

            # Add up to explanatory variables
            RVt25 += 0.25*RVtp
            xt25 += 0.25*xtp

        RVt626 = np.zeros_like(RVt)
        xt626 = np.zeros_like(xt)
        # print('#####')
        for i in range(6, 23):
            prev_tday = self.get_shift_tday(tday, -i*21)
            # print('Previous Trading Day:',prev_tday)
            Htp, Rtp, Dtp, RVtp = self.get_HAR_DRD(prev_tday)
            # print(RVtp[:4])
            il = np.tril_indices(Rtp.shape[0], -1)
            xtp = Rtp[il]

            # Add up to explanatory variables
            RVt626 += RVtp/17
            xt626 += xtp/17
        return RVt, RVt1, RVt25, RVt626, xt, xt1, xt25, xt626

    
if __name__ == "__main__": 
    with open("adj_mat.pkl", "rb") as f:
        G = pickle.load(f)
        
    start_date = "2018-01-01"
    end_date = "2021-12-31"

    tickers = list(G.nodes())
    engine = StockEngine(tickers, start_date=start_date, end_date=end_date)