import os
import yfinance as yf
import pandas as pd
import datetime

def get_exo_df(start_date, end_date):
    '''
    Returns a dataframe of the exogenous signals
    :param start_date:
    :param end_date:
    :return:
    '''
    if not isinstance(start_date, datetime.datetime):
        start_date = pd.to_datetime(start_date)
    if not isinstance(end_date, datetime.datetime):
        end_date = pd.to_datetime(end_date)

    rs = pd.DataFrame(index = pd.date_range(start=start_date,end=end_date, freq='1d'))

    # get the credit spread daily change
    credit_spread = pd.read_csv(os.path.join('data', 'BAMLH0A0HYM2.csv'), index_col=0, parse_dates=True)

    # get the difference between 10Y and 2Y bond yields
    diff_10Y_2Y = pd.read_csv(os.path.join('data', 'T10Y2Y.csv'), index_col=0, parse_dates=True)

    # get the 10yr bond yield daily change
    ten_year_bonds = yf.Ticker("^TNX")
    yld_10Y = ten_year_bonds.history(start=start_date, end=end_date).Close.tz_localize(None)

    # build output
    rs = pd.concat([rs, credit_spread, diff_10Y_2Y, yld_10Y], axis=1, join='outer')
    rs.columns = ['credit_spread', 'diff_10Y_2Y', 'yld_10Y']
    rs = rs.ffill()
    rs.index.name = 'date'
    return rs