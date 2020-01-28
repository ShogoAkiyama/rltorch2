import datetime
import pandas as pd
import numpy as np
import argparse
from utils.func import norm_time


if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--train_date', type=int, default=2015)
   parser.add_argument('--test_date', type=int, default=2016)
   parser.add_argument('--code', type=list, default=['7203'])
   # parser.add_argument('--code', type=list, default=['7203', '9984'])
   # parser.add_argument('--code', type=list, default=['8301', '7203', '9501', '6758', '9984',
   #                                                   '8306', '8411', '6501', '6752', '6502',
   #                                                   '7201', '9983', '6753', '8604', '7267',
   #                                                   '8316', '5401', '6702', '9503', '7974'])

   args = parser.parse_args()

   code_list = [str(x) for x in args.code]

   for i, date in enumerate(range(2011, 2020)):
      tmp = pd.read_csv('./data/news/' + str(date) + '.csv', encoding='cp932')
      tmp = tmp[tmp['Company_IDs(TSE)'].isin(code_list)]
      # tmp = tmp[tmp['Company_Relevance'] == str(100)]
      tmp = tmp[['Time_Stamp_Original(JST)', 
                        'Company_Code(TSE)', 
                        'Headline', 
                        'News_Source',
                        'Company_Relevance', 
                        'Keyword_Article']]

      # 欠損除去
      tmp = tmp[~tmp["Keyword_Article"].isnull()]

      # タグ除去
      tmp = tmp[(tmp['News_Source'] == '日経') | 
                     (tmp['News_Source'] == 'ＮＱＮ') |
                     (tmp['News_Source'] == 'ＱＵＩＣＫ') | 
                     (tmp['News_Source'] == 'Ｒ＆Ｉ')]

      tmp['code'] = tmp['Company_Code(TSE)'].astype(int)
      tmp['date'] = pd.to_datetime(tmp["Time_Stamp_Original(JST)"]).map(norm_time)
      tmp = tmp.set_index(['date', 'code'], drop=True)
      tmp = tmp.drop(['Time_Stamp_Original(JST)', 'Company_Code(TSE)'], axis=1)

      if i == 0:
         df1 = tmp.copy()
      else:
         df1 = pd.concat([df1, tmp])
   print(df1.shape)

   # 株価を取り出す
   for i, code in enumerate(code_list):
      tmp = pd.read_csv('./data/stock_price/' + str(code) + '.csv', index_col=0)
      tmp['code'] = int(code)
      if i == 0:
         df2 = tmp
      else:
         df2 = pd.concat([df2, tmp])

   df2['date'] = pd.to_datetime(df2['date'])
   df2 = df2.set_index(['date', 'code'], drop=True)
   print(df2.shape)

   # 時系列をくっつける
   df3 = pd.concat([df1,df2], axis=1, join_axes=[df1.index], levels=[0,1])
   df3 = df3.sort_values(by=['code', 'date'])
   df3['Keyword_Article'] = \
   df3.groupby(level=[0,1]).apply(lambda x: ':<pad>:'.join(list(x['Keyword_Article'])))

   df3 = df3.dropna()
   df3 = df3[~df3.duplicated(subset=['Keyword_Article'])]
   df3['price'] = \
         df3['adj_close'].groupby(level=['code']).pct_change(1).shift(-1)*100
   df3 = df3.dropna()

   # CSVファイルに保存する
   df4 = pd.concat([df3[['Keyword_Article', 'price']].rename(
                                       columns={'Keyword_Article': 'state', 'price': 'reward'}),
                                 df3[['Keyword_Article']].shift(-1).rename(
                                       columns={'Keyword_Article': 'next_state'})], axis=1).dropna()
   df4 = df4[['state', 'next_state', 'reward']]

   date_year = df4.index.map(lambda x: x[0].year)

   df4[date_year <= args.train_date].to_csv(
        './data/news/text_train.tsv',
        header=None,
        index=None,
        sep='\t')

   df4[(args.train_date < date_year) & (date_year < args.test_date)].to_csv(
        './data/news/text_val.tsv',
        header=None,
        index=None,
        sep='\t')

   df4[(args.test_date <= date_year)].to_csv(
        './data/news/text_test.tsv',
        header=None,
        index=None,
        sep='\t')
