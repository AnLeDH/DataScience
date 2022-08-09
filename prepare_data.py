import pandas as pd
import numpy as np



def prepare_data():
    #read raw data into pandas Dataframe 
    df_base = pd.read_excel("C:\\Users\\anna-\\Nextcloud2\\DHBW\\2. Sem\\DataScienceProject\\dataexport_20220730T081818.xlsx", engine='openpyxl')
    #drop first 9 unneeded columns
    df_base = df_base.drop(labels = np.arange(0,9), axis = 0)
    #reset index (starting from 0)
    df_base = df_base.reset_index(drop=True)
    #rename columns
    df_base.set_axis(["day","precipitation","temp_min","temp_max"], axis=1, inplace=True)
    #format timestamp in raw data and insert new columns for month and day 
    df_base.insert(0, "month", df_base["day"].astype('string').str.replace('00:00:00','').str.split('-').str[1].astype('int'))
    df_base.insert(0, "year", df_base["day"].astype('string').str.replace('00:00:00','').str.split('-').str[0].astype('int'))
    df_base["day"] = df_base["day"].astype('string').str.replace('00:00:00','').str.split('-').str[2].astype('int')
    df_base = df_base.dropna(axis=0)
    
    
    """ create DataFrames for average data """
    years = np.arange(1984,2023)
    months = np.arange(1, 13)
    
    df_av_yearly = {"index":[], "year":[],"temp_min":[], "temp_max":[], "precipitation":[]}
    df_av_monthly = {"index":[],"y-m": [], "month":[], "temp_min":[], "temp_max":[], "precipitation":[]}
    index_years = 1
    index_months = 1
    
    for y in years:
        df_av_yearly["index"].append(index_years)
        df_av_yearly["year"].append(y)
        df_av_yearly["temp_min"].append((df_base["temp_min"][df_base["year"] == y]).mean())
        df_av_yearly["temp_max"].append((df_base["temp_max"][df_base["year"] == y]).mean())
        df_av_yearly["precipitation"].append((df_base["precipitation"][df_base["year"] == y]).mean())
        index_years += 1
        
        for m in months:
            df_av_monthly["index"].append(index_months)
            df_av_monthly["y-m"].append(f"{y}-{m}")
            df_av_monthly["month"].append(m)
            df_av_monthly["temp_min"].append((df_base["temp_min"][(df_base["year"] == y) & (df_base["month"] == m)]).mean())
            df_av_monthly["temp_max"].append((df_base["temp_max"][(df_base["year"] == y) & (df_base["month"] == m)]).mean())
            df_av_monthly["precipitation"].append((df_base["precipitation"][(df_base["year"] == y) & (df_base["month"] == m)]).mean())
            index_months += 1
            
    df_av_yearly = pd.DataFrame.from_dict(df_av_yearly).dropna(axis=0)
    df_av_monthly = pd.DataFrame.from_dict(df_av_monthly).dropna(axis=0)
    
    return df_av_yearly, df_av_monthly


