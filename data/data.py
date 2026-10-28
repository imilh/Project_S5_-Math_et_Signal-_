import pandas as pd

CSV_PATH = "data/strasbourg_entzheim.csv"
def load_data(path=CSV_PATH): 
    df = pd.read_csv(path , parse_dates=["time"], index_col="time")  #parse_dates turns the dates to something the computer understands and index_col takes the time and turns it from cols to the row labels 
    df = df.sort_index()    # this is used to sort by date btw
    df = df.asfreq("D")    # used to take one row per day hehe
    return df

def load_series(var="tavg", path=CSV_PATH):
    df = load_data(path)
    s = df[var]
    n_missing = s.isna().sum()  #isna is used to turn the series into true or false (true if a value is missing) sum counts the true values where the result is the result of the missing days but for tavg and prcp it is 0
    if n_missing > 0 :
        s = s.interpolate(limit=5) #fills a gap by drawing a straight line between each known vals from both sides
                    #a gap of 1-5 days is short and the weather barely changes in that time
    n_left = s.isna().sum()
    if n_left > 0 :
        print(f"Warning : {n_left} NaN left in {var}")  # if even after cleaning we still have NaN it will tell that the var is not clean instead of getting wrong results silently
    return s