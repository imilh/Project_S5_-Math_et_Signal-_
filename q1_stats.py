import numpy as np
import matplotlib.pyplot as plt 
from data import load_series

plt.rcParams.update({'font.size': 12, "figure.dpi" : 100})
Vars = {"tavg": ("Mean temperature", "°C"), "prcp": ("Precipitation", "mm")}

for var ,(name, unit) in Vars.items():
    s = load_series(var)
    
    # ------ 1- descriptive.stats -------
    
    print(f"\n === {name} ({var}) ===")
    print(f"Mean : {s.mean(): .2f} {unit} | std {s.std():.2f} |"     #std is for standard deviation
          f"Min : {s.min():.1f} ({s.idxmin().date()}) | Max : {s.max():.1f} ({s.idxmax().date()})")   #idmin and idmax are used to find the indexs which are the dates of the smallest and the largest values 
    if var == "prcp" :
        print(f" Number of dry days : {(s==0).mean()*100:.1f} % | Skewness : {s.skew():.2f}")  #this is used to find the number of dry days by counting the number of days where the precipitation is 0 and then dividing it by the total number of days to get a percentage
        #skweness is caluclated to find the asymmetry of the distribution of the data. A positive skewness that-> data is skewed to the right | a negative skewness -> skewed to the left. A skewness of 0 -> a symmetric distribution.
    
    
    
    # ------ 2- time series + 365-day rolling mean -------
    
    
    fig, ax = plt.subplots(figsize= (10, 4))   #used to create a figure and a plotting area (ax) with a size of 10x4 inches
    ax.plot(s.index, s.values, label= "daily", lw =0.3, alpha= 0.6)   # x axes is the time and y axes is the vals while the label of the graph is daily nd the opacity is 0.6 and the line width is 0.3
    ax.plot(s.index, s.rolling(365, center=True).mean(),  #we calculating the rolling mean for 365 days meaaaans that we are taking the average of the values for each day and the 182 days before and after it. This is done to smooth out the data and make it easier to see trends over time.
        color="red", lw=1.5, label="365-day mean")   #color of the rolling mea is redddd with a line width of 1,5 
    ax.set(xlabel = "Year", ylabel = f"{name} ({unit})", title = f"{name} 1950-2024")   #used to name the x and the y axes alsooooo the graph
    ax.legend()   #Displays a legend which means explains what each line represents in the graph
    fig.tight_layout()  #manages space between subplots automatically to make sure that the labels and titles do not overlap with each other or with the axes 
    
    fig.savefig(f"figures/q1_{var}_series.png", dpi=200)   #save the graph as an image in the figures folder with a dpi of 200
    plt.close(fig)    #Closes the figure and releases resources used by Matplotlib.
    
    
    # ------ 3- histogram ------
    
    
    fig, ax = plt.subplots(figsize= (6, 4))
    ax.hist(s.values, bins=60, edgecolor="k", lw=0.3)  #used to create a histogram of vals in s with intervals(bins) = 60 and a black color for the edges but alsooo 0.3 line width for the edges
    ax.set(xlabel = f"{name} ({unit})",
           ylabel = "Number of days",
           title = f"Histogram of {var}")   #adds lables and title to the histogram
    if var == "prcp" :
        ax.set_yscale("log")   # If the variable is precipitation (prcp), use a logarithmic scale on the Y-axis
        #Precipitation datasets often contain many days with zero or low rainfall and relatively few days with extreme rainfall
        #A logarithmic scale helps visualize bars with very different frequencies
    fig.tight_layout()  #automatically adjusts the spacing between labels, titlles and axes to prevent them from overlapping
    fig.savefig(f"figures/q1_{var}_hist.png", dpi=200)   #save the histogram as a png image in the figures folder with a dpi of 200
    plt.close(fig)   #Closes the figure and releases resources used by Matplotlib.
    
    
    
    # ------ 4- trend on annual means  ------
    
         
         #preforming a linear trend analysis of climate data. Its purpose is to determine whether temperature or precipitation has increased or decreased over the years, and by how much.
    annual = s.resample("YE").mean() if var == "tavg" else s.resample("YE").sum()   #ternary expression is used to resample(group data by calendar year) the data to annual frequency( if tavg -> mean of the year | if prcp -> sum of the year)
    years = annual.index.year.values  #extracts the years from index, where these years become the x axis vals for the regression 
    a, b = np.polyfit(years, annual.values, 1)   #np.polyfit() finds the best polynomial that fits the data where 1 means the degree (linear regression y = ax + b)
    #y: predicted annual temperature or precipitation
    #x: years
    #a: slope of the line (rate of change per year)
    #b: y-intercept (value of y when x=0)
    r = np.corrcoef(years, annual.values)[0, 1]  #np.corrcoef() used to calc the correlation coeff between years and annual values
    #r=1: perfect positive linear correlation. r = -1: perfect negative. r = 0: no linear correaltion
    lab = "annual mean" if var == "tavg" else "annual sum"
    print(f"trend of {lab}: {a*10:+.3f} {unit}/decade  (r = {r:.2f}, R² = {r**2:.2f})") # lab displays the label | a*10 to convert the slope from per year to per decade | +.3f Displays 3 decimal places and an explicit + or - sign | {r:.2f} Displays correlation with 2 decimal places | {r**2:.2f} Displays R² with 2 decimal places
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(years, annual.values,"o-", ms = 3, label= lab)   #plots the annual values with a line and circular markers connected by solid lines(-o) where ms = marker size = 3 points and the label is the lab variable
    ax.plot(years, a*years + b,"r--",label=f"fit: {a*10:+.2f} {unit}/decade")   # r-- red dashed line for the fitted linear regression
    #a*years + b: predicted values from the fitted straight line  | label=___  displays the slope in the legend
    ax.set(xlabel = "Year", ylabel = f"{lab} ({unit})", title =f"Trend of {var}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/q1_{var}_trend.png", dpi=200)
    plt.close(fig)
    
    
    # ------ 5- seasonality (mean by day of year +- 1 std) -------
    
    #analyzing the seasonal patterns in climate data such as temperature or precipitation changes throughout the year in a pattern that tends to repeat every year
    
    g = s.groupby(s.index.dayofyear)  # groups the data by day of the year: .dayofyear converts each date to a number (1-366)
    # all observations with day number 1 go together, then all with number 2, ...
    m, sd = g.mean().iloc[:365], g.std().iloc[:365]  # mean and std of each group, keeping days 1-365 only
    # day 366 exists only in leap years (19 values instead of 75), so its mean is unreliable and we drop it
    # a small sd means the same calendar day is similar from year to year, a large sd means it varies a lot
    
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(m.index, m.values, label="mean")   # plots the average temperature for each day of the year
    ax.fill_between(m.index, m.values - sd.values, m.values + sd.values, alpha=0.25, label="± 1 std")   #shades the area between the mean minus one standard deviation and the mean plus one standard deviation with an opacity of 0.25
    ax.set(xlabel="Day of year",
       ylabel=f"{name} ({unit})",
       title=f"Seasonality of {var}")

    ax.legend()
    fig.tight_layout()
    fig.savefig(f"figures/q1_{var}_season.png", dpi=200)
    plt.close(fig)
    
    if var == "tavg":
        print(f"seasonal amplitude (max-min of daily climatology): {m.max()-m.min():.1f} °C "
             f"(coldest day-of-year {m.idxmin()}, warmest {m.idxmax()})")
    else:
        mm = s.groupby(s.index.month).mean() * 30.4
        print("monthly mean precipitation (mm/month):",
            mm.round(0).astype(int).tolist())