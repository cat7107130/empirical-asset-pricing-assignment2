import pandas as pd, numpy as np
f="data.csv"
cols=pd.read_csv(f,nrows=0).columns.tolist()
i0,i1=cols.index("div12m_me"),cols.index("qmj_safety")
char_cols=cols[i0:i1+1]
print("n_total_cols",len(cols),"n_char_cols(div12m_me..qmj_safety)",len(char_cols))
keep=["eom","crsp_exchcd","common","excntry","obs_main","primary_sec","exch_main","ret_exc_lead1m","market_equity","ff49"]
funnel={"total":0,"common":0,"usa":0,"nyse":0,"obs_flags":0,"ret_notnull":0}
yr_nyse={}
for ch in pd.read_csv(f,usecols=keep,chunksize=500000):
    funnel["total"]+=len(ch)
    c=ch["common"]==1
    funnel["common"]+=c.sum()
    u=c&(ch["excntry"]=="USA")
    funnel["usa"]+=u.sum()
    n=u&(ch["crsp_exchcd"]==1)
    funnel["nyse"]+=n.sum()
    o=n&(ch["obs_main"]==1)&(ch["primary_sec"]==1)&(ch["exch_main"]==1)
    funnel["obs_flags"]+=o.sum()
    r=o&ch["ret_exc_lead1m"].notna()
    funnel["ret_notnull"]+=r.sum()
    sub=ch[r]
    yr=pd.to_datetime(sub["eom"]).dt.year
    for y,cnt in yr.value_counts().items(): yr_nyse[y]=yr_nyse.get(y,0)+cnt
print("FUNNEL",funnel)
ys=sorted(yr_nyse)
print("year range",ys[0],ys[-1])
print("sample yearly final-sample counts:")
for y in [ys[0]]+[y for y in ys if y%5==0]+[ys[-1]]:
    print(" ",y,yr_nyse[y])
print("total final-sample firm-months",sum(yr_nyse.values()))
