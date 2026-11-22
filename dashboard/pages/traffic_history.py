import pandas as pd
import streamlit as st
import plotly.express as px
from dashboard.data_loader import bundle
from dashboard.components.view import downloads
data=bundle();choice=st.session_state['choice'];t=choice['time'];start=t-pd.Timedelta(hours=6)
st.subheader('過去六小時路況')
vd=data['vd'];vd=vd[vd.available_at.le(t)&vd.observed_time.ge(start)]
for field,label in [('speed','VD速度（km/h）'),('occupancy','占有率（%）')]:
    fig=px.line(vd,x='observed_time',y=field,color='site',title=label+'，僅顯示預測時已可取得資料');fig.update_layout(font_family='Microsoft JhengHei');st.plotly_chart(fig,width='stretch');downloads(fig,vd,'history_'+field)
y=data['y'];y=y[y.time.between(start,t-pd.Timedelta(minutes=10))]
fig=px.line(y,x='time',y='y',line_shape='hv',title='事後計算的歷史路段Y（截至可用時間）');st.plotly_chart(fig,width='stretch')
