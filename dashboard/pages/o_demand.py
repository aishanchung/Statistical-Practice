import streamlit as st
import plotly.express as px
from dashboard.data_loader import bundle
from dashboard.components.view import downloads
data=bundle();choice=st.session_state['choice'];d=data['demand'];t=choice['time']
st.subheader('O 歷史預期需求')
mode=st.radio('需求建構方式',['P','P+Q'],horizontal=True)
methods={'ALL':'M1','Top20+OTHER':'M2','K=5':'M3','K=10':'M4','K=20':'M5'}
method=st.selectbox('分組方法',list(methods) if mode=='P' else ['Top20+OTHER'])
prefix=methods[method] if mode=='P' else 'B0_PQ';cols=[c for c in d if c.startswith(prefix+'__')]
z=d[d.target_time.dt.date.eq(t.date())]
st.caption('P為零延遲預期通過需求，P+Q為Q對齊後的預期抵達需求。兩者是替代建構，不可相加。僅使用2025-01-01前已成熟歷史。')
if z.empty:st.info('此日期沒有需求匯出。')
else:
    long=z[['target_time',*cols]].melt('target_time',var_name='群別',value_name='歷史預期需求');long['群別']=long['群別'].str.replace(prefix+'__','',regex=False)
    fig=px.line(long,x='target_time',y='歷史預期需求',color='群別',title=f'{method} {mode}，輛／五分鐘');fig.update_layout(font_family='Microsoft JhengHei');st.plotly_chart(fig,width='stretch');downloads(fig,long,'O_historical_demand')
