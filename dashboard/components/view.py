from io import BytesIO
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from dashboard.data_loader import snapshot_state,rows_at
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib import dates as mdates
import threading
PNG_LOCK=threading.Lock()

NAMES={'M0':'Baseline','M1':'ALL + P','M2':'Top20 + P','M3':'K=5 + P','M4':'K=10 + P','M5':'K=20 + P'}
def controls(data):
    p=data['predictions'];pairs=p[['algorithm','configuration']].drop_duplicates().sort_values(['algorithm','configuration'])
    labels=[a+' / '+NAMES.get(b,b) for a,b in pairs.itertuples(index=False,name=None)]
    index=st.sidebar.selectbox('模型',range(len(labels)),format_func=lambda i:labels[i],key='model_choice')
    alg,config=pairs.iloc[index]
    horizon=st.sidebar.selectbox('預測長度（分鐘）',[30,15],key='horizon_choice')
    method=st.sidebar.selectbox('預警決策',['D0','D1','D2'],format_func=lambda x:{'D0':'原始 argmax','D1':'Balanced 成本敏感','D2':'低信心Y0 待複核'}[x],key='method_choice')
    available=p[p.algorithm.eq(alg)&p.configuration.eq(config)&p.horizon.eq(horizon)]
    dates=sorted(available.prediction_time.dt.date.unique())
    observed_dates=set(data['y'].time.dt.date)
    default_date=next((d for d in dates if d in observed_dates),dates[0])
    date=st.sidebar.date_input('歷史日期',value=default_date,min_value=dates[0],max_value=dates[-1],key='date_choice')
    slots=sorted(available.loc[available.prediction_time.dt.date.eq(date),'prediction_time'].unique())
    if not slots:st.warning('該日期沒有符合研究定義且有有效目標的預測，請選研究假日或相鄰晚間。');st.stop()
    default_slot=min(range(len(slots)),key=lambda i:abs(pd.Timestamp(slots[i]).hour*60+pd.Timestamp(slots[i]).minute-720))
    time=st.sidebar.selectbox('五分鐘預測時點',slots,index=default_slot,format_func=lambda t:pd.Timestamp(t).strftime('%H:%M'),key='time_choice')
    answers=st.sidebar.checkbox('顯示事後答案（分析用）',value=False,key='show_answers')
    st.sidebar.caption('研究資料不是即時串流。未來實際Y不參與預測。')
    return {'algorithm':alg,'configuration':config,'horizon':horizon,'method':method,'time':pd.Timestamp(time),'answers':answers}
def overview(data,choice,t=None):
    t=pd.Timestamp(t or choice['time']);weights=data['metadata']['config']['Y']['weights']
    readings,c,y,coverage=snapshot_state(data['vd'],t,weights)
    st.subheader(f'預測發出　{t:%Y-%m-%d %H:%M}')
    st.caption('當時可知資訊採 t−10 分鐘的VD，假設10分鐘發布延遲。缺值不回填，RECOVERED_P1不作輸入。')
    cols=st.columns(3);cols[0].metric('最近可觀測 C',f'{c:.3f}' if np.isfinite(c) else 'NA');cols[1].metric('最近可觀測 Y',str(int(y)) if np.isfinite(y) else 'NA');cols[2].metric('有效空間權重',f'{coverage:.1%}')
    st.dataframe(readings[['site','observed_time','available_at','speed','occupancy','source_status']].rename(columns={'site':'VD站','observed_time':'量測時間','available_at':'假設可用時間','speed':'速度 km/h','occupancy':'占有率 %','source_status':'來源狀態'}),hide_index=True,width='stretch')
    row=rows_at(data,choice,t)
    if row.empty:st.warning('這個回放時點沒有該模型的有效預測。');return
    r=row.iloc[0];prob=pd.DataFrame({'類別':['Y0 暢通','Y1 局部壅塞','Y2 大範圍壅塞'],'預測機率':[r.p0,r.p1,r.p2]})
    fig=px.bar(prob,x='類別',y='預測機率',range_y=[0,1],text_auto='.1%',color_discrete_sequence=['#087f8c']);fig.update_layout(title=f'預測目標 {r.target_time:%H:%M}（h={int(r.horizon)}分鐘）',font_family='Microsoft JhengHei',height=330)
    st.plotly_chart(fig,width='stretch');downloads(fig,prob,'probabilities')
    decision='R 待複核' if r.decision==-1 else f'Y{int(r.decision)} '+{0:'無觀測壅塞',1:'局部壅塞',2:'大範圍壅塞警報'}[int(r.decision)]
    st.info('最終決策　'+decision+'　｜　'+choice['method'])
    if choice['answers']:st.warning(f'事後驗證答案　{r.target_time:%Y-%m-%d %H:%M} 實際Y={int(r.y)}。這項答案在預測時尚未知。')
def downloads(fig,frame,key):
    st.download_button('下載圖表CSV',frame.to_csv(index=False).encode('utf-8-sig'),key+'.csv',mime='text/csv',key=key+'_csv')
    if st.button('產生圖表PNG',key=key+'_png_make'):
        try:
            # Native scientific plot export avoids a browser subprocess in the
            # Streamlit worker. Data are the very same Plotly trace arrays.
            with PNG_LOCK:
                f=Figure(figsize=(12,5.5),dpi=160);FigureCanvasAgg(f);ax=f.subplots()
                f.set_layout_engine('constrained')
                for i,trace in enumerate(fig.data):
                    x=list(trace.x);y=list(trace.y);label=trace.name or None
                    color=['#087f8c','#af5448','#16344a','#a38f55'][i%4]
                    if trace.type=='bar':
                        positions=np.arange(len(x));width=.8/max(1,len(fig.data));ax.bar(positions+i*width,y,width=width,label=label,color=color);ax.set_xticks(positions+.4-width/2,[str(v) for v in x])
                    else:ax.plot(x,y,label=label,color=color)
                ax.set_title(fig.layout.title.text or '',fontfamily='Microsoft JhengHei')
                ax.set_xlabel(fig.layout.xaxis.title.text or '',fontfamily='Microsoft JhengHei');ax.set_ylabel(fig.layout.yaxis.title.text or '',fontfamily='Microsoft JhengHei')
                for text in [*ax.get_xticklabels(),*ax.get_yticklabels()]:text.set_fontfamily('Microsoft JhengHei')
                if len(fig.data)>1:ax.legend(prop={'family':'Microsoft JhengHei','size':9})
                if fig.layout.yaxis.range:ax.set_ylim(fig.layout.yaxis.range)
                ax.grid(alpha=.15);buffer=BytesIO();f.savefig(buffer,format='png',bbox_inches='tight');raw=buffer.getvalue()
            st.download_button('下載PNG',raw,key+'.png',mime='image/png',key=key+'_png')
        except Exception as e:st.error('PNG匯出失敗，原始CSV與既有研究圖仍可使用。'+str(e)[:150])
