import pandas as pd
import streamlit as st
import plotly.express as px
from dashboard.data_loader import bundle,rows_at
from dashboard.components.view import overview,downloads
data=bundle();choice=st.session_state['choice'];events=data['events'];events=events[events.qualifying&events.onset.dt.year.eq(2025)].sort_values('onset')
examples=data['examples'];st.subheader('真實歷史事件回放')
st.caption('本頁採事件回放時點，側欄日期供其他頁面使用。案例均來自真實回測，R不視為成功預警。')
st.caption('案例標籤依 LightGBM Baseline、D0、h=30 選取；切換其他模型時僅作同事件回放對照。')
example_view=examples[['case','event_time']].copy();example_view['case']=example_view['case'].map({'success':'成功預警','miss':'漏報','Y1':'局部壅塞'})
example_view['選例方式']='依固定規則選取最早乾淨案例'
st.dataframe(example_view.rename(columns={'case':'案例','event_time':'事件時間'}),width='stretch',hide_index=True)
eventid=st.selectbox('歷史Y2事件',events.event_id,format_func=lambda e:pd.to_datetime(e,format='%Y%m%d_%H%M').strftime('%Y-%m-%d %H:%M'))
event=events[events.event_id.eq(eventid)].iloc[0]
times=pd.date_range(event.onset-pd.Timedelta(minutes=40),event.onset+pd.Timedelta(minutes=40),freq='5min')
case_predictions=data['predictions']
case_predictions=case_predictions[case_predictions.algorithm.eq(choice['algorithm'])&case_predictions.configuration.eq(choice['configuration'])&case_predictions.horizon.eq(choice['horizon'])&case_predictions.prediction_time.ge(event.onset-pd.Timedelta(minutes=30))&case_predictions.prediction_time.lt(event.onset)&case_predictions[choice['method']].eq(2)]
default_time=case_predictions.prediction_time.min() if len(case_predictions) else times[4]
replay=st.select_slider('回放時點',options=list(times),value=default_time,key='replay_'+eventid,format_func=lambda t:t.strftime('%H:%M'))
overview(data,choice,replay)
p=data['predictions'];p=p[p.algorithm.eq(choice['algorithm'])&p.configuration.eq(choice['configuration'])&p.horizon.eq(choice['horizon'])&p.prediction_time.between(times[0],times[-1])]
if not choice['answers']:p=p[p.prediction_time.le(replay)]
fig=px.line(p,x='prediction_time',y='p2',title='實際保存的Y2預測機率，橫軸為發出時間')
hard=p[p[choice['method']].eq(2)]
fig.add_scatter(x=hard.prediction_time,y=hard.p2,mode='markers',name='Y2硬警報',marker={'symbol':'triangle-up','color':'#af5448','size':9})
if choice['answers']:fig.add_vline(x=event.onset.isoformat(),line_dash='dash',line_color='#af5448')
fig.update_xaxes(range=[times[0].isoformat(),times[-1].isoformat()],title='預測發出時間')
fig.update_yaxes(title='p(Y2)',range=[0,1])
st.plotly_chart(fig,width='stretch');downloads(fig,p.drop(columns=['y']) if not choice['answers'] else p,'event_predictions')
if choice['answers']:
    st.warning('以下為事後真值，不是預測時已知資訊。')
    y=data['y'];y=y[y.time.between(times[0],times[-1])];fig=px.line(y,x='time',y='y',line_shape='hv',title='事後實際Y及onset');fig.add_vline(x=event.onset.isoformat(),line_dash='dash',line_color='#af5448');fig.update_yaxes(tickvals=[0,1,2],range=[-.1,2.1]);st.plotly_chart(fig,width='stretch')
