import streamlit as st
import plotly.express as px
from dashboard.data_loader import bundle
from dashboard.components.view import downloads
data=bundle();m=data['metrics'];h=st.session_state['choice']['horizon']
st.subheader('已完成模型比較')
split=st.radio('評估範圍',['2024時序驗證','2025探索性回測'],horizontal=True)
z=m[m.horizon.eq(h)&m.segment.eq('ALL')&(m.fold.eq('EXP2025') if split.startswith('2025') else m.fold.ne('EXP2025'))]
z=z.groupby(['algorithm','configuration'],as_index=False).agg(AP2=('average_precision_2','mean'),Macro_F1=('macro_f1','mean'),Log_loss=('log_loss','mean'),Y1_Recall=('recall_1','mean'),Y2_Recall=('recall_2','mean'),n=('n','sum'))
st.caption('2024顯示各折指標平均，n為各折有效樣本數加總。只有已完成並保存機率的模型入表。2025不參與選K。')
st.dataframe(z,width='stretch',hide_index=True)
fig=px.bar(z,x='configuration',y='AP2',color='algorithm',barmode='group',title=f'{split}，h={h}分鐘，Y2 Average Precision');fig.update_layout(font_family='Microsoft JhengHei');st.plotly_chart(fig,width='stretch');downloads(fig,z,'model_comparison')
