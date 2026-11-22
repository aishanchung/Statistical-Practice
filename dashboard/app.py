from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import streamlit as st
from dashboard.data_loader import settings,bundle
from dashboard.components.view import controls

st.set_page_config(page_title='國三北向壅塞歷史回放',page_icon='🚗',layout='wide')
cfg=settings();st.title(cfg['title']);st.warning('Historical Replay / Offline Evaluation　歷史回放與離線評估，尚未串接即時 VD 或即時 O 車流。')
data=bundle();st.session_state['choice']=controls(data)
st.caption('Y版本　'+data['metadata']['Y_version']+'　｜　模型歷史截止2025-01-01　｜　2025為探索性回測')
pages=[st.Page('pages/overview.py',title='Overview',default=True),st.Page('pages/traffic_history.py',title='Traffic History'),st.Page('pages/o_demand.py',title='O Historical Demand'),st.Page('pages/comparison.py',title='Prediction Comparison'),st.Page('pages/event_replay.py',title='Event Replay')]
st.navigation(pages).run()
