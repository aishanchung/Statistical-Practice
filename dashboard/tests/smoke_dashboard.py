"""Run against real exported data. Saves UI checks outside Git."""
import json
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'experiments/congestion_2026/outputs/workpackages_0ABCDE_v1'
checks=[]
a=AppTest.from_file(str(ROOT/'dashboard/app.py'),default_timeout=60).run()
checks.append({'page':'Overview','exceptions':[x.message for x in a.exception],'metrics':[(x.label,x.value) for x in a.metric]})
assert not a.exception
a.selectbox(key='horizon_choice').set_value(15).run()
checks.append({'action':'horizon15','exceptions':[x.message for x in a.exception]});assert not a.exception
for page in ['traffic_history','o_demand','comparison','event_replay']:
    a.switch_page('pages/'+page+'.py').run()
    checks.append({'page':page,'exceptions':[x.message for x in a.exception]})
    assert not a.exception,checks[-1]
(OUT/'dashboard_ui_tests.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps(checks,ensure_ascii=False))
