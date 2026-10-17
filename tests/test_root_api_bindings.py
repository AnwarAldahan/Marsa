"""Current legacy API bindings in the unchanged root frontend design."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "Marsa updated.html"
NODE = shutil.which("node")
HARNESS = r"""
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const html=fs.readFileSync(process.argv[1],'utf8'),mode=process.argv[2];
const source=[...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m=>m[1]).find(s=>s.includes('MARSA CURRENT API BINDINGS'));
const pure={module:{exports:{}}};vm.runInNewContext(source,pure);const api=pure.module.exports;
const row={candidate_id:'baseline',rank:1,kpis:{open_berths:2,avg_wait_hours:0,max_wait_hours:3,vessels_berthed:1},vessels:[{call_id:'W',arrival_h:-1,berth_h:2,service_h:2},{call_id:'F',arrival_h:4,berth_h:null,service_h:3}],timeline:[{waiting:1,at_berth:1,yard_occupancy_pct:98}]};
const data=ts=>({authoritative_timestamp:ts,decision_support:{highest_ranked_candidate:'baseline'},digital_twin:{simulation_horizon_hours:6,scenario:{open_berths:2,vessels_at_berth:[{call_id:'B',leaves_in_h:5}]},results:[row]},strategy_candidates:[{candidate_id:'baseline',actions:[]},{candidate_id:'extra_berth',actions:[{type:'add_berths',value:1}]}]});
async function main(){
 if(mode==='kpi'){
  assert.equal(api.kpi(row,'avg_wait_hours'),0);assert.equal(api.kpi(row,'missing'),null);assert.equal(api.kpi({kpis:{avg_wait_hours:{mean:9}}},'avg_wait_hours'),null);
  const d=data('2025-07-10T19:00:00Z');d.digital_twin.results.unshift({candidate_id:'user_decision',rank:1});assert.equal(api.winner(d),'baseline');assert.equal(api.esc('<img>'),'&lt;img&gt;');
 }else if(mode==='forecast'){
  assert.equal(api.forecast({status:'success',congestion_score:0}).score,0);assert.equal(api.forecast({status:'ml_unavailable',congestion_score:.9}).score,null);
  assert.equal(Object.keys(api.forecast({})).join(','),'score,target');
 }else if(mode==='actions'){
  const d=data('2025-07-10T19:00:00Z');assert.equal(api.actionRequest(d,'shortest',3).user_actions.length,2);assert.equal(api.actionRequest(d,'fcfs',1).stress.berths_closed,1);
  for(const rule of ['own','random','plan:cranes'])assert.throws(()=>api.actionRequest(d,rule,2));assert.throws(()=>api.actionRequest(d,'plan:extra_berth',3));assert.throws(()=>api.actionRequest(d,'fcfs',1.5));
 }else if(mode==='manual'){
  const r=api.manualResult(data('2025-07-10T19:00:00Z'),row);assert.equal(r.avgWait,0);assert.equal(r.utilization,null);assert.equal(r.berthIntervals[0][0].end,5);assert.equal(r.berthIntervals[1][0].start,2);assert.equal(r.vessels.find(v=>v.id==='F').assignTime,null);
 }else{
  const nodes={},pending=[];
  const get=id=>nodes[id] ||= {value:'',innerHTML:'',textContent:'',hidden:true,disabled:false,style:{},firstChild:{textContent:''},classList:{add(){},remove(){},toggle(){}},querySelector(){return get(id+'child')},querySelectorAll(){return []},addEventListener(){},setAttribute(){}};
  let script=source.replace("  load($('mlPreset').value);",`flow=timeline=factors=alert=cards=()=>{};globalThis.audit={load,runManual,state:()=>DATA,now:()=>{manualMode='now'},select:(id,h)=>{selected=id;hour=h;frame()}};`);
  const context={document:{getElementById:get,querySelector:get,querySelectorAll:()=>[],addEventListener(){}},fetch:(url,options)=>new Promise((resolve,reject)=>pending.push({url,options,resolve,reject})),setInterval(){},clearInterval(){},console,MARSA_NORMALIZE:s=>s,MARSA_DRAWMAP:s=>{context.drawn=s}};context.window=context;
  vm.runInNewContext(script,context);const a=context.audit,finish=(r,d)=>r.resolve({ok:true,json:async()=>d});
  const A='2025-07-10T19:00:00Z',B='2025-11-05T12:00:00Z';
  if(mode==='stale_success'||mode==='stale_error'){
   const old=a.load(A),current=a.load(B);finish(pending[1],data(B));await current;if(mode==='stale_error')pending[0].reject(Error('old error'));else finish(pending[0],data(A));await old;assert.equal(a.state().authoritative_timestamp,B);
  }else{
   const load=a.load(A);finish(pending[0],data(A));await load;
   if(mode==='failure'){
    get('twCards').innerHTML='old results';const failed=a.load(B);pending[1].reject(Error('unavailable'));await failed;assert.equal(a.state(),null);assert.equal(get('twCards').innerHTML,'');assert.equal(get('noAlert').textContent,'لا توجد نتائج صالحة للساعة المطلوبة.');assert.equal(pending.length,2);
   }else if(mode==='manual_race'){
    a.now();get('rule').value='fcfs';get('berths').value='2';const old=a.runManual();const fresh=a.load(B);finish(pending[2],data(B));await fresh;finish(pending[1],{});await old;assert.equal(a.state().authoritative_timestamp,B);assert.equal(get('stageContent').innerHTML,'');
   }else if(mode==='replay'){
    context.TwinUI.open();assert.equal(context.drawn.waiting,1);assert.equal(context.drawn.berths_occupied,1);assert(get('twTitle').textContent.includes('بدون تدخل'));assert(!get('twTitle').textContent.includes('ونش'));assert.equal(pending.length,1);
   }
  }
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
"""


@pytest.mark.skipif(NODE is None, reason="Node is optional for Python-only installs")
@pytest.mark.parametrize("mode", ["kpi", "forecast", "actions", "manual", "stale_success",
                                 "stale_error", "failure", "manual_race", "replay"])
def test_root_api_binding(mode):
    result = subprocess.run([NODE, "-e", HARNESS, str(PAGE), mode], capture_output=True,
                            text=True, encoding="utf-8", timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_root_binding_uses_no_hybrid_or_snapshot_substitution():
    html = PAGE.read_text(encoding="utf-8")
    adapter = next(s for s in re.findall(r"<script>(.*?)</script>", html, re.S)
                   if "MARSA CURRENT API BINDINGS" in s)
    assert "/api/decision-support/analyze" in adapter
    assert all(value not in adapter for value in ["simulate-hybrid", "simulate-coupled",
                                                  "snapshots/", ".mean", "Monte-Carlo"])
    assert all(f'id="{symbol}"' in html for symbol in ["pmShip", "pmCrane", "pmTruck"])


def test_performance_cards_match_saved_xgboost_evaluation():
    import csv

    results = ROOT / "prediction_model" / "final_proxy" / "results"

    def selected(name):
        with (results / name).open(encoding="utf-8-sig", newline="") as handle:
            return next(row for row in csv.DictReader(handle)
                        if row["Model"] == "XGBoost" and row["Horizon"] == "+6h")

    hourly = selected("aggregate_results.csv")
    episode = selected("episode_early_warning_results.csv")
    html = PAGE.read_text(encoding="utf-8")
    assert f"v:'{float(hourly['Precision']) * 100:.1f}%'" in html
    assert f"{hourly['TP']} إصابة صحيحة من {int(hourly['TP']) + int(hourly['FP'])} تنبيهًا ساعيًا" in html
    assert f"v:'{int(float(episode['Median_Warning_Lead_Time_Hours']))}'" in html
    assert f"v:'{episode['Episodes_Detected_Before_Onset']} من {episode['Congestion_Episodes']}'" in html
    assert "للحالات المكتشفة قبل البداية فقط" in html
    assert "مؤشر Cargo اصطناعي؛ لا يثبت فعالية تشغيلية" in html
    assert "v:'63%'" not in html and "22 من 35" not in html


def test_weekly_chart_matches_saved_target_time_predictions():
    import csv
    from collections import defaultdict
    from datetime import datetime, timedelta

    weeks = defaultdict(lambda: [0, 0])
    path = ROOT / "prediction_model" / "final_proxy" / "results" / "fold_predictions.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if (row["model"], row["horizon_hours"], row["evaluation_role"]) != ("XGBoost", "6", "walk_forward"):
                continue
            day = datetime.fromisoformat(row["target_timestamp_6h"]).date()
            saturday = day + timedelta(days=(5 - day.weekday()) % 7)
            weeks[(saturday.month, saturday.day)][0] += int(row["predicted_label"])
            weeks[(saturday.month, saturday.day)][1] += int(float(row["target_6h"]))
    html = PAGE.read_text(encoding="utf-8")
    chart = re.search(r"const WK = \[(.*?)\];", html, re.S).group(1)
    rows = re.findall(r"d:'(\d+) (يونيو|يوليو|أغسطس|سبتمبر)',\s*p:(\d+),\s*a:(\d+)", chart)
    assert len(rows) == 14
    months = {"يونيو": 6, "يوليو": 7, "أغسطس": 8, "سبتمبر": 9}
    for day, month, predicted, reference in rows:
        assert weeks[(months[month], int(day))] == [int(predicted), int(reference)]
