"""Unified Hybrid API bindings in the unchanged root frontend design."""
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
const scenario={parameters:{berth_slots:2,yard_capacity:10000,initial_inventory:0,gate_units_per_hour:100,unloading_units_per_hour:100},vessels:[]};
const row={candidate_id:'baseline',title:'بدون تدخل',rank:1,workload_signature:'same-work',units:'normalized_units',ranking_kpi:'queue_exposure_vessel_hours',scenario,kpis:{queue_exposure_vessel_hours:0,avg_completed_queue_wait_hours:2,completed_vessels:1,berth_utilization:.5},vessels:[{simulation_id:'B',arrival_hour:-1,berth_start_hour:-1,completion_hour:5,initial_state:'berthed'},{simulation_id:'W',arrival_hour:-1,berth_start_hour:2,completion_hour:4,initial_state:'waiting'},{simulation_id:'F',arrival_hour:4,berth_start_hour:null,completion_hour:null,initial_state:'future'}],timeline:Array.from({length:7},(_,hour)=>({hour,waiting:1,at_berth:1,completed:0,yard_occupancy_pct:0})),flows:Array(6).fill({unloaded:100,gate_cleared:100,yard_blocked_unloading:0})};
const data=ts=>({authoritative_timestamp:ts,context_id:'a'.repeat(32),context_signature:'frozen',decision_support:{highest_ranked_candidate:'baseline'},digital_twin:{mode:'unified_hybrid',units:'normalized_units',simulation_horizon_hours:6,scenario,results:[row]},strategy_candidates:[{candidate_id:'baseline',actions:[]},{candidate_id:'slot_plus_one',actions:[{type:'add_slots',value:1}]}]});
async function main(){
 if(mode==='kpi'){
  assert.equal(api.kpi(row,'queue_exposure_vessel_hours'),0);assert.equal(api.kpi(row,'missing'),null);assert.equal(api.kpi({kpis:{avg_wait_hours:{mean:9}}},'avg_wait_hours'),null);
  const d=data('2025-07-10T19:00:00Z');d.digital_twin.results.unshift({candidate_id:'user_decision',rank:1});assert.equal(api.winner(d),'baseline');assert.equal(api.esc('<img>'),'&lt;img&gt;');
 }else if(mode==='forecast'){
  assert.equal(api.forecast({status:'success',congestion_score:0}).score,0);assert.equal(api.forecast({status:'ml_unavailable',congestion_score:.9}).score,null);
  assert.equal(Object.keys(api.forecast({})).join(','),'score,target');
 }else if(mode==='actions'){
  const d=data('2025-07-10T19:00:00Z');assert.equal(api.actionRequest(d,'shortest',3).user_actions.length,2);assert.equal(api.actionRequest(d,'fcfs',1).user_actions[0].value,1);
  for(const rule of ['own','random','plan:cranes'])assert.throws(()=>api.actionRequest(d,rule,2));assert.throws(()=>api.actionRequest(d,'plan:slot_plus_one',3));assert.throws(()=>api.actionRequest(d,'fcfs',1.5));
 }else if(mode==='contract'){
  const d=data('2025-07-10T19:00:00Z');assert.equal(api.validateContract(d),d);
  d.digital_twin.mode='legacy';assert.throws(()=>api.validateContract(d));d.digital_twin.mode='unified_hybrid';
  d.digital_twin.results.push({...row,candidate_id:'bad',workload_signature:'different'});assert.throws(()=>api.validateContract(d));
 }else if(mode==='manual'){
  const r=api.manualResult(data('2025-07-10T19:00:00Z'),row);assert.equal(r.avgWait,2);assert.equal(r.utilization,50);assert.equal(r.berthIntervals[0][0].end,5);assert.equal(r.berthIntervals[1][0].start,2);assert.equal(r.vessels.find(v=>v.id==='F').assignTime,null);
 }else{
  const nodes={},pending=[];
  const get=id=>nodes[id] ||= {value:'',innerHTML:'',textContent:'',hidden:true,disabled:false,style:{},firstChild:{textContent:''},classList:{add(){},remove(){},toggle(){}},querySelector(){return get(id+'child')},querySelectorAll(){return []},addEventListener(){},setAttribute(){}};
  let script=source.replace("  load($('mlPreset').value);",`flow=timeline=factors=alert=cards=()=>{};globalThis.audit={load,runManual,state:()=>DATA,now:()=>{manualMode='now'},select:(id,h)=>{selected=id;hour=h;frame()}};`);
  const context={document:{getElementById:get,querySelector:get,querySelectorAll:()=>[],addEventListener(){}},fetch:(url,options)=>new Promise((resolve,reject)=>pending.push({url,options,resolve,reject})),setInterval(){},clearInterval(){},console,MARSA_NORMALIZE:s=>s,MARSA_DRAWMAP:s=>{context.drawn=s}};context.window=context;
  if(mode==='hosted'||mode==='override')context.location={protocol:'https:',origin:'https://marsa.example'};
  if(mode==='override')context.MARSA_API_BASE='https://backend.example/';
  if(mode==='file')context.location={protocol:'file:',origin:'null'};
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
   }else if(['hosted','override','file'].includes(mode)){
    const base=mode==='override'?'https://backend.example':mode==='hosted'?'https://marsa.example':'http://127.0.0.1:8000';
    assert.equal(pending[0].url,base+'/api/decision-support/unified');
   }else if(mode==='replay'){
    context.TwinUI.open();assert.equal(context.drawn.waiting,1);assert.equal(context.drawn.berths_occupied,1);assert(get('twTitle').textContent.includes('بدون تدخل'));assert(!get('twTitle').textContent.includes('ونش'));assert.equal(pending.length,1);
   }
  }
 }
}
main().catch(e=>{console.error(e);process.exitCode=1});
"""


@pytest.mark.skipif(NODE is None, reason="Node is optional for Python-only installs")
@pytest.mark.parametrize("mode", ["kpi", "forecast", "actions", "contract", "manual", "stale_success",
                                 "stale_error", "failure", "manual_race", "replay", "hosted", "override", "file"])
def test_root_api_binding(mode):
    result = subprocess.run([NODE, "-e", HARNESS, str(PAGE), mode], capture_output=True,
                            text=True, encoding="utf-8", timeout=20, check=False)
    assert result.returncode == 0, result.stderr


def test_root_binding_uses_no_hybrid_or_snapshot_substitution():
    html = PAGE.read_text(encoding="utf-8")
    adapter = next(s for s in re.findall(r"<script>(.*?)</script>", html, re.S)
                   if "MARSA CURRENT API BINDINGS" in s)
    assert "/api/decision-support/unified" in adapter
    assert "/api/decision-support/analyze" not in adapter
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
