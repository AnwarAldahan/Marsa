"""Focused checks for existing Main views consuming connected simulation results."""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def node(script):
    if NODE is None:
        pytest.skip("Node not installed")
    result = subprocess.run([NODE, "-e", script], capture_output=True, text=True,
                            encoding="utf-8", timeout=15, check=False)
    assert result.returncode == 0, result.stderr


def test_root_scalar_kpis_include_zero_and_reject_legacy_mean_objects():
    html = (ROOT / "Marsa updated.html").read_text(encoding="utf-8")
    definition = re.search(r"const kpi = .*?;", html).group()
    node("const assert=require('node:assert/strict');" + definition + """
      assert.equal(kpi({kpis:{cargo_unloaded:0}},'cargo_unloaded'),0);
      assert.equal(kpi({kpis:{cargo_unloaded:12}},'cargo_unloaded'),12);
      assert.equal(kpi({kpis:{cargo_unloaded:{mean:12}}},'cargo_unloaded'),null);
    """)


def test_forecast_shows_only_actual_six_hour_score_and_unavailable_state():
    html = (ROOT / "Marsa updated.html").read_text(encoding="utf-8")
    source = html.split("function renderTimeline(ml) {")[1].split("  function renderFactors")[0]
    node("""
      const assert=require('node:assert/strict');const nodes={};
      const $=id=>nodes[id]||=( {innerHTML:'',textContent:'',style:{},firstChild:{textContent:''}} );
      const document={querySelector:$},window={MARSA_LAST:{authoritative_timestamp:'2025-07-10T19:00:00Z'}};
      const escapeText=v=>String(v??''),num=(v,d=0)=>v===null?'—':Number(v).toFixed(d);
      function renderTimeline(ml){
    """ + source + """
      renderTimeline({status:'success',congestion_score:.959,target_timestamp_utc:'2025-07-11T01:00:00Z'});
      assert.equal(($('tlineSvg').innerHTML.match(/<circle/g)||[]).length,1);
      assert(!$('tlLabels').innerHTML.includes('+3'));
      assert(!$('tlLabels').innerHTML.includes('%'));
      assert($('tlLabels').innerHTML.includes('0.959'));
      renderTimeline({status:'success',congestion_score:0,target_timestamp_utc:'target'});
      assert($('tlLabels').innerHTML.includes('0.000'));
      renderTimeline({status:'ml_unavailable',congestion_score:.9});
      assert.equal($('tlineSvg').innerHTML,'');
    """)


@pytest.mark.parametrize("old_fails", [False, True])
def test_stale_analysis_cannot_overwrite_newer_results_or_errors(old_fails):
    html = (ROOT / "Marsa updated.html").read_text(encoding="utf-8")
    source = html.split("  async function load() {")[1].split("  $('mlLoad').addEventListener")[0]
    node("""
      const assert=require('node:assert/strict');let loadVersion=0,manualVersion=0,live=null,selected=null;
      const nodes={},$=id=>nodes[id]||=({value:'',innerHTML:'',textContent:'',disabled:false,hidden:false,style:{}});
      const document={querySelector:$,querySelectorAll:()=>[]},window={};const alertActions='buttons',API_BASE='http://test';
      const pending=[];const analyze=ts=>new Promise((resolve,reject)=>pending.push({ts,resolve,reject}));
      const stopReplay=()=>{},prepareCurrent=()=>{},render=o=>{live=o},ar=x=>x;
      let status;const st=s=>status=s;
      async function load(){
    """ + source + """
      async function check(){
        $('mlCustom').value='July';const old=load();$('mlCustom').value='November';const current=load();
        const value={authoritative_timestamp:'November',ml_forecast:{status:'success',classification:'LOW'},decision_support:{reasoning_mode:'deterministic'}};
        pending[1].resolve(value);await current;
    """ + ("pending[0].reject(Error('stale failure'));" if old_fails else "pending[0].resolve({...value,authoritative_timestamp:'July'});") + """
        await old;assert.equal(live,value);assert.equal(window.MARSA_LAST,value);assert(status.includes('November'));
      }check().catch(e=>{console.error(e);process.exitCode=1});
    """)


def test_performance_numbers_and_existing_weekly_chart_match_saved_evaluation():
    import csv
    from datetime import UTC, datetime, timedelta

    directory = ROOT / "prediction_model/final_proxy/results"
    with (directory / "aggregate_results.csv").open(encoding="utf-8") as f:
        evaluation = next(r for r in csv.DictReader(f) if r["Model"] == "XGBoost" and r["Horizon"] == "+6h")
    html = (ROOT / "Marsa updated.html").read_text(encoding="utf-8")
    assert f"{float(evaluation['Precision']) * 100:.1f}%" in html
    assert int(evaluation["TP"]) == 63 and int(evaluation["TP"]) + int(evaluation["FP"]) == 118
    assert "63 إصابة صحيحة من 118" in html
    with (directory / "fold_predictions.csv").open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["model"] == "XGBoost" and r["horizon_hours"] == "6"]
    displayed = [(int(p), int(a)) for p, a in re.findall(r"\{\s*d:'[^']+',\s*p:(\d+),\s*a:(\d+)\s*\}", html)]
    expected = []
    for i in range(14):
        end = datetime(2025, 6, 22, tzinfo=UTC) + timedelta(days=7 * i)
        week = [r for r in rows if end - timedelta(days=7) <= datetime.fromisoformat(r["target_timestamp_6h"]) < end]
        expected.append((sum(int(r["predicted_label"]) for r in week), sum(int(float(r["target_6h"])) for r in week)))
    assert displayed == expected


def test_existing_dashboard_does_not_release_blocked_initial_vessel_at_recorded_end():
    html = (ROOT / "dashboard/marsa.html").read_text(encoding="utf-8")
    source = html.split("function stateAt(sc, r, h) {")[1].split("  const methodAr")[0]
    node("const assert=require('node:assert/strict');function stateAt(sc,r,h){" + source + """
      const sc={vessels_at_berth:[{call_id:'A',mmsi:1,leaves_in_h:1}]};
      const r={units:'normalized_service_effort',vessels:[{call_id:'A',mmsi:1,berth_h:-1,arrival_h:-1,service_h:2,completion_h:null}]};
      assert.equal(stateAt(sc,r,2).berthed.length,1);
      assert.equal(stateAt(sc,r,2).berthed[0].leaves,null);
      r.vessels[0].completion_h=3;
      assert.equal(stateAt(sc,r,3).berthed.length,0);
    """)


@pytest.mark.parametrize("name", ["Marsa updated.html", "dashboard/marsa.html"])
def test_inline_javascript_syntax(name):
    if NODE is None:
        pytest.skip("Node not installed")
    for script in re.findall(r"<script>(.*?)</script>", (ROOT / name).read_text(encoding="utf-8"), re.S):
        result = subprocess.run([NODE, "--check"], input=script, capture_output=True,
                                text=True, encoding="utf-8", timeout=15, check=False)
        assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("name", ["Marsa updated.html", "dashboard/marsa.html"])
def test_main_design_and_static_components_are_preserved(name):
    original = subprocess.check_output(["git", "show", "origin/main:" + name], cwd=ROOT).decode("utf-8")
    current = (ROOT / name).read_text(encoding="utf-8")
    assert re.findall(r"<style[^>]*>(.*?)</style>", original, re.S) == re.findall(r"<style[^>]*>(.*?)</style>", current, re.S)
    original_dom = re.sub(r"<script>.*?</script>", "", original, flags=re.S)
    current_dom = re.sub(r"<script>.*?</script>", "", current, flags=re.S)
    assert re.findall(r'\bid="([^"]+)"', original_dom) == re.findall(r'\bid="([^"]+)"', current_dom)


def test_live_replay_speed_updates_immediately_and_playback_stops_at_horizon():
    html = (ROOT / "Marsa updated.html").read_text(encoding="utf-8")
    functions = html.split("  function stopReplay()")[1].split("  function drawLiveFrame()")[0]
    listeners = html.split("  $('twPlay').addEventListener('click',e=>")[1].split("  function prepareCurrent()")[0]
    node("""
      const assert=require('node:assert/strict');
      const handlers={},nodes={twPlay:{},twSpeed:{value:'1'}};
      const $=id=>({...nodes[id],addEventListener:(type,fn)=>handlers[id]=fn});
      let live={digital_twin:{simulation_horizon_hours:3}},replayHour=0,replayTimer=null,timer,delay;
      const setInterval=(fn,ms)=>{timer=fn;delay=ms;return 7},clearInterval=()=>{},drawLiveFrame=()=>{};
      function stopReplay()
    """ + functions + "  $('twPlay').addEventListener('click',e=>" + listeners + """
      const e={stopImmediatePropagation(){}};
      handlers.twPlay(e);assert.equal(delay,900);timer();assert.equal(replayHour,1);
      nodes.twSpeed.value='2';handlers.twSpeed(e);assert.equal(delay,450);assert.equal(replayHour,1);
      timer();timer();assert.equal(replayHour,3);assert.equal(replayTimer,null);
      handlers.twPlay(e);assert.equal(replayHour,0);assert.equal(replayTimer,7);
      handlers.twPlay(e);assert.equal(replayTimer,null);
    """)
