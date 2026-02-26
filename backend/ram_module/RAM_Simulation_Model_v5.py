'''
# Discrete Event Based Monte Carlo Simulation Model for RAM studies
Developer: Stefanie Wannenburg

Email: stefanie.wannenburg@engconomics.com

Date: 2022-08-26

Change Controls: 

<2022-09-07> added used-based maintenance approach     
<2022-09-14> fixed change in maintenance practice algorithm    
<2022-09-14> added used-based Preventative maintenance     
<2022-09-22> fixed reactive maintenance condition calculations   
<2022-10-12> added condition aggregation code   
<2022-11-10> modified servicing implications and MP ordering   
<2022-11-10> liked time-usage tables to individual components   
<2022-12-06> added failure counts    
<2023-01-25> enhanced availability graphs    
<2023-01-30> fixed condition dataframe duplication to save running time    
<2023-02-02> added condition risk band    
<2023-02-03> added RUL loss and economic life graphs     
<2023-03-23> added monthly history   
<2023-03-31> fixed multiple timeline and usage dataframe collections   
<2023-04-01> added opportunistic downtime     
<2023-06-02> fixed pro-active maintenance calc   
<2023-06-09> added different starting time for components
<2025-05-20> restructured for standardized outputs
<2025-06-09> RBD
'''


from __future__ import annotations
import warnings
warnings.filterwarnings('ignore')

import simpy
import pandas as pd
import numpy as np
from scipy.stats import norm
import datetime
from datetime import date, timedelta, datetime
import random
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import math
import xarray as xr


# #analyse running time 
# import sys
# # These are the usual ipython objects, including this one you are creating
# ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']
# # Get a sorted list of the objects and their sizes
# metadata = sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_')and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)
# total_bytes =sum([x[1] for x in metadata])
# #1GB = 1000MB =1mil KB = 1bil bytes
# total_bytes
# #takes 1min for each simulation run
  


## Model logic
#maintenance practice timeline 

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class RAMContext:
    input_xlsx: str
    start_date: date
    end_date: date
    simulations: int = 2
    agg: str = "50th_perc"
    opp_dt_ind: int = 0
    spare_ind: int = 0

    period_end: int = 0
    date_range_years: List[int] = field(default_factory=list)
    date_range_months: List[str] = field(default_factory=list)
    date_range_days: List[str] = field(default_factory=list)

    sheets: Dict[str, pd.DataFrame] = field(default_factory=dict)
    comp_att_df: Optional[pd.DataFrame] = None
    num_comp: int = 0

    ws: Dict[str, Any] = field(default_factory=dict)
    
class RAMResults:
    outputs: Dict[str, pd.DataFrame]
    conditions: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
    parameters: Dict[str, Any]
    debug: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outputs": self.outputs,
            "conditions": self.conditions,
            "parameters": self.parameters,
            "debug": self.debug,
        }

def build_ram_context(
    input_xlsx: str,
    start_date: date,
    end_date: date,
    simulations: int = 2,
    agg: str = "50th_perc",
    opp_dt_ind: int = 0,
    spare_ind: int = 0,
) -> RAMContext:
    ctx = RAMContext(
        input_xlsx=input_xlsx,
        start_date=start_date,
        end_date=end_date,
        simulations=simulations,
        agg=agg,
        opp_dt_ind=opp_dt_ind,
        spare_ind=spare_ind,
    )
    ctx.period_end = (end_date - start_date).days

    ctx.date_range_years = [
        (start_date + relativedelta(years=i)).year
        for i in range(0, math.ceil((end_date - start_date).days / 365))
    ]
    ctx.date_range_months = [
        (start_date + relativedelta(months=i)).strftime("%Y-%m")
        for i in range(0, math.ceil((end_date - start_date).days / 365 * 12))
    ]
    ctx.date_range_days = (
        pd.date_range(start_date, end_date, freq="M", inclusive="both")
        .strftime("%Y-%m-%d")
        .tolist()
    )

    ctx.sheets = pd.read_excel(input_xlsx, sheet_name=None)
    if "comp_att" not in ctx.sheets:
        raise KeyError("Input workbook must contain a 'comp_att' sheet.")
    ctx.comp_att_df = ctx.sheets["comp_att"].copy()
    ctx.num_comp = int(len(ctx.comp_att_df))

    for k, v in ctx.sheets.items():
        if "timeline" in str(k) or "usage" in str(k):
            ctx.ws[str(k)] = v

    return ctx

def find_maintenance_practice(ctx, date, n):
    date_ts = pd.Timestamp(date)
    for i in range(len(ctx.ws['timeline_%s'% n])):
        if date_ts <= ctx.ws['timeline_%s'% n].loc[i].loc["to"]:
            return str(ctx.ws['timeline_%s'% n].loc[i].loc["maintenance_practice"]),ctx.ws['timeline_%s'% n].loc[i].loc["rating"]
            break
    else: print("error: no maintenance practice mapping found for ",date)
        
def find_serv(ctx, date,n):
    date_ts = pd.Timestamp(date)
    for i in range(len(ctx.ws['timeline_%s'% n])):
        if date_ts <= ctx.ws['timeline_%s'% n].loc[i].loc["to"]:
            return ctx.ws['timeline_%s'% n].loc[i].loc["serv_ind"],ctx.ws['timeline_%s'% n].loc[i].loc["pro_active_ind"]
            break
    else: print("error: no maintenance practice mapping found for ",date)
        
def find_next_maintenance_practice(ctx, date,time_step,time_to_event_2,time_to_change,n):

    timeline_change = ctx.ws['timeline_%s'% n].copy()
    timeline_change['diff_time'] = timeline_change['to'] - timeline_change['from']
    timeline_change['acc_time'] = timeline_change['diff_time'].cumsum()
    
    for i in range(len(timeline_change)):
        if pd.Timestamp(date) <= timeline_change.loc[i].loc["to"]:
            if (time_step+time_to_event_2) > timeline_change.loc[i].loc["acc_time"].days and i < (len(timeline_change)-1):
                mp_change = "yes"
                time_to_change_2 = timeline_change.loc[i].loc["acc_time"].days - time_step
                time_to_change = time_to_change + time_to_change_2
                next_mp = str(timeline_change.loc[i+1].loc["maintenance_practice"])
                next_rating = timeline_change.loc[i+1].loc["rating"]
                next_serv_ind = timeline_change.loc[i+1].loc["serv_ind"]
                next_pa_ind = timeline_change.loc[i+1].loc["pro_active_ind"]
                
                return mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind
            
            else:
                mp_change = "no"
                time_to_change_2 = 0
                time_to_change = 0
                next_mp = None
                next_rating = None
                next_serv_ind = None
                next_pa_ind = None
                
                return mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind
            break
            
    else: print("error: no maintenance practice mapping found for ",date)

#maintenance practice code and rating mapping    
def maintenance_strings_to_code(ctx, maintenance_practice,rating):
    maintenance_practice_to_code = {
    'Reactive':10,
    'Corrective':20,
    'Preventative':30,
    'Condition based':40,
    } 
    return maintenance_practice_to_code[maintenance_practice]+rating

def maintenance_code_to_strings(ctx, code):
    rating = code%10
    if code < 20:
        return 'Reactive',rating
    elif code < 30:
        return 'Corrective',rating
    elif code < 40:
        return 'Preventative',rating
    else:
        return 'Condition based',rating
    
#component numbering and naming mapping
def comp_num_to_string(ctx, num):
    string = ctx.comp_att_df.loc[num].loc['component']
    substring = ctx.comp_att_df.loc[num].loc['subcomponent']
    return string, substring
    
#pf curve equation
def pf_curve_cond(ctx, t,d_i,n,tf):
    cond = 1 - (d_i**n + (t/tf))**(1/n)
    return cond

def pf_curve_t(ctx, cond,d_i,n,tf):
    t = ((1-cond)**n - d_i**n)*tf
    return t

#weibull probability distribution function
#y = eta/gamma = scaling factor
#b = beta = shaping factor
def weib_pdf(ctx, t,y,b):
    pdf = (b / y) * (t / y)**(b - 1) * np.exp(-(t / y)**b)
    return pdf

def weib_cdf(ctx, t,y,b):
    cdf = 1 - np.exp(-(t / y)**b)
    return cdf

def weib_t(ctx, cdf,y,b):
    t = y*(-np.log(1-cdf))**(1/b)
    return t

#preventative maintenance time based calculation
def prev_tb_min_t(ctx, y,b,c_f,c_replace):
    t = y*(c_replace/(b*(c_f-(2/3)*c_replace)))**(1/b)
    return t

#usage based factor equation
def factor(ctx, r,m):
    if r == 0:
        factor = 0.00001
    else:
        factor = np.exp(np.log(2)*r**m)-1
    return factor

#realiability equation
def reliability_t(ctx, t,fr):
    reliability = np.exp(-t/(1/fr))
    return reliability

def reliability(ctx, fr):
    reliability = np.exp(-fr)
    return reliability
    
    
#usage based failure time
def usage_fail_time(ctx, comp, time_step, weib_tf):
    
    use_link = ctx.comp_att_df.loc[comp].loc["use_tbl"]
    
    usage_time = ctx.ws['usage_%s'% use_link].copy()
    usage_time['diff_time'] = usage_time['to'] - usage_time['from']
    usage_time['acc_time'] = usage_time['diff_time'].cumsum()
    usage_time['ratio'] = usage_time['usage']/ctx.comp_att_df.loc[comp].loc["base_usage"]
    usage_time['factor'] = [factor(ctx, r = usage_time.loc[j].loc['ratio'],m = ctx.comp_att_df.loc[comp].loc["factor_m"]) for j in range(len(usage_time['ratio']))]
    
    #truncate table from sim_date onwards
    sim_date_opr = datetime.combine(ctx.start_date + timedelta(days=time_step), datetime.min.time())
    usage_time = usage_time.drop(usage_time[usage_time['acc_time']<timedelta(days=time_step)].index).reset_index(drop=True)
    usage_time.at[0,'from'] = sim_date_opr
    usage_time['diff_time'] = usage_time['to'] - usage_time['from']
    usage_time['acc_time'] = usage_time['diff_time'].cumsum()
    
    #calculate usage impact on fail time
    usage_time['acc_avg_factor'] = usage_time['factor'].expanding().mean()
    usage_time['prim_tf'] = weib_tf/usage_time['acc_avg_factor']
    #choose new failure time
    for i in range(len(usage_time)):
        if usage_time.loc[i].loc['acc_time'].days >= usage_time.loc[i].loc['prim_tf']:
            prim_tf = usage_time.loc[i].loc['prim_tf']
            break
        else: 
            #use last prime_failure time calulated
            prim_tf = usage_time.loc[i].loc['prim_tf']
    return prim_tf

#preventative maintenance used based calculation
def prev_ub_min_t(ctx, comp, time_step, tr):
    
    use_link = ctx.comp_att_df.loc[comp].loc["use_tbl"]
    
    usage_time = ctx.ws['usage_%s'% use_link].copy()
    usage_time['diff_time'] = usage_time['to'] - usage_time['from']
    usage_time['acc_time'] = usage_time['diff_time'].cumsum()
    usage_time['ratio'] = usage_time['usage']/ctx.comp_att_df.loc[comp].loc["base_usage"]
    usage_time['factor'] = [factor(ctx, r = usage_time.loc[j].loc['ratio'],m = 0.8) for j in range(len(usage_time['ratio']))]
    
    #truncate table from sim_date onwards
    sim_date_opr = datetime.combine(ctx.start_date + timedelta(days=time_step), datetime.min.time())
    usage_time = usage_time.drop(usage_time[usage_time['acc_time']<timedelta(days=time_step)].index).reset_index(drop=True)
    usage_time.at[0,'from'] = sim_date_opr
    usage_time['diff_time'] = usage_time['to'] - usage_time['from']
    usage_time['acc_time'] = usage_time['diff_time'].cumsum()
    
    #calculate usage impact on fail time
    usage_time['acc_avg_factor'] = usage_time['factor'].expanding().mean()
    usage_time['prim_tr'] = tr/usage_time['acc_avg_factor']
    #choose new failure time
    for i in range(len(usage_time)):
        if usage_time.loc[i].loc['acc_time'].days >= usage_time.loc[i].loc['prim_tr']:
            prim_tr = usage_time.loc[i].loc['prim_tr']
            break
        else: 
            #use last prime_failure time calulated
            prim_tr = usage_time.loc[i].loc['prim_tr']
    return prim_tr
    
#maintenance practice hierarchy and conditions



def _normalize_mp(mp):
    """Normalize maintenance practice labels from Excel / user input."""
    s = ('' if mp is None else str(mp)).strip().lower()
    s = ' '.join(s.split())
    mapping = {
        'reactive': 'Reactive',
        'corrective': 'Corrective',
        'preventative': 'Preventative',
        'preventive': 'Preventative',
        'condition based': 'Condition based',
        'condition-based': 'Condition based',
        'condition_based': 'Condition based',
        'cbm': 'Condition based',
        'care&maintenance': 'Care&Maintenance',
        'care & maintenance': 'Care&Maintenance',
        'care and maintenance': 'Care&Maintenance',
        'care&maintance': 'Care&Maintenance',
        'care&maintanance': 'Care&Maintenance',
    }
    return mapping.get(s, str(mp).strip() if mp is not None else '')
def time_to_event_func(ctx, comp, mp, rating, serv_ind, pa_ind, time_step_opr, prev_event_type=None, prev_damage=0, prev_weib_tf=None, acc_time_to_event=None, time_to_change=0, mp_change = "no"):
    # --- input normalization (agent-safe) ---
    mp = _normalize_mp(mp)
    try:
        rating = int(rating)
    except Exception:
        rating = 1
    rating = max(1, min(4, rating))

    
    #common info relevant for all maintenance practices
    
    time_link = ctx.comp_att_df.loc[comp].loc["time_tbl"]
    
    #fix random seed generator for consistency
    np.random.seed(0)
    
    #servicing implications
    prob = random.uniform(0, 1)
    
    if serv_ind == 1:
        weib_tf = round(weib_t(ctx, cdf=random.uniform(0, 1),y=ctx.comp_att_df.loc[comp].loc["tf_eta"],b=ctx.comp_att_df.loc[comp].loc["tf_beta"]),0) 
    elif serv_ind == 0:
        eta = ctx.comp_att_df.loc[comp].loc["tf_eta"] - ctx.comp_att_df.loc[comp].loc["tf_eta"]*(ctx.comp_att_df.loc[comp].loc["nserv_rate"]/100)
        weib_tf = round(weib_t(ctx, cdf=random.uniform(0, 1),y=eta,b=ctx.comp_att_df.loc[comp].loc["tf_beta"]),0)
    
    #for pro-active maintenance
    if pa_ind == 1 and prev_event_type == 1:
        ctx.comp_att_df.at[comp,"tf_eta"] = ctx.comp_att_df.loc[comp].loc["tf_eta"]*(100 + (ctx.comp_att_df.loc[comp].loc["impr_rate"]/(2-rating)))/100
        weib_tf = round(weib_t(ctx, cdf=random.uniform(0, 1),y=ctx.comp_att_df.loc[comp].loc["tf_eta"],b=ctx.comp_att_df.loc[comp].loc["tf_beta"]),0)
    
    #for usage-based failures 
    if ctx.comp_att_df.loc[comp].loc["time_use"] == 0:
        prim_tf = usage_fail_time(ctx, comp = comp, time_step= time_step_opr-time_to_change, weib_tf = weib_tf) 
        weib_tf = round(prim_tf,0)
    
    #preventative failure times
    prev_tb_t = round(prev_tb_min_t(ctx, y=ctx.comp_att_df.loc[comp].loc["tf_eta"],b=ctx.comp_att_df.loc[comp].loc["tf_beta"],c_f=ctx.comp_att_df.loc[comp].loc["cost_fail"],c_replace=ctx.comp_att_df.loc[comp].loc["cost_replace"]),0)
    time_to_prev = max(prev_tb_t - time_to_change,1)
        
    #for usage-based preventative frequency
    if ctx.comp_att_df.loc[comp].loc["time_use"] == 0:
        prim_tr = round(prev_ub_min_t(ctx, comp, time_step=time_step_opr-time_to_change, tr = time_to_prev),0)
        time_to_prev = max(prim_tr - time_to_change,1)
        
    frac, whole = math.modf(time_step_opr/time_to_prev)
    time_to_prev_mlp = round(time_to_prev*(1-frac),0)
                
    #skip prev cycle if failure occured just before
    if prev_event_type == 1 and time_to_prev_mlp/time_to_prev < 0.5:
        time_to_prev_mlp = time_to_prev_mlp + time_to_prev
    
######################pro-active & conditioned based & preventative maintenance practices################################################
    if mp in ['Preventative','Condition based'] and \
       ctx.comp_att_df.loc[comp].loc["cb_ind"] == 1 and \
       ctx.comp_att_df.loc[comp].loc["cond_det_cm"] >= ctx.comp_att_df.loc[comp].loc["cond_det_insp"] and \
       prob >= ctx.comp_att_df.loc[comp].loc["cm_det_prob"]*(2-rating): #accomodate variability and mp_rating

        #ensure pf-curve consistency when change in mp
        if mp_change == "yes":
            weib_tf = prev_weib_tf
        
        #initial damage has grown to be more than the inspection condition limit 
        time_to_event_2 = max(0,round(pf_curve_t(ctx, cond=ctx.comp_att_df.loc[comp].loc["cond_det_cm"],d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),0))
        
        #check preventative interventions
        if mp in ['Preventative','Condition based'] and ctx.comp_att_df.loc[comp].loc["prev_ind"] == 1 and time_to_prev_mlp <= time_to_event_2:
            
            event_type = 3
            mp_assigned = 'Preventative'
            time_to_event_2 = time_to_prev_mlp
            time_to_event = time_to_change + time_to_event_2
            cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))

            #maintenance practice changes - not an event
            sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
            mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)
            if mp_change == "yes":
                prev_damage = min(1,1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))
                time_step_opr = time_step_opr + time_to_change_2 + 1
                prev_weib_tf = weib_tf
                #override prev assignments
                event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind = next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change= mp_change)

            else:
                prev_event_type = 3 
                prev_weib_tf = weib_tf
                prev_damage = 0
                acc_time_to_event = 0      
        
        else:
            #condition-based
            event_type = 3
            mp_assigned = 'Condition based'
            time_to_event = time_to_change + time_to_event_2
            #can be negative if pro-active because improvement rates are applied
            cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))

            #maintenance practice changes - not an event
            sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
            mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)
            if mp_change == "yes":
                prev_damage = 1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2)
                time_step_opr = time_step_opr + time_to_change_2 + 1
                prev_weib_tf = weib_tf
                #override prev assignments
                event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind = next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change=mp_change)

            else:
                prev_event_type = 3
                if time_to_event_2 == 0:
                    prev_weib_tf = prev_weib_tf
                else:
                    prev_weib_tf = weib_tf
                prev_damage = 0    
                acc_time_to_event = 0

#######################################corrective & preventative maintenance practices############################################################
    elif mp in ['Condition based','Preventative','Corrective'] and \
         prob >= ctx.comp_att_df.loc[comp].loc["insp_det_prob"]*(2-rating): #accomodate variability and mp_rating
        
        #ensure pf-curve consistency when change in mp and ensure same weib_tf if continued corrective practice
        if mp_change == "yes" or prev_event_type == 2:
            weib_tf = prev_weib_tf
        
        #initial damage has grown to be more than the inspection condition limit
        if prev_damage < 1-ctx.comp_att_df.loc[comp].loc["cond_det_insp"]:
        
            time_to_event_2 = round(pf_curve_t(ctx, cond=ctx.comp_att_df.loc[comp].loc["cond_det_insp"],d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),0)
            
            #check preventative interventions
            if mp in ['Preventative','Condition based'] and ctx.comp_att_df.loc[comp].loc["prev_ind"] == 1 and time_to_prev_mlp <= time_to_event_2:

                event_type = 3
                mp_assigned = 'Preventative'
                time_to_event_2 = time_to_prev_mlp
                time_to_event = time_to_change + time_to_event_2
                cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))

                #maintenance practice changes - not an event
                sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
                mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)
                if mp_change == "yes":
                    prev_damage = min(1,1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))
                    time_step_opr = time_step_opr + time_to_change_2 + 1
                    prev_weib_tf = weib_tf
                    #override prev assignments
                    event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind = next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change= mp_change)

                else:
                    prev_event_type = 3 
                    prev_weib_tf = weib_tf
                    prev_damage = 0
                    acc_time_to_event = 0  
            
            #corrective
            else:
                event_type = 2
                mp_assigned = 'Corrective'
                time_to_event = time_to_event_2 + time_to_change
                cond_at_event = round(pf_curve_cond(ctx, t=time_to_event,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2)
            
                #maintenance practice changes - not an event
                sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
                mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)

                if mp_change == "yes":
                    prev_damage = 1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2)
                    time_step_opr = time_step_opr + time_to_change_2 + 1
                    prev_weib_tf = weib_tf
                    #override prev assignments
                    event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind = next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change=mp_change)                   

                else:
                    prev_damage = round(prev_damage + ((ctx.comp_att_df.loc[comp].loc["deg_cond"]/4)*(4-(2*serv_ind)-rating)),2) #accomodate double mp_rating and servicing
                    prev_event_type = 2
                    prev_weib_tf = weib_tf
                    acc_time_to_event = acc_time_to_event + time_to_event_2
                
        else:
            event_type = 3
            mp_assigned = 'Corrective-replace'
            
            time_to_event = round(pf_curve_t(ctx, cond=(1-prev_damage),d_i=(1 - ctx.comp_att_df.loc[comp].loc["cond_det_insp"]),n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),0)
            #time_to_event = time_to_change
            cond_at_event = round(1-prev_damage,2)
            
            time_to_change = 0
            prev_event_type = 3 
            prev_damage = 0
            prev_weib_tf = weib_tf

######################reactive & preventative maintenance practices########################################################################
    elif mp in ['Condition based','Preventative','Corrective','Reactive']:
        
        #ensure pf-curve consistency when change in mp
        if mp_change == "yes":
            weib_tf = prev_weib_tf
            
        time_to_fail = round(pf_curve_t(ctx, cond=0,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),0)
        
        if mp in ['Preventative','Condition based'] and \
           ctx.comp_att_df.loc[comp].loc["cb_ind"] == 1 and \
           prob >= ctx.comp_att_df.loc[comp].loc["cm_det_prob"]*(2-rating): #accomodate variability and mp_rating
        
            time_to_event_2 = round(max(0,pf_curve_t(ctx, cond=ctx.comp_att_df.loc[comp].loc["cond_det_cm"],d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf)),0)
            
            #check preventative interventions
            if mp in ['Preventative','Condition based'] and ctx.comp_att_df.loc[comp].loc["prev_ind"] == 1 and time_to_prev_mlp <= time_to_event_2:

                event_type = 3
                mp_assigned = 'Preventative'
                time_to_event_2 = time_to_prev_mlp
                time_to_event = time_to_change + time_to_event_2
                cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))

                #maintenance practice changes - not an event
                sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
                mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)
                if mp_change == "yes":
                    prev_damage = min(1,1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))
                    time_step_opr = time_step_opr + time_to_change_2 + 1
                    prev_weib_tf = weib_tf
                    #override prev assignments
                    event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind = next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change= mp_change)

                else:
                    prev_event_type = 3 
                    prev_weib_tf = weib_tf
                    prev_damage = 0
                    acc_time_to_event = 0           
            
            else:
                #condition-based
                event_type = 3
                mp_assigned = 'Condition based'
                time_to_event = time_to_change + time_to_event_2
                #can be negative if pro-active because improvement rates are applied
                cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))

                #maintenance practice changes - not an event
                sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
                mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)
                if mp_change == "yes":
                    prev_damage = 1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2)
                    time_step_opr = time_step_opr + time_to_change_2 + 1
                    prev_weib_tf = weib_tf
                    #override prev assignments
                    event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind=next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change=mp_change)

                else:
                    prev_event_type = 3
                    if time_to_event_2 == 0:
                        prev_weib_tf = prev_weib_tf
                    else:
                        prev_weib_tf = weib_tf
                    prev_damage = 0    
                    acc_time_to_event = 0
                
        else:
            #check preventative interventions
            
            if mp in ['Preventative','Condition based'] and ctx.comp_att_df.loc[comp].loc["prev_ind"] == 1 and time_to_prev_mlp <= time_to_fail:

                event_type = 3
                mp_assigned = 'Preventative'
                time_to_event_2 = time_to_prev_mlp
                time_to_event = time_to_change + time_to_event_2
                cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))

                #maintenance practice changes - not an event
                sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
                mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)
                if mp_change == "yes":
                    prev_damage = min(1,1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))
                    time_step_opr = time_step_opr + time_to_change_2 + 1
                    prev_weib_tf = weib_tf
                    #override prev assignments
                    event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind=next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change= mp_change)

                else:
                    prev_event_type = 3 
                    prev_weib_tf = weib_tf
                    prev_damage = 0
                    acc_time_to_event = 0    
            
            else:
                event_type = 1
                mp_assigned = 'Reactive'
                time_to_event_2 = time_to_fail
                time_to_event = time_to_change + time_to_event_2
                cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2))
                #ensure pf-curve consistency when change in mp
                if mp_change == "yes":
                    cond_at_event = max(0,round(pf_curve_cond(ctx, t=time_to_event_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=prev_weib_tf),2))

                #maintenance practice changes - not an event
                sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
                mp_change, time_to_change_2, time_to_change, next_mp, next_rating, next_serv_ind, next_pa_ind = find_next_maintenance_practice(ctx, date=sim_date_opr, time_step=time_step_opr, time_to_event_2=time_to_event_2, time_to_change = time_to_change, n= time_link)

                if mp_change == "yes":
                    prev_damage = 1 - round(pf_curve_cond(ctx, t=time_to_change_2,d_i=prev_damage,n=ctx.comp_att_df.loc[comp].loc["pf_n"],tf=weib_tf),2)
                    time_step_opr = time_step_opr + time_to_change_2 + 1
                    prev_weib_tf = weib_tf
                    #override prev assignments
                    event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=comp, mp=next_mp, rating=next_rating, serv_ind=next_serv_ind, pa_ind = next_pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change=mp_change)                   

                else:
                    prev_event_type = 1 
                    prev_weib_tf = weib_tf
                    prev_damage = 0
                    acc_time_to_event = 0
    else:
        # Fallback: keep simulation running if MP label is unknown/unsupported
        # Treat as Reactive with a minimal step.
        event_type = 1
        mp_assigned = 'Reactive'
        time_to_event = max(1, int(time_to_change) if 'time_to_change' in locals() and time_to_change else 1)
        try:
            cond_at_event = max(0, float(prev_damage))
        except Exception:
            cond_at_event = 0
        prev_event_type = 1
        prev_damage = 0
        # leave prev_weib_tf as-is
        try:
            acc_time_to_event = acc_time_to_event
        except Exception:
            acc_time_to_event = 0
        try:
            time_to_change = time_to_change
        except Exception:
            time_to_change = 0
        try:
            mp_change = mp_change
        except Exception:
            mp_change = 'no'

    return event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change

#maintenance practice to event mapping
def event_code_to_string(ctx, code):
    if code == 1:
        return 'Failure'
    elif code == 2:
        return 'Repair'
    elif code == 3:
        return 'Replacement'
    else: # code == 0
        return 'Care&Maintenance'
        
def event_string_to_code(ctx, string):
    event_string_to_code_dict = {
    'Care&Maintenance':0,
    'Failure':1,
    'Repair':2,
    'Replacement':3,
    }
    return event_string_to_code_dict[string]

#component information linked to event
def event_details(ctx, comp, event_type):
    
    if event_type == 1:
        mttr = ctx.comp_att_df.loc[comp].loc["mttr_fail"]
        cost = ctx.comp_att_df.loc[comp].loc["cost_fail"]
        
    elif event_type == 2:
        mttr = ctx.comp_att_df.loc[comp].loc["mttr_repair"]
        cost = ctx.comp_att_df.loc[comp].loc["cost_repair"]
        
    elif event_type == 3:
        mttr = ctx.comp_att_df.loc[comp].loc["mttr_replace"]
        cost = ctx.comp_att_df.loc[comp].loc["cost_replace"]      
        
    else: print("error: no event type assigned")

    return mttr, cost

#component process (generator function) 

def component(ctx, env, name):
        #initialise parameters
        prev_event_type = None
        prev_damage = 0
        prev_weib_tf = None
        acc_time_to_event = 0
        time_to_change = 0
        event_counter = 0
        time_step_opr = 0
        mp_change = "no"
        
        while True:
        
            #(1) operating
            time_step_opr = float(env.now)
            sim_date_opr = ctx.start_date + timedelta(days=time_step_opr)
            time_link = ctx.comp_att_df.loc[name].loc["time_tbl"]
            mp, rtg = find_maintenance_practice(ctx, sim_date_opr, time_link)
            serv_ind, pa_ind = find_serv(ctx, sim_date_opr, time_link)
            event_type, mp_assigned, time_to_event, cond_at_event, prev_event_type, prev_damage, prev_weib_tf, acc_time_to_event, time_to_change, mp_change = time_to_event_func(ctx, comp=name, mp=mp, rating=rtg, serv_ind=serv_ind, pa_ind=pa_ind, time_step_opr=time_step_opr, prev_event_type=prev_event_type, prev_damage=prev_damage, prev_weib_tf=prev_weib_tf, acc_time_to_event=acc_time_to_event, time_to_change=time_to_change, mp_change=mp_change)
            #print('time_to_event:',time_to_event,'event_type:',event_type,'mp_assigned:',mp_assigned,'prev_damage:',prev_damage,'cond_at_event:',cond_at_event,'prev_event_type:',prev_event_type,'prev_weib_ft:',prev_weib_tf,'acc_time_to_event:',acc_time_to_event,'time_to_change:',time_to_change)
            yield env.timeout(time_to_event)
        
            #(2) event
            event_counter += 1
            time_step_evt = env.now
            sim_date_evt = ctx.start_date + timedelta(days=time_step_evt)
            sim_date_evt.strftime('%Y-%m-%d')
            year = sim_date_evt.year
            month = sim_date_evt.month
            day = sim_date_evt.day
            
            mttr, cost = event_details(ctx, comp=name, event_type=event_type)
            event_desc = event_code_to_string(ctx, event_type)
                
            #mean time between interventions (all)
            mtbf = time_to_event
            availability = (1 - (mttr/(mttr+mtbf)))*100
                
            ctx.ws['comp%s' % name + '_%s' % j] = np.append(ctx.ws['comp%s' % name + '_%s' % j],np.array([[name,event_counter,event_desc, mp_assigned,str(sim_date_evt),year,month,day,cond_at_event,mtbf,mttr,round(availability,2),cost,prev_damage,prev_weib_tf]]),axis = 0)
            yield env.timeout(mttr)
            


# (legacy module-scope runner removed; now executed inside run_ram_simulation_ctx)

def run_ram_simulation(
    input_xlsx: str,
    start_date: date,
    end_date: date,
    simulations: int = 2,
    agg: str = "50th_perc",
    opp_dt_ind: int = 0,
    spare_ind: int = 0,
):
    """Backwards-compatible entrypoint."""
    ctx = build_ram_context(
        input_xlsx=input_xlsx,
        start_date=start_date,
        end_date=end_date,
        simulations=simulations,
        agg=agg,
        opp_dt_ind=opp_dt_ind,
        spare_ind=spare_ind,
    )
    return run_ram_simulation_ctx(ctx)


def run_ram_simulation_ctx(ctx: RAMContext):
    """Run the model using an explicit RAMContext (no module globals)."""

    # Create convenient locals for legacy code blocks below
    start_date = ctx.start_date
    end_date = ctx.end_date
    input_xlsx = ctx.input_xlsx
    simulations = ctx.simulations
    agg = ctx.agg
    opp_dt_ind = ctx.opp_dt_ind
    spare_ind = ctx.spare_ind
    period_end = ctx.period_end
    date_range_years = ctx.date_range_years
    date_range_months = ctx.date_range_months
    date_range_days = ctx.date_range_days
    comp_att_df = ctx.comp_att_df
    num_comp = ctx.num_comp
    """
    Run the RAM simulation with standardized parameters and return organized outputs.

    Parameters:
    -----------
    ctx.input_xlsx : str
        Path to input Excel file
    ctx.start_date : datetime.date
        Start date for simulation
    ctx.end_date : datetime.date
        End date for simulation
    ctx.simulations : int, optional
        Number of ctx.simulations to run (default: 2)
    ctx.agg : str, optional
        Condition aggregation method (default: '50th_perc')
    ctx.opp_dt_ind : int, optional
        Whether to apply opportunistic downtime (default: 0)
    ctx.spare_ind : int, optional
        Whether to apply spare systems (default: 0)
    
    Returns:
    --------
    dict
        Dictionary containing all simulation outputs organized by category
    """
    # Calculate simulation period
    ctx.period_end = (ctx.end_date - ctx.start_date).days

    # Generate date ranges
    ctx.date_range_years = []
    for i in range(0, math.ceil((ctx.end_date-ctx.start_date).days/365)):
        dates = (ctx.start_date + relativedelta(years=i)).year
        ctx.date_range_years.append(dates)
    
    ctx.date_range_months = []
    for i in range(0, math.ceil((ctx.end_date-ctx.start_date).days/365*12)):
        dates = (ctx.start_date + relativedelta(months=i)).strftime("%Y-%m")
        ctx.date_range_months.append(dates)
    
    ctx.date_range_days = pd.date_range(ctx.start_date, ctx.end_date, freq='M', inclusive="both").strftime('%Y-%m-%d').tolist()

    # Load component attributes
    ctx.comp_att_df = pd.read_excel(ctx.input_xlsx, 'comp_att')
    ctx.num_comp = len(ctx.comp_att_df)

    # Extract timeline and usage data
    xl = pd.read_excel(ctx.input_xlsx, None)
    tl = [i for i in xl.keys() if 'timeline' in i]
    ub = [i for i in xl.keys() if 'usage' in i]

    for l in range(len(tl)):
        ctx.ws[str(tl[l])] = pd.read_excel(ctx.input_xlsx, str(tl[l]))
    for l in range(len(ub)):
        ctx.ws[str(ub[l])] = pd.read_excel(ctx.input_xlsx, str(ub[l]))

    # ... [Rest of the existing simulation code, but remove global variable assignments] ...

        # ------------------------------------------------------------
    # Collect component condition tables (optional, can be big)
    # ------------------------------------------------------------
    component_conditions: Dict[str, pd.DataFrame] = {}
    for i in range(ctx.num_comp):
        for j in range(ctx.simulations):
            k = f"comp{i}_{j}_cond"
            v = ctx.ws.get(k)
            if isinstance(v, pd.DataFrame):
                component_conditions[k] = v

    # ------------------------------------------------------------
    # Build structured outputs
    # Prefer local vars if your simulation code defines them;
    # otherwise pull from ctx.ws (works with your RAMContext approach).
    # ------------------------------------------------------------
    def _pick(name: str, local_fallback: Any = None):
        if local_fallback is not None:
            return local_fallback
        return ctx.ws.get(name)

    outputs_tables = {
        "yearly_component": _pick("yrly_comp_df", locals().get("yrly_comp_df")),
        "monthly_component": _pick("mthly_comp_df", locals().get("mthly_comp_df")),
        "yearly_subcomponent": _pick("yrly_sub_comp_df", locals().get("yrly_sub_comp_df")),
        "monthly_subcomponent": _pick("mthly_sub_comp_df", locals().get("mthly_sub_comp_df")),
        "yearly_simulations": _pick("yrly_all_comp_sim_df", locals().get("yrly_all_comp_sim_df")),
        "monthly_simulations": _pick("mthly_all_comp_sim_df", locals().get("mthly_all_comp_sim_df")),
    }

    # Optional: drop any non-DataFrame entries so downstream archiving is simple
    outputs_tables = {k: v for k, v in outputs_tables.items() if isinstance(v, pd.DataFrame)}

    percentiles = {}
    for p in [
        "min",
        "10th_perc",
        "20th_perc",
        "30th_perc",
        "40th_perc",
        "50th_perc",
        "60th_perc",
        "70th_perc",
        "80th_perc",
        "90th_perc",
        "max",
    ]:
        v = ctx.ws.get(f"all_comp_cond_df_{p}")
        if isinstance(v, pd.DataFrame):
            percentiles[p] = v

    conditions: Dict[str, Any] = {
        "condition_data": {
            "all_components": _pick("all_comp_cond_df", locals().get("all_comp_cond_df")),
            "simulations": _pick("all_comp_sims", locals().get("all_comp_sims")),
            "percentiles": percentiles,
            "component_conditions": component_conditions,
        }
    }

    # If all_components / simulations aren't DataFrames, remove them (keeps contract clean)
    cd = conditions["condition_data"]
    if not isinstance(cd.get("all_components"), pd.DataFrame):
        cd.pop("all_components", None)
    if not isinstance(cd.get("simulations"), pd.DataFrame):
        cd.pop("simulations", None)

    parameters = {
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "simulations": ctx.simulations,
        "aggregation": ctx.agg,
        "opportunistic_downtime": ctx.opp_dt_ind,
        "spare_systems": ctx.spare_ind,
        "component_attributes": ctx.comp_att_df,
        "date_ranges": {
            "years": ctx.date_range_years,
            "months": ctx.date_range_months,
            "days": ctx.date_range_days,
        },
    }

    return RAMResults(
        outputs=outputs_tables,
        conditions=conditions,
        parameters=parameters,
        debug=None,  # set to ctx.ws if you want full internals (can be huge)
    )

if __name__ == "__main__":
    # Example usage (manual)
    results = run_ram_simulation(
        input_xlsx=r"outputs\ram_input_sheet.xlsx",
        start_date=date(2011, 1, 1),
        end_date=date(2017, 12, 31),
        simulations=2,
        agg="50th_perc",
        opp_dt_ind=0,
        spare_ind=0,
    )
    print("Simulation complete. Output keys:", list(results.keys()))