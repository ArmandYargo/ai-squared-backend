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
    """
    Canonical result contract for the RAM model.

    The assistant pipeline expects:
      - outputs: flat dict[str, DataFrame]
      - conditions: dict[str, DataFrame | dict[str, DataFrame]]
      - parameters: dict[str, Any]
    """
    outputs: Dict[str, pd.DataFrame]
    conditions: Dict[str, Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
    parameters: Dict[str, Any]

    # Optional: keep a pointer to internal workspace/debug info if you want later
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

def time_to_event_func(ctx, comp, mp, rating, serv_ind, pa_ind, time_step_opr, prev_event_type=None, prev_damage=0, prev_weib_tf=None, acc_time_to_event=None, time_to_change=0, mp_change = "no"):
    
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

    else: print("error: invalid maintenance practice used or Care&Maintance is not functional yet")
    
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



def run_ram_simulation(
    input_xlsx: str,
    start_date: date,
    end_date: date,
    simulations: int = 200,
    agg: str = "50th_perc",
    opp_dt_ind: int = 0,
    spare_ind: int = 0,
):
    """Agent-safe entrypoint.

    IMPORTANT:
    - This function is the ONLY place the simulation executes.
    - Importing this module must have zero side effects.

    Returns a plain dict compatible with ram_results.save_ram_results().
    """
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
    """Run the RAM simulation using an explicit RAMContext.

    This function contains the *full* legacy model logic, moved from module scope
    so that the module can be imported safely by LangGraph tools.

    Returns a plain dict matching the expected contract:
      {
        'yearly_component': DataFrame,
        'monthly_component': DataFrame,
        'yearly_subcomponent': DataFrame,
        'monthly_subcomponent': DataFrame,
        'yearly_simulations': DataFrame,
        'monthly_simulations': DataFrame,
        'condition_data': {...},
        'parameters': {...}
      }
    """
    #simulation running information

    for j in range(ctx.simulations):
        #reset eta
        print('sim',j)
        ##!!! check if this works
        #ctx.ws['ctx.comp_att_df'] = pd.read_excel (r"%s" % ctx.input_xlsx,'comp_att')
        ctx.comp_att_df = pd.read_excel (r"%s" % ctx.input_xlsx,'comp_att')
    
        env = simpy.Environment()

        for i in range(ctx.num_comp):
        
            #initialise output arrays for each component separately
            ctx.ws['comp%s' % i + '_%s' % j] = np.empty(shape=(0,15), dtype=object)
            #add independent components to simulation environment
            env.process(component(ctx, env,i))

        env.run(until=ctx.period_end)

    ## Manipulate/aggregate results

    #create dataframes for each component i and simulation j

    for i in range(ctx.num_comp):
        for j in range(ctx.simulations):
     
            ctx.ws['comp%s' % i + '_%s' % j + '_df'] = pd.DataFrame(ctx.ws['comp%s' % i + '_%s' % j], columns = ['component','event','event_desc','mp_assigned','sim_date','year','month','day','cond','mtbf','mttr','availability','cost','damage','failure_time'])
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['month'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['month'].apply(lambda x: '{0:0>2}'.format(x))
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['year_month'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['year']+'-'+ctx.ws['comp%s' % i + '_%s' % j + '_df']['month']

            #add failure count
            # using dictionary to convert specific columns
            convert_dict = {'component': int,
                            'event': int,
                            'event_desc': object,
                            'mp_assigned': object,
                            'sim_date': object,
                            'year': int,
                            'month': int,
                            'day': int,
                            'cond': float,
                            'mtbf': float,
                            'mttr': float,
                            'availability': float,
                            'cost': float,
                            'damage': float,
                            'failure_time': float,
                            'year_month': object
                            }

            ctx.ws['comp%s' % i + '_%s' % j + '_df'] = ctx.ws['comp%s' % i + '_%s' % j + '_df'].astype(convert_dict)
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['failures'] = (ctx.ws['comp%s' % i + '_%s' % j + '_df']['event_desc']=='Failure')*1
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['replacements'] = (ctx.ws['comp%s' % i + '_%s' % j + '_df']['event_desc']=='Replacement')*1
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['repairs'] = (ctx.ws['comp%s' % i + '_%s' % j + '_df']['event_desc']=='Repair')*1
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['fail_dt'] = (ctx.ws['comp%s' % i + '_%s' % j + '_df']['event_desc']=='Failure')*ctx.ws['comp%s' % i + '_%s' % j + '_df']['mttr']

            #accumulate over mttr and cost
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['cum_downtime'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['mttr'].cumsum()
            ctx.ws['comp%s' % i + '_%s' % j + '_df']['cum_cost'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['cost'].cumsum()

    #CONDITION
    #calculate monthly condition for each component and simulation run

    for i in range(ctx.num_comp):
        for j in range(ctx.simulations):
            ##extract event dates from start of simulation
            sim_date_ev = [datetime.strptime(i,"%Y-%m-%d").date() for i in ctx.ws['comp%s' % i + '_%s' % j + '_df']['sim_date']]

            ##initialise first stretch of simulation

            #fill months between start and first event and calculate accumulated days
            if len(sim_date_ev) > 0:
                first_date_list = pd.date_range(ctx.start_date, sim_date_ev[0], freq='M', inclusive="both").strftime('%Y-%m-%d').tolist()
                first_date_list.insert(0,ctx.start_date.strftime('%Y-%m-%d'))
                first_date_list.append(sim_date_ev[0].strftime('%Y-%m-%d'))
            else:
                # If no events, just use the full date range
                first_date_list = pd.date_range(ctx.start_date, ctx.end_date, freq='M', inclusive="both").strftime('%Y-%m-%d').tolist()
                first_date_list.insert(0,ctx.start_date.strftime('%Y-%m-%d'))
                first_date_list.append(ctx.end_date.strftime('%Y-%m-%d'))

            ctx.ws['comp%s' % i + '_%s' % j + '_cond'] = pd.DataFrame(first_date_list, columns=['date'])
            ctx.ws['comp%s' % i + '_%s' % j + '_cond']['days'] = pd.to_datetime(ctx.ws['comp%s' % i + '_%s' % j + '_cond']['date']).diff().dt.days.cumsum()
            ctx.ws['comp%s' % i + '_%s' % j + '_cond']['days'] = ctx.ws['comp%s' % i + '_%s' % j + '_cond']['days'].fillna(0)

            #add mttr, n, prev_damage and prev_weibtf - duplicate from event backwards
            if len(sim_date_ev) > 0:
                ctx.ws['comp%s' % i + '_%s' % j + '_cond']['mttr'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['mttr'][0]
                ctx.ws['comp%s' % i + '_%s' % j + '_cond']['failure_time'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['failure_time'][0]
            else:
                # If no events, use default values
                ctx.ws['comp%s' % i + '_%s' % j + '_cond']['mttr'] = 0
                ctx.ws['comp%s' % i + '_%s' % j + '_cond']['failure_time'] = 365  # Default to 1 year

            ctx.ws['comp%s' % i + '_%s' % j + '_cond']['pf_n'] = ctx.comp_att_df.loc[i].loc["pf_n"]
            ctx.ws['comp%s' % i + '_%s' % j + '_cond']['i_damage'] = 0

            ##calculate condition
            #cond = 1 - (d_i**n + (t/tf))**(1/n)
            ctx.ws['comp%s' % i + '_%s' % j + '_cond']['cond'] = 1 - ((ctx.ws['comp%s' % i + '_%s' % j + '_cond']['i_damage']*ctx.ws['comp%s' % i + '_%s' % j + '_cond']['pf_n']) + (ctx.ws['comp%s' % i + '_%s' % j + '_cond']['days']/ctx.ws['comp%s' % i + '_%s' % j + '_cond']['failure_time'])**(1/ctx.ws['comp%s' % i + '_%s' % j + '_cond']['pf_n']))

            #repeat steps for all events
            for k in range(1,len(sim_date_ev)):
                #add mttr after last event date and fill months between events
                temp_mttr_date = (pd.to_datetime(ctx.ws['comp%s' % i + '_%s' % j + '_cond']['date'].iloc[-1]) + timedelta(days = ctx.ws['comp%s' % i + '_%s' % j + '_cond']['mttr'].iloc[-1])).strftime('%Y-%m-%d')
                temp_date_list = pd.date_range(temp_mttr_date, sim_date_ev[k], freq='M', inclusive="both").strftime('%Y-%m-%d').tolist()
                temp_date_list.insert(0,temp_mttr_date)
                temp_date_list.append(sim_date_ev[k].strftime('%Y-%m-%d'))

                temp_cond_df = pd.DataFrame(temp_date_list, columns=['date'])
                temp_cond_df['days'] = pd.to_datetime(temp_cond_df['date']).diff().dt.days.cumsum()
                temp_cond_df['days'] = temp_cond_df['days'].fillna(0)

                #add mttr, n, prev_damage and prev_weibtf - duplicate from event backwards
                temp_cond_df['mttr'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['mttr'][k]
                temp_cond_df['pf_n'] = ctx.comp_att_df.loc[i].loc["pf_n"]
                temp_cond_df['i_damage'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['damage'][k-1]
                temp_cond_df['failure_time'] = ctx.ws['comp%s' % i + '_%s' % j + '_df']['failure_time'][k]

                ##calculate condition
                #cond = 1 - (d_i**n + (t/tf))**(1/n)
                temp_cond_df['cond'] = 1 - ((temp_cond_df['i_damage']**temp_cond_df['pf_n']) + (temp_cond_df['days']/temp_cond_df['failure_time'])**(1/temp_cond_df['pf_n']))

                #append condition information to component level array
                ctx.ws['comp%s' % i + '_%s' % j + '_cond'] = pd.concat([
                    ctx.ws['comp%s' % i + '_%s' % j + '_cond'],
                    temp_cond_df
                ], ignore_index=True)
        
            #change small negative numbers for failures to 0
            ctx.ws['comp%s' % i + '_%s' % j + '_cond']['cond'][ctx.ws['comp%s' % i + '_%s' % j + '_cond']['cond']<0] = 0
        
            #drop duplacte rows
            ctx.ws['comp%s' % i + '_%s' % j + '_cond'] = ctx.ws['comp%s' % i + '_%s' % j + '_cond'].drop_duplicates()
        
    #remove inactive data for each component i and simulation j
    for i in range(ctx.num_comp):

        #get active ind from excel spreadsheet
        time_link = ctx.comp_att_df.loc[i].loc["time_tbl"]
        act_df = ctx.ws['timeline_%s' % time_link][['from','to','active_ind']][ctx.ws['timeline_%s' % time_link]['active_ind']==0]

        #only if there are inactive periods:
        if act_df.shape[0] > 0:
            for l in range(act_df.shape[0]):
                act_ind = act_df.iloc[l]

                #convert 'from' and 'to' columns to datetime format
                act_ind['from'] = pd.to_datetime(act_ind['from'])
                act_ind['to'] = pd.to_datetime(act_ind['to'])

                for j in range(ctx.simulations):

                    #convert the date column in numpy array to datetime format
                    data_dates = pd.to_datetime(ctx.ws['comp%s' % i + '_%s' % j + '_df']['sim_date'])

                    #create a boolean mask for rows in array that fall between the specified dates
                    mask = ~((data_dates >= act_ind['from']) & (data_dates <= act_ind['to']))

                    #apply the mask to filter out rows from array
                    ctx.ws['comp%s' % i + '_%s' % j + '_df'] = ctx.ws['comp%s' % i + '_%s' % j + '_df'][mask]

    #CONDITION
    #remove inactive data for each component i and simulation j
    for i in range(ctx.num_comp):

        #get active ind from excel spreadsheet
        time_link = ctx.comp_att_df.loc[i].loc["time_tbl"]
        act_df = ctx.ws['timeline_%s' % time_link][['from','to','active_ind']][ctx.ws['timeline_%s' % time_link]['active_ind']==0]

        #only if there are inactive periods:
        if act_df.shape[0] > 0:
            for l in range(act_df.shape[0]):
                act_ind = act_df.iloc[l]

                #convert 'from' and 'to' columns to datetime format
                act_ind['from'] = pd.to_datetime(act_ind['from'])
                act_ind['to'] = pd.to_datetime(act_ind['to'])

                for j in range(ctx.simulations):

                    #convert the date column in numpy array to datetime format
                    data_dates = pd.to_datetime(ctx.ws['comp%s' % i + '_%s' % j + '_cond']['date'])

                    #create a boolean mask for rows in array that fall between the specified dates
                    mask = ~((data_dates >= act_ind['from']) & (data_dates <= act_ind['to']))

                    #apply the mask to filter out rows from array
                    ctx.ws['comp%s' % i + '_%s' % j + '_cond'] = ctx.ws['comp%s' % i + '_%s' % j + '_cond'][mask]

    #apply opportunistic downtime for availability calculations

    if ctx.opp_dt_ind == 1:
    
        #add opportunistic downtime (if two component in a simulation fail at the same time fixing can be done simulateously instead of counting it twice)
        for j in range(ctx.simulations):
            ctx.ws['opp_dt_sim_%s' % j] = pd.DataFrame(columns = ['sim_date']+['mttr_'+str(i) for i in range(ctx.num_comp)])

            #check which combination of components fail on the same day
            for i in range(ctx.num_comp):
                for k in range(i+1,ctx.num_comp):
                    opp_dt_one_temp = pd.merge(ctx.ws['comp%s' % i + '_%s' % j + '_df'][['sim_date','mttr']], ctx.ws['comp%s' % k + '_%s' % j + '_df'][['sim_date','mttr']], how= 'inner', on='sim_date', suffixes=(str(i),str(k)) )
                    if opp_dt_one_temp.shape[0] > 0:
                        ctx.ws['opp_dt_sim_%s' % j] = pd.concat([
                            ctx.ws['opp_dt_sim_%s' % j],
                            pd.DataFrame({
                                'sim_date': opp_dt_one_temp['sim_date'],
                                'mttr_'+str(i): opp_dt_one_temp['mttr'+str(i)],
                                'mttr_'+str(k): opp_dt_one_temp['mttr'+str(k)]
                            })
                        ], ignore_index=True)

            #group by sim_date to obtain more than one overlap
            ctx.ws['opp_dt_groups_sim_%s' % j] = ctx.ws['opp_dt_sim_%s' % j].groupby('sim_date', as_index = False).ctx.agg({'mttr_'+str(l) : 'max' for l in range(ctx.num_comp)})
            #identify first component where max mttr is found
            ctx.ws['opp_dt_groups_sim_%s' % j]['mttr_max'] = ctx.ws['opp_dt_groups_sim_%s' % j].drop('sim_date', axis = 1).max(axis = 1)
            ctx.ws['opp_dt_groups_sim_%s' % j]['max_comp'] = ctx.ws['opp_dt_groups_sim_%s' % j].drop('sim_date', axis = 1).idxmax(axis = 1)

        #override mttr to 0 for components (not the max comp) where ovelapping failures occur in the same day
        for j in range(ctx.simulations):
            for i in range(ctx.num_comp):
                compi_temp = ctx.ws['opp_dt_groups_sim_%s' % j][(~ctx.ws['opp_dt_groups_sim_%s' % j]['mttr_'+str(i)].isna())&(ctx.ws['opp_dt_groups_sim_%s' % j]['max_comp']!= 'mttr_'+str(i))]
                if compi_temp.shape[0] > 0:
                    for l in range(compi_temp.shape[0]):
                        ctx.ws['comp%s' % i + '_%s' % j + '_df'].loc[ctx.ws['comp%s' % i + '_%s' % j + '_df'].index[ctx.ws['comp%s' % i + '_%s' % j + '_df']['sim_date']==compi_temp['sim_date'].values[l]][0], "mttr"] = 0

    #apply condition for swapping in spare buses to increase service availability

    #override mttr to 1 day for components that has a mttr of more than 1 day
    if ctx.spare_ind == 1:
        for j in range(ctx.simulations):
            for i in range(ctx.num_comp):
                compi_j_temp = ctx.ws['comp%s' % i + '_%s' % j + '_df'][ctx.ws['comp%s' % i + '_%s' % j + '_df']['mttr'] > 1]

                if compi_j_temp.shape[0] > 0:
                    for l in range(compi_j_temp.shape[0]):
                        ctx.ws['comp%s' % i + '_%s' % j + '_df'].loc[ctx.ws['comp%s' % i + '_%s' % j + '_df'].index[ctx.ws['comp%s' % i + '_%s' % j + '_df']['sim_date']==compi_j_temp['sim_date'].values[l]][0], 'mttr'] = 1
    #convert events to months and years

    left_y = {'year':ctx.date_range_years}
    left_m = {'year_month':ctx.date_range_months}

    for i in range(ctx.num_comp):
        for j in range(ctx.simulations):

            #average over mtbf & mttr and sum over downtime & cost & failures & fail_dt - group by months and years
            ctx.ws['avg_y_comp%s' % i + '_%s' % j + '_df'] = ctx.ws['comp%s' % i + '_%s' % j + '_df'].groupby('year', as_index=False).ctx.agg(component=('component','mean'), mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('mttr','sum'), cost=('cost','sum'), events=('year','count'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))
            ctx.ws['avg_m_comp%s' % i + '_%s' % j + '_df'] = ctx.ws['comp%s' % i + '_%s' % j + '_df'].groupby('year_month', as_index=False).ctx.agg(component=('component','mean'), mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('mttr','sum'), cost=('cost','sum'), events=('year','count'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))

            #ensure sequence for all years
            ctx.ws['left_y_comp%s' % i + '_%s' % j + '_df'] = pd.DataFrame(data=left_y)
            ctx.ws['left_y_comp%s' % i + '_%s' % j + '_df']['component'] = float(i)

            ctx.ws['left_m_comp%s' % i + '_%s' % j + '_df'] = pd.DataFrame(data=left_m)
            ctx.ws['left_m_comp%s' % i + '_%s' % j + '_df']['component'] = float(i)

            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df'] = pd.merge(ctx.ws['left_y_comp%s' % i + '_%s' % j + '_df'], ctx.ws['avg_y_comp%s' % i + '_%s' % j + '_df'], how='left', on=['year','component'])
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df'] = pd.merge(ctx.ws['left_m_comp%s' % i + '_%s' % j + '_df'], ctx.ws['avg_m_comp%s' % i + '_%s' % j + '_df'], how='left', on=['year_month','component'])

            # replace NaN values with 0 where no events are detected.
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['downtime'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['downtime'].fillna(0)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['cost'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['cost'].fillna(0)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['events'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['events'].fillna(0)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['failures'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['failures'].fillna(0)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['replacements'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['replacements'].fillna(0)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['repairs'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['repairs'].fillna(0)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['fail_dt'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['fail_dt'].fillna(0)

            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['downtime'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['downtime'].fillna(0)
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['cost'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['cost'].fillna(0)
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['events'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['events'].fillna(0)
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['failures'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['failures'].fillna(0)
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['replacements'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['replacements'].fillna(0)
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['repairs'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['repairs'].fillna(0)
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['fail_dt'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['fail_dt'].fillna(0)
        
            #recalculate availability, add accumulated dowtime and cost
            avail_y = 1 - (ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['downtime']/365)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df'].insert(loc=4, column="availability",value=avail_y)
            avail_y_fail = 1 - (ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['fail_dt']/365)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df'].insert(loc=5, column="fail_avail",value=avail_y_fail)
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['cum_downtime'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['downtime'].cumsum()
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['cum_cost'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df']['cost'].cumsum()

            avail_m = 1 - (mthly_comp0_0_df['downtime']/(365/12))
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df'].insert(loc=4, column="availability",value=avail_m)
            avail_m_fail = 1 - (mthly_comp0_0_df['fail_dt']/(365/12))
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df'].insert(loc=5, column="fail_avail",value=avail_m_fail)
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['cum_downtime'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['downtime'].cumsum()
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['cum_cost'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df']['cost'].cumsum()

            #convert to numpy
            ctx.ws['yrly_comp%s' % i + '_%s' % j + '_np'] = ctx.ws['yrly_comp%s' % i + '_%s' % j + '_df'].drop(['availability','fail_avail','cum_downtime','cum_cost'],axis=1).to_numpy()
            ctx.ws['mthly_comp%s' % i + '_%s' % j + '_np'] = ctx.ws['mthly_comp%s' % i + '_%s' % j + '_df'].drop(['year_month','availability','fail_avail','cum_downtime','cum_cost'], axis=1).to_numpy()
        
    #combine ctx.simulations into 3d array for each component

    for i in range(ctx.num_comp):
        #initialise array shape using results of first simulation run of specific component
        ctx.ws['yrly_comp%s' % i + '_np'] = np.empty((ctx.ws['yrly_comp%s' % i + '_0_np'].shape[0],11,0),dtype='float64')
        ctx.ws['mthly_comp%s' % i + '_np'] = np.empty((ctx.ws['mthly_comp%s' % i + '_0_np'].shape[0],10,0),dtype='float64')
        for j in range(ctx.simulations):
            ctx.ws['yrly_comp%s' % i + '_np'] = np.append(ctx.ws['yrly_comp%s' % i + '_np'],np.atleast_3d(ctx.ws['yrly_comp%s' % i + '_%s' % j + '_np']),axis=2)
            ctx.ws['mthly_comp%s' % i + '_np'] = np.append(ctx.ws['mthly_comp%s' % i + '_np'],np.atleast_3d(ctx.ws['mthly_comp%s' % i + '_%s' % j + '_np']),axis=2)
        
    ## Aggregated Simulations
    #CONDITION
    #aggregate ctx.simulations of condition data for each component based on 5 number summary

    left_m_d = {'date':ctx.date_range_days}

    for i in range(ctx.num_comp):
    ##initialise first simulation
        ctx.ws['comp%s' % i + '_cond'] = pd.DataFrame(data=left_m_d)
        ctx.ws['comp%s' % i + '_cond'] = pd.merge(ctx.ws['comp%s' % i + '_cond'], ctx.ws['comp%s' % i + '_0_cond'][['date', 'cond']], how='left', on='date')
    
        for j in range(1,ctx.simulations):
        
            ##inner join condition of all ctx.simulations
            ctx.ws['comp%s' % i + '_cond'] = pd.merge(ctx.ws['comp%s' % i + '_cond'], ctx.ws['comp%s' % i + '_%s' % j + '_cond'][['date','cond']], how='left', on='date', suffixes=(str(j-1),str(j)))
    
            #identify indexes with duplicate dates
            dup_df = ctx.ws['comp%s' % i + '_cond']['date'][ctx.ws['comp%s' % i + '_cond']['date'].duplicated(keep=False)]
            dup_df = dup_df.groupby(list(dup_df)).apply(lambda x: list(x.index))
            dup_df_df = pd.DataFrame(dup_df, columns=['date']).rename(columns={'date':'ind'})
            #filter duplicates of more than 2
            dup_df_df['len'] = dup_df_df.apply(lambda x: len(x['ind']), axis =1)
            dup_df_df = dup_df_df[dup_df_df['len'] > 2]
            #keep only first and last of duplicates
            if dup_df_df.empty == False:
                dup_df_df['drop'] = dup_df_df.apply(lambda x: x['ind'][1:-1], axis=1)
                #remove specified duplicates from original dataset
                drop_list = [i for s in dup_df_df['drop'] for i in s]
                ctx.ws['comp%s' % i + '_cond'].drop(drop_list, axis = 0, inplace = True)    
    
    
        ctx.ws['comp%s' % i + '_cond'].rename(columns={"cond": "cond"+str(ctx.simulations-1)}, inplace=True)

        ##calculate 5 number summary values
        base_conditions = ctx.ws['comp%s' % i + '_cond'].drop(['date'], axis = 1).astype('float64')
        #min
        ctx.ws['comp%s' % i + '_cond']['cond_min'] = base_conditions.min(axis = 1)
        #10th percentile
        ctx.ws['comp%s' % i + '_cond']['cond_10th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 10), axis=1)
        #20th percentile
        ctx.ws['comp%s' % i + '_cond']['cond_20th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 20), axis=1)
        #30th percentile
        ctx.ws['comp%s' % i + '_cond']['cond_30th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 30), axis=1)
        #40th percentile
        ctx.ws['comp%s' % i + '_cond']['cond_40th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 40), axis=1)
        #50th percentile / mean 
        ctx.ws['comp%s' % i + '_cond']['cond_50th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 50), axis=1)
        #60th percentile 
        ctx.ws['comp%s' % i + '_cond']['cond_60th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 60), axis=1)
        #70th percentile  
        ctx.ws['comp%s' % i + '_cond']['cond_70th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 70), axis=1)
        #80th percentile  
        ctx.ws['comp%s' % i + '_cond']['cond_80th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 80), axis=1)
        #90th percentile
        ctx.ws['comp%s' % i + '_cond']['cond_90th_perc'] = base_conditions.apply(lambda x: np.nanpercentile(x, 90), axis=1)
        #min
        ctx.ws['comp%s' % i + '_cond']['cond_max'] = base_conditions.max(axis = 1)
    
    
    #CONDITION
    #aggregate all components based on their weight allocation 
    #weight allocation are per main asset group

    ##initialise first component
    all_comp_cond_df = comp0_cond[['date','cond_' + str(ctx.agg)]].copy()

    for i in range(1,ctx.num_comp):

        ##inner join condition of all components into one dataframe
        all_comp_cond_df = pd.merge(all_comp_cond_df, ctx.ws['comp%s' % i + '_cond'][['date','cond_' + str(ctx.agg)]], how='left', on='date', suffixes=(str(i-1),str(i)))  
        
        #identify indexes with duplicate dates
        dup_df = all_comp_cond_df['date'][all_comp_cond_df['date'].duplicated(keep=False)]
        dup_df = dup_df.groupby(list(dup_df)).apply(lambda x: list(x.index))
        dup_df_df = pd.DataFrame(dup_df, columns=['date']).rename(columns={'date':'ind'})
        #filter duplicates of more than 2
        dup_df_df['len'] = dup_df_df.apply(lambda x: len(x['ind']), axis =1)
        dup_df_df = dup_df_df[dup_df_df['len'] > 2]
        #keep only first and last of duplicates
        if dup_df_df.empty == False:
            dup_df_df['drop'] = dup_df_df.apply(lambda x: x['ind'][1:-1], axis=1)
            #remove specified duplicates from original dataset
            drop_list = [i for s in dup_df_df['drop'] for i in s]
            all_comp_cond_df.drop(drop_list, axis = 0, inplace = True)         
        
    all_comp_cond_df.rename(columns={'cond_' + str(ctx.agg): 'cond_' + str(ctx.agg) + str(ctx.num_comp-1)}, inplace=True)

    #drop nan rows at this point because you cannot assign a weigted average to rows entirely empty
    all_comp_cond_df = all_comp_cond_df.dropna(subset=all_comp_cond_df.columns.difference(['date']),how = 'all')

    ##calculate aggregation values based on weights
    #create list of weigths
    weight_list = ctx.comp_att_df['weight'].tolist()

    base_conditions = all_comp_cond_df.drop(['date'], axis = 1).astype('float64')

    #calculate mean
    all_comp_cond_df['cond_avg'] = base_conditions.mean(axis=1)

    #update weight_List where comp are inactive
    weight_avg_list = []
    for idx, row in base_conditions.iterrows():
        row_values = row.values
        nan_mask = np.isnan(row_values)
        new_sum = 1 - sum([weight_list[i] if nan_mask[i] else 0.0 for i in range(len(row_values))])
        updated_weight_list = [weight_list[i]/new_sum if not nan_mask[i] else 0.0 for i in range(len(row_values))]
        weighted_avg = np.average(np.nan_to_num(row_values), weights = updated_weight_list)
        weight_avg_list.append(weighted_avg)

    all_comp_cond_df['cond_weighted_avg'] = weight_avg_list
    #all_comp_cond_df['cond_weighted_avg'] = base_conditions.mul(weight_list).sum(axis=1)
    

    #CONDITION
    #create tables with each of the percentiles

    choices = ['min', '10th_perc', '20th_perc', '30th_perc', '40th_perc', '50th_perc', '60th_perc', '70th_perc', '80th_perc', '90th_perc', 'max']

    for l in range(len(choices)):
    
        ctx.agg = choices[l]
    
        ##initialise first component
        ctx.ws['all_comp_cond_df_%s' % ctx.agg] = comp0_cond[['date','cond_' + str(ctx.agg)]].copy()
    
        for i in range(1,ctx.num_comp):
        
            ##inner join condition of all components into one dataframe
            ctx.ws['all_comp_cond_df_%s' % ctx.agg] = pd.merge(ctx.ws['all_comp_cond_df_%s' % ctx.agg], ctx.ws['comp%s' % i + '_cond'][['date','cond_' + str(ctx.agg)]], how='left', on='date', suffixes=(str(i-1),str(i)))  
    
            #identify indexes with duplicate dates
            dup_df = ctx.ws['all_comp_cond_df_%s' % ctx.agg]['date'][ctx.ws['all_comp_cond_df_%s' % ctx.agg]['date'].duplicated(keep=False)]
            dup_df = dup_df.groupby(list(dup_df)).apply(lambda x: list(x.index))
            dup_df_df = pd.DataFrame(dup_df, columns=['date']).rename(columns={'date':'ind'})
            #filter duplicates of more than 2
            dup_df_df['len'] = dup_df_df.apply(lambda x: len(x['ind']), axis =1)
            dup_df_df = dup_df_df[dup_df_df['len'] > 2]
            #keep only first and last of duplicates
            if dup_df_df.empty == False:
                dup_df_df['drop'] = dup_df_df.apply(lambda x: x['ind'][1:-1], axis=1)
                #remove specified duplicates from original dataset
                drop_list = [i for s in dup_df_df['drop'] for i in s]
                ctx.ws['all_comp_cond_df_%s' % ctx.agg].drop(drop_list, axis = 0, inplace = True)     
    
        ctx.ws['all_comp_cond_df_%s' % ctx.agg].rename(columns={'cond_' + str(ctx.agg): 'cond_' + str(ctx.agg) + str(ctx.num_comp-1)}, inplace=True)

        #drop nan rows at this point because you cannot assign a weigted average to rows entirely empty
        ctx.ws['all_comp_cond_df_%s' % ctx.agg] = ctx.ws['all_comp_cond_df_%s' % ctx.agg].dropna(subset=ctx.ws['all_comp_cond_df_%s' % ctx.agg].columns.difference(['date']),how = 'all')

        ##calculate aggregation values based on weights
        #create list of weigths
        weight_list = ctx.comp_att_df['weight'].tolist()

        base_conditions = ctx.ws['all_comp_cond_df_%s' % ctx.agg].drop(['date'], axis = 1).astype('float64')
    
        #calculate mean
        ctx.ws['all_comp_cond_df_%s' % ctx.agg]['cond_avg_'+str(ctx.agg)] = base_conditions.mean(axis=1)
    
        #update weight_List where comp are inactive
        weight_avg_list = []
        for idx, row in base_conditions.iterrows():
            row_values = row.values
            nan_mask = np.isnan(row_values)
            new_sum = 1 - sum([weight_list[i] if nan_mask[i] else 0.0 for i in range(len(row_values))])
            updated_weight_list = [weight_list[i]/new_sum if not nan_mask[i] else 0.0 for i in range(len(row_values))]
            weighted_avg = np.average(np.nan_to_num(row_values), weights = updated_weight_list)
            weight_avg_list.append(weighted_avg)

        ctx.ws['all_comp_cond_df_%s' % ctx.agg]['cond_weighted_avg_'+str(ctx.agg)] = weight_avg_list
        #ctx.ws['all_comp_cond_df_%s' % ctx.agg]['cond_weighted_avg_'+str(ctx.agg)] = base_conditions.mul(weight_list).sum(axis=1) 

    #merge ctx.simulations 

    for i in range(ctx.num_comp):

        ctx.ws['yrly_comp%s' % i + '_avg']= np.nanmean(ctx.ws['yrly_comp%s' % i + '_np'], axis=2)
        ctx.ws['mthly_comp%s' % i + '_avg'] = np.nanmean(ctx.ws['mthly_comp%s' % i + '_np'], axis=2)

        ctx.ws['yrly_comp%s' % i + '_df'] = pd.DataFrame(ctx.ws['yrly_comp%s' % i + '_avg'], columns=['year','component', 'mtbf', 'mttr', 'downtime', 'cost', 'events', 'failures', 'replacements', 'repairs', 'fail_dt'])
        ctx.ws['mthly_comp%s' % i + '_df'] = pd.DataFrame(ctx.ws['mthly_comp%s' % i + '_avg'], columns=['component', 'mtbf', 'mttr', 'downtime', 'cost', 'events', 'failures', 'replacements', 'repairs', 'fail_dt'])
        ctx.ws['mthly_comp%s' % i + '_df'].insert(loc=0, column="year_month",value=left_m['year_month'])

        #recalculate availability and accumulated values 
        avail_y = 1 - (ctx.ws['yrly_comp%s' % i + '_df']['downtime']/365)
        ctx.ws['yrly_comp%s' % i + '_df'].insert(loc=4, column="availability",value=avail_y)
        avail_y_fail = 1 - (ctx.ws['yrly_comp%s' % i + '_df']['fail_dt']/365)
        ctx.ws['yrly_comp%s' % i + '_df'].insert(loc=5, column="fail_avail",value=avail_y_fail)
        ctx.ws['yrly_comp%s' % i + '_df']['cum_downtime'] = ctx.ws['yrly_comp%s' % i + '_df']['downtime'].cumsum()
        ctx.ws['yrly_comp%s' % i + '_df']['cum_cost'] = ctx.ws['yrly_comp%s' % i + '_df']['cost'].cumsum()
        ctx.ws['yrly_comp%s' % i + '_df'] = ctx.ws['yrly_comp%s' % i + '_df'].astype({'year':'int'})

        avail_m = 1 - (ctx.ws['mthly_comp%s' % i + '_df']['downtime']/(365/12))
        ctx.ws['mthly_comp%s' % i + '_df'].insert(loc=4, column="availability",value=avail_m)
        avail_m_fail = 1 - (ctx.ws['mthly_comp%s' % i + '_df']['fail_dt']/(365/12))
        ctx.ws['mthly_comp%s' % i + '_df'].insert(loc=5, column="fail_avail",value=avail_m_fail)
        ctx.ws['mthly_comp%s' % i + '_df']['cum_downtime'] = ctx.ws['mthly_comp%s' % i + '_df']['downtime'].cumsum()
        ctx.ws['mthly_comp%s' % i + '_df']['cum_cost'] = ctx.ws['mthly_comp%s' % i + '_df']['cost'].cumsum()

        #join component and subcomponent descritpions
        string, substring = comp_num_to_string(ctx, i)
        ctx.ws['yrly_comp%s' % i + '_df'].insert(2,'component_desc',string)
        ctx.ws['yrly_comp%s' % i + '_df'].insert(3,'subcomponent_desc',substring)
        ctx.ws['mthly_comp%s' % i + '_df'].insert(2,'component_desc',string)
        ctx.ws['mthly_comp%s' % i + '_df'].insert(3,'subcomponent_desc',substring)
    
    #combine all components into one dataframe

    yrly_all_comp_df = pd.DataFrame()
    mthly_all_comp_df = pd.DataFrame()
    for i in range(ctx.num_comp):
        yrly_all_comp_df = pd.concat([yrly_all_comp_df,ctx.ws['yrly_comp%s' % i + '_df']],ignore_index=True, axis =0)
        mthly_all_comp_df = pd.concat([mthly_all_comp_df,ctx.ws['mthly_comp%s' % i + '_df']],ignore_index=True, axis =0)


    #aggregate over component

    yrly_comp_df = yrly_all_comp_df.groupby(['component_desc','year'], as_index=False).ctx.agg(mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('downtime','sum'), cost=('cost','sum'), events=('events','sum'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))
    yrly_comp_df['availability'] = 1 - (yrly_comp_df['downtime']/365)
    yrly_comp_df['fail_avail'] = 1 - (yrly_comp_df['fail_dt']/365)
    yrly_comp_df['cum_downtime'] = yrly_comp_df['downtime'].cumsum()
    yrly_comp_df['cum_cost'] = yrly_comp_df['cost'].cumsum()

    mthly_comp_df = mthly_all_comp_df.groupby(['component_desc','year_month'], as_index=False).ctx.agg(mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('downtime','sum'), cost=('cost','sum'), events=('events','sum'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))
    mthly_comp_df['availability'] = 1 - (mthly_comp_df['downtime']/(365/12))
    mthly_comp_df['fail_avail'] = 1 - (mthly_comp_df['fail_dt']/(365/12))
    mthly_comp_df['cum_downtime'] = mthly_comp_df['downtime'].cumsum()
    mthly_comp_df['cum_cost'] = mthly_comp_df['cost'].cumsum()

    #aggregate over subcomponent

    yrly_sub_comp_df = yrly_all_comp_df.groupby(['component_desc','subcomponent_desc','year'], as_index=False).ctx.agg(mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('downtime','sum'), cost=('cost','sum'), events=('events','sum'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))
    yrly_sub_comp_df['availability'] = 1 - (yrly_sub_comp_df['downtime']/365)
    yrly_sub_comp_df['fail_avail'] = 1 - (yrly_sub_comp_df['fail_dt']/365)
    yrly_sub_comp_df['cum_downtime'] = yrly_sub_comp_df['downtime'].cumsum()
    yrly_sub_comp_df['cum_cost'] = yrly_sub_comp_df['cost'].cumsum()

    mthly_sub_comp_df = mthly_all_comp_df.groupby(['component_desc','subcomponent_desc','year_month'], as_index=False).ctx.agg(mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('downtime','sum'), cost=('cost','sum'), events=('events','sum'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))
    mthly_sub_comp_df['availability'] = 1 - (mthly_sub_comp_df['downtime']/(365/12))
    mthly_sub_comp_df['fail_avail'] = 1 - (mthly_sub_comp_df['fail_dt']/(365/12))
    mthly_sub_comp_df['cum_downtime'] = mthly_sub_comp_df['downtime'].cumsum()
    mthly_sub_comp_df['cum_cost'] = mthly_sub_comp_df['cost'].cumsum()

    ## Non-aggregated ctx.simulations

    #CONDITION
    #aggregate component based on their weights (ctx.simulations)
    #weight allocation are per main asset group

    for j in range(ctx.simulations):
    
        ##add condition of each component to simulation dataframe
        ctx.ws['all_comp_sim%s' % j] = comp0_cond[['date','cond'+str(j)]]

        for i in range(1,ctx.num_comp):

            ctx.ws['all_comp_sim%s' % j] = pd.merge(ctx.ws['all_comp_sim%s' % j], ctx.ws['comp%s' % i + '_cond'][['date','cond'+str(j)]], how='left', on='date', suffixes=('_'+str(i-1),'_'+str(i)))  

            #identify indexes with duplicate dates
            dup_df = ctx.ws['all_comp_sim%s' % j]['date'][ctx.ws['all_comp_sim%s' % j]['date'].duplicated(keep=False)]
            dup_df = dup_df.groupby(list(dup_df)).apply(lambda x: list(x.index))
            dup_df_df = pd.DataFrame(dup_df, columns=['date']).rename(columns={'date':'ind'})
            #filter duplicates of more than 2
            dup_df_df['len'] = dup_df_df.apply(lambda x: len(x['ind']), axis =1)
            dup_df_df = dup_df_df[dup_df_df['len'] > 2]
            #keep only first and last of duplicates
            if dup_df_df.empty == False:
                dup_df_df['drop'] = dup_df_df.apply(lambda x: x['ind'][1:-1], axis=1)
                #remove specified duplicates from original dataset
                drop_list = [i for s in dup_df_df['drop'] for i in s]
                ctx.ws['all_comp_sim%s' % j].drop(drop_list, axis = 0, inplace = True)         
        
        
        ctx.ws['all_comp_sim%s' % j].rename(columns={'cond'+str(j): 'cond'+str(j)+'_'+str(ctx.num_comp-1)}, inplace=True)
    
        #drop nan rows at this point because you cannot assign a weigted average to rows entirely empty
        ctx.ws['all_comp_sim%s' % j] = ctx.ws['all_comp_sim%s' % j].drop_duplicates()
        ctx.ws['all_comp_sim%s' % j] = ctx.ws['all_comp_sim%s' % j].dropna(subset=ctx.ws['all_comp_sim%s' % j].columns.difference(['date']),how = 'all')    
    
        ##calculate aggregation condition based on weights for each simulation
        #create list of weigths
        weight_list = ctx.comp_att_df['weight'].tolist()

        base_conditions_sim = ctx.ws['all_comp_sim%s' % j].drop(['date'], axis = 1).astype('float64')
    
        #calculate mean
        ctx.ws['all_comp_sim%s' % j]['cond_avg'] = base_conditions_sim.mean(axis=1)
    
        #update weight_List where comp are inactive
        weight_avg_list = []
        for idx, row in base_conditions_sim.iterrows():
            row_values = row.values
            nan_mask = np.isnan(row_values)
            new_sum = 1 - sum([weight_list[i] if nan_mask[i] else 0.0 for i in range(len(row_values))])
            updated_weight_list = [weight_list[i]/new_sum if not nan_mask[i] else 0.0 for i in range(len(row_values))]
            weighted_avg = np.average(np.nan_to_num(row_values), weights = updated_weight_list)
            weight_avg_list.append(weighted_avg)

        ctx.ws['all_comp_sim%s' % j]['cond_weighted_avg'] = weight_avg_list
        #ctx.ws['all_comp_sim%s' % j]['cond_weighted_avg'] = base_conditions_sim.mul(weight_list).sum(axis=1)

    #CONDITION
    #combine weighted averages

    all_comp_sims = all_comp_sim0[['date','cond_weighted_avg']]

    for j in range(1,ctx.simulations):
        all_comp_sims = pd.merge(all_comp_sims, ctx.ws['all_comp_sim%s' % j][['date','cond_weighted_avg']], how='inner', on='date', suffixes=('_'+str(j-1),'_'+str(j)))   

        #identify indexes with duplicate dates
        dup_df = all_comp_sims['date'][all_comp_sims['date'].duplicated(keep=False)]
        dup_df = dup_df.groupby(list(dup_df)).apply(lambda x: list(x.index))
        dup_df_df = pd.DataFrame(dup_df, columns=['date']).rename(columns={'date':'ind'})
        #filter duplicates of more than 2
        dup_df_df['len'] = dup_df_df.apply(lambda x: len(x['ind']), axis =1)
        dup_df_df = dup_df_df[dup_df_df['len'] > 2]
        #keep only first and last of duplicates
        if dup_df_df.empty == False:
            dup_df_df['drop'] = dup_df_df.apply(lambda x: x['ind'][1:-1], axis=1)
            #remove specified duplicates from original dataset
            drop_list = [i for s in dup_df_df['drop'] for i in s]
            all_comp_sims.drop(drop_list, axis = 0, inplace = True) 
    
    all_comp_sims.rename(columns={"cond_weighted_avg": "cond_weighted_avg_"+str(ctx.simulations-1)}, inplace=True)

    #create datasets with without aggregating over ctx.simulations 

    left_m_df = pd.DataFrame(left_m['year_month'], columns = ['year_month'])
    left_m_df = left_m_df.reset_index()
    left_m_df = left_m_df.rename(columns={'index':'rows'})

    for i in range(ctx.num_comp):

        ctx.ws['yrly_comp%s' % i + '_xr'] = xr.DataArray(ctx.ws['yrly_comp%s' % i + '_np'], coords={'attributes':['year','component', 'mtbf', 'mttr', 'downtime', 'cost', 'events', 'failures', 'replacements', 'repairs', 'fail_dt'],'ctx.simulations':list(range(0,ctx.simulations)) }, dims=['rows','attributes','ctx.simulations'], name='comp_xr')
        ctx.ws['yrly_comp%s' % i + '_sim_df'] = ctx.ws['yrly_comp%s' % i + '_xr'].to_dataframe(name='comp_xr', dim_order=['ctx.simulations','rows','attributes']).unstack()
        ctx.ws['yrly_comp%s' % i + '_sim_df'].columns = ctx.ws['yrly_comp%s' % i + '_sim_df'].columns.get_level_values(1)
        ctx.ws['yrly_comp%s' % i + '_sim_df'] = ctx.ws['yrly_comp%s' % i + '_sim_df'].reset_index()
        ctx.ws['yrly_comp%s' % i + '_sim_df'] = ctx.ws['yrly_comp%s' % i + '_sim_df'].drop(['rows'], axis = 1)
        ctx.ws['yrly_comp%s' % i + '_sim_df'] = ctx.ws['yrly_comp%s' % i + '_sim_df'].astype({'year':'int'})

        #join component and subcomponent descritpions
        string, substring = comp_num_to_string(ctx, i)
        ctx.ws['yrly_comp%s' % i + '_sim_df'].insert(3,'component_desc',string)
        ctx.ws['yrly_comp%s' % i + '_sim_df'].insert(4,'subcomponent_desc',substring)

 
        ctx.ws['mthly_comp%s' % i + '_xr'] = xr.DataArray(ctx.ws['mthly_comp%s' % i + '_np'], coords={'attributes':['component', 'mtbf', 'mttr', 'downtime', 'cost', 'events', 'failures', 'replacements', 'repairs', 'fail_dt'],'ctx.simulations':list(range(0,ctx.simulations)) }, dims=['rows','attributes','ctx.simulations'], name='comp_xr')
        ctx.ws['mthly_comp%s' % i + '_sim_df'] = ctx.ws['mthly_comp%s' % i + '_xr'].to_dataframe(name='comp_xr', dim_order=['ctx.simulations','rows','attributes']).unstack()
        ctx.ws['mthly_comp%s' % i + '_sim_df'].columns = ctx.ws['mthly_comp%s' % i + '_sim_df'].columns.get_level_values(1)
        ctx.ws['mthly_comp%s' % i + '_sim_df'] = ctx.ws['mthly_comp%s' % i + '_sim_df'].reset_index()

        ctx.ws['mthly_comp%s' % i + '_sim_df'] = pd.merge(ctx.ws['mthly_comp%s' % i + '_sim_df'],left_m_df, how = 'left', on = 'rows')
        ctx.ws['mthly_comp%s' % i + '_sim_df'] = ctx.ws['mthly_comp%s' % i + '_sim_df'].drop(['rows'], axis = 1)
        ctx.ws['mthly_comp%s' % i + '_sim_df'] = ctx.ws['mthly_comp%s' % i + '_sim_df'][['ctx.simulations', 'year_month','component', 'mtbf', 'mttr', 'downtime', 'cost', 'events', 'failures', 'replacements', 'repairs', 'fail_dt']]

        #join component and subcomponent descritpions
        string, substring = comp_num_to_string(ctx, i)
        ctx.ws['mthly_comp%s' % i + '_sim_df'].insert(3,'component_desc',string)
        ctx.ws['mthly_comp%s' % i + '_sim_df'].insert(4,'subcomponent_desc',substring)


    #combine components into one dataset

    yrly_all_comp_sim_df = pd.DataFrame()
    mthly_all_comp_sim_df = pd.DataFrame()

    for i in range(ctx.num_comp):
        yrly_all_comp_sim_df = pd.concat([yrly_all_comp_sim_df,ctx.ws['yrly_comp%s' % i + '_sim_df']],ignore_index=True, axis =0)
        mthly_all_comp_sim_df = pd.concat([mthly_all_comp_sim_df,ctx.ws['mthly_comp%s' % i + '_sim_df']],ignore_index=True, axis =0)

    #aggregate over subcomponent

    yrly_sub_comp_sim_df = yrly_all_comp_sim_df.groupby(['ctx.simulations','component_desc','subcomponent_desc','year'], as_index=False).ctx.agg(mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('downtime','sum'), cost=('cost','sum'), events=('events','sum'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))
    yrly_sub_comp_sim_df['availability'] = 1 - (yrly_sub_comp_sim_df['downtime']/365)
    yrly_sub_comp_sim_df['fail_avail'] = 1 - (yrly_sub_comp_sim_df['fail_dt']/365)

    mthly_sub_comp_sim_df = mthly_all_comp_sim_df.groupby(['ctx.simulations','component_desc','subcomponent_desc','year_month'], as_index=False).ctx.agg(mtbf=('mtbf','mean'), mttr=('mttr','mean'), downtime=('downtime','sum'), cost=('cost','sum'), events=('events','sum'), failures=('failures','sum'), replacements=('replacements','sum'), repairs=('repairs','sum'), fail_dt=('fail_dt','sum'))
    mthly_sub_comp_sim_df['availability'] = 1 - (mthly_sub_comp_sim_df['downtime']/(365/12))
    mthly_sub_comp_sim_df['fail_avail'] = 1 - (mthly_sub_comp_sim_df['fail_dt']/(365/12))


    # -----------------------------
    # Package results consistently
    # -----------------------------
    def _pick(name: str):
        # prefer locals, fallback to workspace
        return locals().get(name) if name in locals() else ctx.ws.get(name)

    outputs_tables = {
        "yearly_component": _pick("yrly_comp_df"),
        "monthly_component": _pick("mthly_comp_df"),
        "yearly_subcomponent": _pick("yrly_sub_comp_df"),
        "monthly_subcomponent": _pick("mthly_sub_comp_df"),
        "yearly_simulations": _pick("yrly_all_comp_sim_df"),
        "monthly_simulations": _pick("mthly_all_comp_sim_df"),
        # Some legacy code builds per-sim aggregated subcomponent tables:
        # keep them if present
        "yearly_subcomponent_sim": _pick("yrly_sub_comp_sim_df"),
        "monthly_subcomponent_sim": _pick("mthly_sub_comp_sim_df"),
    }
    # keep only DataFrames
    outputs_tables = {k: v for k, v in outputs_tables.items() if isinstance(v, pd.DataFrame)}

    # ---- Condition outputs (saved under key 'condition_data' for the agent pipeline)
    percentiles = {}
    for p in [
        "min","10th_perc","20th_perc","30th_perc","40th_perc","50th_perc",
        "60th_perc","70th_perc","80th_perc","90th_perc","max",
    ]:
        v = ctx.ws.get(f"all_comp_cond_df_{p}")
        if isinstance(v, pd.DataFrame):
            percentiles[p] = v

    # component_conditions can be huge; include only if present
    component_conditions = {}
    for k, v in ctx.ws.items():
        if isinstance(v, pd.DataFrame) and k.endswith("_cond"):
            component_conditions[k] = v

    condition_data = {
        "all_components": ctx.ws.get("all_comp_cond_df"),
        "simulations": ctx.ws.get("all_comp_sims"),
        "percentiles": percentiles,
        "component_conditions": component_conditions,
    }
    # Drop non-dataframes to keep archiver simple
    if not isinstance(condition_data.get("all_components"), pd.DataFrame):
        condition_data.pop("all_components", None)
    if not isinstance(condition_data.get("simulations"), pd.DataFrame):
        condition_data.pop("simulations", None)

    parameters = {
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "simulations": ctx.simulations,
        "aggregation": ctx.agg,
        "opportunistic_downtime": ctx.opp_dt_ind,
        "spare_systems": ctx.spare_ind,
    }

    # Flatten contract for downstream tools
    results = {
        **outputs_tables,
        "condition_data": condition_data,
        "parameters": parameters,
    }
    return results


if __name__ == "__main__":
    # Manual smoke test (safe; only runs when executed directly)
    res = run_ram_simulation(
        input_xlsx=r"outputs\ram_input_sheet.xlsx",
        start_date=date(2011, 1, 1),
        end_date=date(2017, 12, 31),
        simulations=2,
        agg="50th_perc",
        opp_dt_ind=0,
        spare_ind=0,
    )
    print("Simulation complete. Keys:", list(res.keys()))
