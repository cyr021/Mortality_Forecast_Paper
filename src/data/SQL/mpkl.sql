-- ------------------------------------------------------------------
-- Title: Experimental group: Retrieves the clinical data for the first 72 hours
-- Notes: this query does not specify a schema. To run it on your local
-- MIMIC schema, run the following command:
--  SET SEARCH_PATH TO mimiciii;
-- Where "mimiciii" is the name of your schema, and may be different.
-- ------------------------------------------------------------------


DROP MATERIALIZED VIEW IF EXISTS MPKL CASCADE;
CREATE MATERIALIZED VIEW MPKL as
with cpap as
(
  select ie.icustay_id
    , min(charttime - interval '1' hour) as starttime
    , max(charttime + interval '4' hour) as endtime
    , max(case when lower(ce.value) similar to '%(cpap mask|bipap mask)%' then 1 else 0 end) as cpap
    , case 
        when ce.charttime BETWEEN (ie.intime - interval '6' hour) AND (ie.intime + interval '1' day) Then '1'
        when ce.charttime BETWEEN (ie.intime + interval '1' day) AND (ie.intime + interval '2' day) Then '2'
        when ce.charttime BETWEEN (ie.intime + interval '2' day) AND (ie.intime + interval '3' day) Then '3'
        ELSE null
      end as t_day
  from icustays ie
  inner join chartevents ce
    on ie.icustay_id = ce.icustay_id
  where itemid in
  (
    -- TODO: when metavision data import fixed, check the values in 226732 match the value clause below
    467, 469, 226732
  )
  and lower(ce.value) similar to '%(cpap mask|bipap mask)%'
  -- exclude rows marked as error
  AND ce.error IS DISTINCT FROM 1
  group by ie.icustay_id, t_day
)
-- extract a flag for surgical service
-- this combined with "elective" from admissions table defines elective/non-elective surgery
, surgflag as
(
  select adm.hadm_id
    , case when lower(curr_service) like '%surg%' then 1 else 0 end as surgical
    , ROW_NUMBER() over
    (
      PARTITION BY adm.HADM_ID
      ORDER BY TRANSFERTIME
    ) as serviceOrder
  from admissions adm
  left join services se
    on adm.hadm_id = se.hadm_id
)
-- icd-9 diagnostic codes are our best source for comorbidity information
-- unfortunately, they are technically a-causal
-- however, this shouldn't matter too much for the SAPS II comorbidities
, comorb as
(
select hadm_id
-- these are slightly different than elixhauser comorbidities, but based on them
-- they include some non-comorbid ICD-9 codes (e.g. 20302, relapse of multiple myeloma)
  , max(CASE
    when icd9_code between '042  ' and '0449 ' then 1
  		end) as AIDS      /* HIV and AIDS */
  , max(CASE
    when icd9_code between '20000' and '20238' then 1 -- lymphoma
    when icd9_code between '20240' and '20248' then 1 -- leukemia
    when icd9_code between '20250' and '20302' then 1 -- lymphoma
    when icd9_code between '20310' and '20312' then 1 -- leukemia
    when icd9_code between '20302' and '20382' then 1 -- lymphoma
    when icd9_code between '20400' and '20522' then 1 -- chronic leukemia
    when icd9_code between '20580' and '20702' then 1 -- other myeloid leukemia
    when icd9_code between '20720' and '20892' then 1 -- other myeloid leukemia
    when icd9_code = '2386 ' then 1 -- lymphoma
    when icd9_code = '2733 ' then 1 -- lymphoma
  		end) as HEM
  , max(CASE
    when icd9_code between '1960 ' and '1991 ' then 1
    when icd9_code between '20970' and '20975' then 1
    when icd9_code = '20979' then 1
    when icd9_code = '78951' then 1
  		end) as METS      /* Metastatic cancer */
  from
  (
    select hadm_id, seq_num
    , cast(icd9_code as char(5)) as icd9_code
    from diagnoses_icd
  ) icd
  group by hadm_id
)
, pafi1 as
(
  -- join blood gas to ventilation durations to determine if patient was vent
  -- also join to cpap table for the same purpose
  select bg.icustay_id, bg.charttime, bg.t_day
  , PaO2FiO2
  , case when vd.icustay_id is not null then 1 else 0 end as vent
  , case when cp.icustay_id is not null then 1 else 0 end as cpap
  from bloodgasfirst3dayarterial bg
  left join ventdurations vd
    on bg.icustay_id = vd.icustay_id
    and bg.charttime >= vd.starttime
    and bg.charttime <= vd.endtime
  left join cpap cp
    on bg.icustay_id = cp.icustay_id
    and bg.charttime >= cp.starttime
    and bg.charttime <= cp.endtime
)
, pafi2 as
(
  -- get the minimum PaO2/FiO2 ratio *only for ventilated/cpap patients*
  select icustay_id, t_day
  , min(PaO2FiO2) as PaO2FiO2_vent_min
  from pafi1
  where vent = 1 or cpap = 1
  group by icustay_id, t_day
)
select ie.subject_id, ie.hadm_id, ie.icustay_id, ie.t_day
      , ie.intime
      , ie.outtime

      -- the casts ensure the result is numeric.. we could equally extract EPOCH from the interval
      -- however this code works in Oracle and Postgres
      , round( ( cast(ie.intime as date) - cast(pat.dob as date) ) / 365.242 , 2 ) as age

      , vital.heartrate_max
      , vital.heartrate_min
      , vital.sysbp_max
      , vital.sysbp_min
      , vital.tempc_max
      , vital.tempc_min

      -- this value is non-null iff the patient is on vent/cpap
      , pf.PaO2FiO2_vent_min

      , uo.urineoutput

      , labs.bun_min
      , labs.bun_max
      , labs.wbc_min
      , labs.wbc_max
      , labs.potassium_min
      , labs.potassium_max
      , labs.sodium_min
      , labs.sodium_max
      , labs.bicarbonate_min
      , labs.bicarbonate_max
      , labs.bilirubin_min
      , labs.bilirubin_max

      , gcs.mingcs

      , comorb.AIDS
      , comorb.HEM
      , comorb.METS

      , case
          when adm.ADMISSION_TYPE = 'ELECTIVE' and sf.surgical = 1
            then 'ScheduledSurgical'
          when adm.ADMISSION_TYPE != 'ELECTIVE' and sf.surgical = 1
            then 'UnscheduledSurgical'
          else 'Medical'
        end as AdmissionType

from (select *, '1' as t_day from icustays 
  union all select *, '2' as  t_day from icustays 
  union all select *, '3' as t_day from icustays
  order by subject_id, hadm_id, icustay_id
  ) ie
inner join admissions adm
  on ie.hadm_id = adm.hadm_id
inner join patients pat
  on ie.subject_id = pat.subject_id

-- join to above views
left join pafi2 pf
  on ie.icustay_id = pf.icustay_id and ie.t_day = pf.t_day
left join surgflag sf
  on adm.hadm_id = sf.hadm_id and sf.serviceOrder = 1
left join comorb
  on ie.hadm_id = comorb.hadm_id

-- join to custom tables to get more data....
left join gcsfirst3day gcs
  on ie.icustay_id = gcs.icustay_id and ie.t_day = gcs.t_day
left join vitalsfirst3day vital
  on ie.icustay_id = vital.icustay_id and ie.t_day = vital.t_day
left join uofirst3day uo
  on ie.icustay_id = uo.icustay_id and ie.t_day = uo.t_day
left join labsfirst3day labs
  on ie.icustay_id = labs.icustay_id and ie.t_day = labs.t_day
;

 