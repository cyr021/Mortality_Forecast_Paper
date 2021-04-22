-- ------------------------------------------------------------------
-- Title: Simplified Acute Physiology Score II (SAPS II)
-- This query extracts the simplified acute physiology score II.
-- This score is a measure of patient severity of illness.
-- The score is calculated on the first day of each ICU patients' stay.
-- ------------------------------------------------------------------

-- Reference for SAPS II:
--    Le Gall, Jean-Roger, Stanley Lemeshow, and Fabienne Saulnier.
--    "A new simplified acute physiology score (SAPS II) based on a European/North American multicenter study."
--    JAMA 270, no. 24 (1993): 2957-2963.

-- Variables used in SAPS II:
--  Age, GCS
--  VITALS: Heart rate, systolic blood pressure, temperature
--  FLAGS: ventilation/cpap
--  IO: urine output
--  LABS: PaO2/FiO2 ratio, blood urea nitrogen, WBC, potassium, sodium, HCO3

select c.subject_id, c.hadm_id, c.icustay_id, c.sapsii, c.sapsii_prob, b.expire_flag        
from (                      
select * from diagnoses_icd 
where icd9_code in ('8080','8081','80843','80844','80853','80854')
) a                                                                
inner join patients b on a.subject_id=b.subject_id
left join sapsii c on a.subject_id = c.subject_id;