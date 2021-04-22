select c.*, b.gender, b.expire_flag        
from (                      
select * from diagnoses_icd 
where icd9_code in ('8080','8081','80843','80844','80853','80854')
) a                                                                
inner join patients b on a.subject_id=b.subject_id
left join mpkl c on a.subject_id = c.subject_id
where age < 100
;