library(tidyverse)
library(data.table)
library(dummy)
library(caret)

load_data <- function(data_path){
  
  df <- fread(data_path, stringsAsFactors = TRUE)  %>% 
    # extract(age, into = 'age', '([\\d]+).*', remove = FALSE) %>% 
    # value = as.numeric(as.character(value)
    select(-age) %>% 
    mutate(
      hadm_id = as.factor(hadm_id),
      itemid = as.factor(itemid),
      dob = as.POSIXct(as.Date(dob)),      
      admittime = as.POSIXct(as.Date(admittime)),
      charttime = as.POSIXct(as.Date(charttime)),
      intime = as.POSIXct(as.Date(intime)),
      age = round((as.numeric((admittime - dob)))/365)#!!!!!!!
    ) %>% 
    filter(!value %in% c('NEG','','NONE'), 
           !is.na(value),
           !itemid %in% c('51508','50919','51493','50827'),
           los >= 3,
           charttime>=intime) %>% 
    as.data.frame()
  
  # optional: only for control group; select out the measure records within 24 hours after entering ICU

  # group_by(.dots = as.vector(colnames(.)[c(1:15,18)])) %>% 
    # filter(charttime == max(charttime)) %>% 
    # ungroup() %>% 
    # group_by(subject_id, hadm_id) %>% 
    # filter(age == max(age))
    
  # df_filter_chart <- df %>% 
  #   select(hadm_id, intime, charttime) %>% 
  #   distinct() %>% 
  #   group_by(hadm_id) %>% 
  #   filter(intime == max(intime)) %>% 
  #   ungroup %>% 
  #   group_by(hadm_id, intime) %>% 
  #   mutate(row = row_number(charttime)) %>% 
  #   arrange(hadm_id, intime, charttime) %>% 
  #   filter(charttime >= intime) %>% 
  #   filter(row <= 3)
  
  df_pp <- df[,c(1:14,20)] %>% 
    distinct()
  
  df_icu <- df %>% 
    dplyr::select(subject_id, hadm_id, intime, los) %>% 
    distinct() %>% 
    group_by(subject_id, hadm_id) %>% 
    summarise(icu_no = n(), 
              icu_me = mean(los)) %>% 
    ungroup() %>% 
    mutate(icu_no = ifelse(is.na(icu_me), 0, icu_no),
           icu_me = ifelse(is.na(icu_me), 0, icu_me))
  
  # filter out chartevents in the first 3 days after being sent to ICU/transformation for numeric and categorical data
  df_small <- df %>% 
    filter(hadm_id=='180029')
  
  df_itemid <- df %>% 
    dplyr::select(hadm_id, intime, charttime, itemid, value) %>% 
    distinct() %>% 
    mutate(hadm_id = as.factor(hadm_id), itemid = as.factor(itemid)) %>% 
    group_by(hadm_id, itemid) %>% 
    mutate(row = row_number(charttime)) %>% 
    arrange(hadm_id, itemid, charttime) %>% 
    filter(row <= 1) %>% 
    ungroup %>% 
    select(-intime, -charttime, -row) %>% 
    mutate(valuenum = as.numeric(as.character(value))) %>% 
    mutate(value = ifelse(!is.na(valuenum), round(valuenum,1), value)) %>% 
    select(-valuenum) %>% 
    group_by(hadm_id, itemid) %>%
    summarise(itemid_min = median(value)) %>% 
    gather(variable, value, -(hadm_id:itemid)) %>%
    unite(temp, itemid, variable) %>%
    group_by(hadm_id, temp) %>%
    mutate(id=1:n()) %>%
    spread(temp, value) %>%
    select(-id)
    
  # df_raw %>% filter(los>=3) %>% select(hadm_id,expire_flag) %>% distinct() %>% group_by(expire_flag) %>% tally()
  # # A tibble: 2 x 2
  # expire_flag     n
  # <int> <int>
  #   1           0   147
  # 2           1    65
  
  
  return(list(df, df_pp, df_icu, df_itemid))
}