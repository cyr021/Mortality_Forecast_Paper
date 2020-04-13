rm(list=ls())
source("src/data/data_utils.R")

# load data
data_path = 'data/test_records (2).csv'
df_pp <- load_data(data_path)[[2]]
df_icu <- load_data(data_path)[[3]]
df_itemid <- load_data(data_path)[[4]]

out <- df_pp %>% 
  inner_join(df_icu) %>% 
  inner_join(df_itemid)

write.csv(out, file="data/out1028-control.csv", row.names = FALSE)


# EDA
b <- df_itemid %>% 
  select(hadm_id, itemid) %>% 
  distinct() %>% 
  group_by(itemid) %>% 
  tally() %>% 
  filter(itemid %in% c('211','676','677','3494','223762','226850','226852','227519','224167','227243','220179',
                       43638,45304,43966,44706,43365,43365,43348))
  
  


  