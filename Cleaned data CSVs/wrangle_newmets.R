library(tidyverse)

setwd("/work/hs325")
met_df <- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_clean.csv") # 31330 obs of 30 variables 
met_df_nina <- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/Ty_superset_metabolites_March26update.csv") ## 31330 obs of 24 variables

met_plot_df <- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_plot_df.csv") # 16368 obs. of 36 variables
glycerolipids <- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/glycerolipids_fa_TyCOTW.csv") # 3623 obs. of 27 variables
kiran_pc <- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/pc_matches_tysuperset1.csv") # 1941 obs. of 5 variables

## combine met_df_nina with met_df using the feature_id column 
## keep all columns in met_df. resulting df should still have 30 columns, but replace any
## columns with which there are values in met_df_nina with those values. name the resulting dataframe as met_df_merged
replace_from <- function(base_df, update_df, key = "feature_id") {
  stopifnot(key %in% names(base_df), key %in% names(update_df))
  
  overlap <- setdiff(intersect(names(base_df), names(update_df)), key)
  
  joined <- base_df %>%
    left_join(update_df %>% select(all_of(c(key, overlap))),
              by = key,
              suffix = c("", ".upd"))
  
  for (nm in overlap) {
    upd_nm <- paste0(nm, ".upd")
    joined[[nm]] <- dplyr::coalesce(joined[[upd_nm]], joined[[nm]])
  }
  
  joined %>% select(all_of(names(base_df)))
}

met_df_merged <- replace_from(met_df, met_df_nina, key = "feature_id")


## using met_df_merged, do another merge with met_plot_df using feature_id but keep only the rows in met_plot_df. 
# keep all unique columns in both met_df_merged and met_plot_df and name the resulting dataframe met_plot_df_merged
add_new_cols_only <- function(left_df, right_df, key = "feature_id") {
  stopifnot(key %in% names(left_df), key %in% names(right_df))
  
  right_extra <- right_df %>%
    select(all_of(key), all_of(setdiff(names(right_df), names(left_df))))
  
  left_df %>%
    left_join(right_extra, by = key)
}
met_plot_df_merged <- add_new_cols_only(met_plot_df, met_df_merged, key = "feature_id")

## then, using met_plot_df_merged, make a new dataframe with all rows kept from met_plot_df_merged
## but have all columns with values from glycerolipids and kiran_pc (still join on feature_id) in the final merged dataframe
## if certain rows are missing values for some of the new columns from glycerolipids or kiran_pc, make them NAs

gly_1 <- glycerolipids %>% group_by(feature_id) %>% slice(1) %>% ungroup()
pc_1  <- kiran_pc      %>% group_by(feature_id) %>% slice(1) %>% ungroup()

gly_add <- gly_1 %>%
  select(feature_id,
         setdiff(names(gly_1), names(met_plot_df_merged)))
pc_add <- pc_1 %>%
  select(feature_id,
         setdiff(names(pc_1), names(met_plot_df_merged)))

met_plot_df_merged2 <- met_plot_df_merged %>%
  left_join(gly_add, by = "feature_id") %>%
  left_join(pc_add,  by = "feature_id")

final_merged_df <- met_plot_df_merged2

### Save this dataframe for use for redoing analyses
write.csv(final_merged_df, "/work/hs325/World_Corals/Cleaned data CSVs/merged_met_plot_df.csv")


################################################################################

counts1 <- met_plot_df_merged %>%
  count(feature_id, metabolite, name = "n1")

counts2 <- met_plot_df_merged2 %>%
  count(feature_id, metabolite, name = "n2")

count_diffs <- full_join(counts1, counts2, by = c("feature_id", "metabolite")) %>%
  mutate(n1 = coalesce(n1, 0L),
         n2 = coalesce(n2, 0L)) %>%
  filter(n1 != n2) %>%
  arrange(desc(abs(n2 - n1)))

count_diffs

glycerolipids %>%
  count(feature_id) %>%
  filter(n > 1) %>%
  arrange(desc(n)) %>%
  head(20)

kiran_pc %>%
  count(feature_id) %>%
  filter(n > 1) %>%
  arrange(desc(n)) %>%
  head(20)
