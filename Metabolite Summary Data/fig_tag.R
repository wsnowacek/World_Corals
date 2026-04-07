library(tidyverse)
library(knitr)
library(readxl)
library(data.table)
library(vegan)
library(scales)
library(cowplot)
library(RColorBrewer)
library(ggpubr)
library(forcats)
library(ggvenn)
library(ggrepel)
library(ggforce)
library(here)

df <- read.csv(here("Cleaned data CSVs", "qc_data.csv"))

## defining things
met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))
met_df <- met_df %>%
  mutate(
    compound_class = recode(
      compound_class,
      "Carotenoids (C40, Î²-Î²)" = "Carotenoids",
      "Oxidized glycerophospholipids" = "OxPL",
      "Glycerophosphoethanolamines" = "GPEtn",
      "Neutral glycosphingolipids" = "Neutral GSL",
      "Triacylglycerols" = "TAG",
      "Diacylglycerols" = "DAG",
      "Prenyl quinone meroterpenoids" = "TQ/THQs"
    )
  )

### take top 20 compound_class by count, with Unknown forced to end
target_classes <- met_df %>%
  count(compound_class, sort = TRUE) %>%
  slice_head(n = 20) %>%
  pull(compound_class) %>%
  trimws()

target_classes <- c(
  setdiff(target_classes, "Unknown"),
  intersect(target_classes, "Unknown")
)

### colors
provided_hex <- c(
  "#1F77B4FF", "#FF7F0EFF", "#2CA02CFF", "#D62728FF",
  "#9467BDFF", "#8C564BFF", "#E377C2FF", "deepskyblue4", "#BCBD22FF",
  "#17BECFFF", "#AEC7E8FF", "#FFBB78FF", "#98DF8AFF", "#FF9896FF",
  "#C5B0D5FF", "#C49C94FF", "#F7B6D2FF", "#9EDAE5FF", "#DBDB8DFF",
  "#C7C7C7FF"
)

spec_colors <- setNames(provided_hex[seq_along(target_classes)], target_classes)

final_palette <- c(spec_colors, "Other" = "gray30")
class_order <- c(target_classes, "Other")

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)


glycero_df <- read.csv(here("Cleaned data CSVs", "glycerolipids_fa_TyCOTW.csv"))

################################################################################
## datasets to compare

### define 59 df 
met_df_scler_all <- met_df %>%
  filter(scler_ubiquity == 100)

# define 95 %ile df 
met_df_scler_95 <- met_df %>%
  filter(scler_ubiquity >= 95)

# define df of non scler
met_df_nonscler <- met_df %>%
  filter(non_scler_ubiquity >= 95)

# define ML dfs
xgb_df <- met_df %>%
  filter(XGBoost_Importance > 0) %>%
  arrange(desc(XGBoost_Importance)) %>%
  mutate(metabolite = fct_reorder(metabolite, XGBoost_Importance, .desc = TRUE))

rf_df <- met_df %>%
  filter(RandomForest_Importance > 0.001) %>%
  arrange(desc(RandomForest_Importance)) %>%
  mutate(metabolite = fct_reorder(metabolite, RandomForest_Importance, .desc = TRUE))

# define core
df_scler <- df %>% filter(host_order == "Scleractinia", !is.na(host_family))

group_summary_core <- df_scler %>%
  pivot_longer(cols = starts_with("x"), names_to = "metabolite", values_to = "val") %>%
  group_by(host_family, metabolite) %>%
  summarise(present = any(val > 0, na.rm = TRUE), .groups = "drop")

total_families <- length(unique(df_scler$host_family))

core_metabolite_ids <- group_summary_core %>%
  group_by(metabolite) %>%
  summarise(n_families = sum(present)) %>%
  filter(n_families == total_families) %>%
  pull(metabolite)

core_df <- met_df %>%
  filter(metabolite %in% core_metabolite_ids)

# define perm importance df
perm_df <- read.csv(here("machine_learning/perm_importance", "perm_importance_results_kbest.csv"))
important_metabolites_joined <- perm_df %>%
  # Keep only features that contributed positively to the model
  filter(importance_mean > 0) %>%
  left_join(met_df, by = "metabolite")

feature_set_list <- important_metabolites_joined %>%
  group_split(feature_set) %>%
  set_names(map_chr(., ~ first(.x$feature_set)))

all_kbest_df <- feature_set_list[["All_KBest"]] %>%
  select(-X.x)

host_kbest_df <- feature_set_list[["Host_KBest"]] %>%
  select(-X.x)
## x31046_620_59826_13_466 --> perm importance 18.3

################################################################################

## check for overlaps between dfs 
## then look at ones found within glycero_df