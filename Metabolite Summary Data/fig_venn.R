library(tidyverse)
library(knitr)
library(readxl)
library(data.table)
library(vegan)
library(scales)
library(ggraph)
library(cowplot)
library(ggdendro)
library(ggridges)
library(dendextend)
library(RColorBrewer)
library(ggpubr)
library(forcats)
library(ggvenn)

setwd("/work/hs325/World_Corals/Metabolite Summary Data")
df<- read.csv("qc_data.csv")
met_df<- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_clean.csv")

cols_bleaching <- c(
  "Bleached" = "#FF847CFF", 
  "Non-Bleached" = "#019875FF", 
  "Not Applicable" = "#D3D3D3")

df <- df %>%
  mutate(
    bleaching = case_when(
      bleaching == "B"  ~ "Bleached",
      bleaching == "NB" ~ "Non-Bleached",
      is.na(bleaching)  ~ "Not Applicable",
      TRUE              ~ as.character(bleaching)
    ),
    bleaching = factor(bleaching, levels = c("Bleached", "Non-Bleached", "Not Applicable")),
    
    scleractinia = if_else(host_order == "Scleractinia", "1", "0"),
    scleractinia = factor(scleractinia, levels = c("1", "0")),
    location = factor(location),
    symbiont.potential = factor(symbiont.potential),
    host_order = fct_relevel(factor(host_order), "Scleractinia"),
    host_family = factor(host_family),
    host_phylum = factor(host_phylum)
  )
# color palettes
cols_location <-c("#002594FF", "#E0B2CDFF", "#54C4E3FF", "#F3AA4FFF")
# cols_location  <- c("#449DB3FF", "#A3BAC2FF", "#60BFAEFF", "#8C6E5DFF")
cols_symbiont  <- c("#D84D16FF", "#FFF800FF", "#8FDA04FF")
cols_phylum <- c("#24492EFF", "#015B58FF", "#2C6184FF", "#59629BFF", "#89689DFF", "#BA7999FF", "#E69B99FF")
cols_sclero    <- c("1" = "#DE7862FF", "0" = "#D8AF39FF")


feature_importance_comparison_all <- read.csv("/work/hs325/World_Corals/machine_learning/all_mets/featureimportanceallmets.csv")

feature_importance_comparison_all <- feature_importance_comparison_all %>%
  dplyr::rename(metabolite = Feature)

present_metabolites <- df %>% 
  select(starts_with("x")) %>% 
  colnames()

met_df <- met_df %>%
  filter(met_df$metabolite %in% present_metabolites)

met_df <- feature_importance_comparison_all %>%
  inner_join(met_df, by = "metabolite")

################################################################################
## categories for venns

met_df_filtered <- met_df %>% 
  filter(refined_origin != "Unknown")

get_venn_list <- function(data) {
  list(
    Host = data %>% filter(refined_origin %in% c("Host", "Both")) %>% pull(metabolite),
    Symbiont = data %>% filter(refined_origin %in% c("Symbiont", "Both")) %>% pull(metabolite)
  )
}

# Define the subsets
list_all <- get_venn_list(met_df_filtered)
list_xgb <- get_venn_list(met_df_filtered %>% arrange(desc(XGBoost_Importance)) %>% head(43))
list_rf  <- get_venn_list(met_df_filtered %>% arrange(desc(RandomForest_Importance)) %>% head(1541))

### plot
cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")
venn_fill <- c(cols_origin["Host"], cols_origin["Symbiont"])

p_a <- ggvenn(list_all, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))

#xgb 50
p_b <- ggvenn(list_xgb, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))

#Rf 500
p_c <- ggvenn(list_rf, fill_color = venn_fill, stroke_size = 0.5, set_name_size = 4) +
  labs() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))

venn_grid <- plot_grid(p_a, p_b, p_c, ncol = 3, labels = c("A", "B", "C"), label_size = 20)
print(venn_grid)
ggsave("/work/hs325/World_Corals/misc/figs/venn.jpg", 
       venn_grid, width = 15, height = 5, dpi = 300)