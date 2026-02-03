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

setwd("/work/hs325/World_Corals/Metabolite Summary Data")
df<- read.csv("qc_data.csv")
met_df<- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_clean.csv")

present_metabolites <- df %>% 
  select(starts_with("x")) %>% 
  colnames()

met_df <- met_df %>%
  filter(met_df$metabolite %in% present_metabolites)

unique(met_df$compound_superclass) #58
unique(met_df$npc_number_pathway) #8
unique(met_df$npc_number_superclass) #42
unique(met_df$npc_number_class) #83
unique(met_df$coraldb_compound_superclass) #17
unique(met_df$coraldb_compound_class) #34
unique(met_df$coraldb_compound_family) #10
unique(met_df$gnps_compound_class) #13
unique(met_df$classy_fire_number_most_specific_class) #231

class_counts <- met_df %>%
  group_by(compound_superclass) %>%
  summarise(n = n()) %>%
  arrange(desc(n))
print(class_counts, n = 50)

## met_df_npc_number_pathway

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
cols_symbiont  <- c("#D84D16FF", "#FFF800FF", "#8FDA04FF")
cols_phylum <- c("#24492EFF", "#015B58FF", "#2C6184FF", "#59629BFF", "#89689DFF", "#BA7999FF", "#E69B99FF")
cols_sclero    <- c("1" = "#DE7862FF", "0" = "#D8AF39FF")

#################################################################################

## panel A ubq abundance of all mets dashed line for scleractinia color by compound superclass
## panel B ubq abundance of only mets found in scleractinia color by compound superclass
## panel C D E existing ML fig of XGB VIP RF VIP and correlation with top mets
## panel F bar plot colored by ubiquity
## panel G ubiquity plot showing only most important mets (up to 100 for each model)