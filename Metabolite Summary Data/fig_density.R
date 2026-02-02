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

## for metabolites colormap TBA later
# compound superclass (10? levels)
# NPC classifier pathway 7 levels
# TBD coral compound family? 9 levels 
