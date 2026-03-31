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
met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))

## Use glycerolipid dataset provided by Nina with duplicates 
glycero_df <- read.csv(here("Cleaned data CSVs", "glycerolipids_fa_TyCOTW.csv"))

### make volcano plots + enrichment analyses specifically for TAG DAG MADAG in glycero df
### look at top metabolites - what are they? 
### remove duplicates first then look at info in dataframe
