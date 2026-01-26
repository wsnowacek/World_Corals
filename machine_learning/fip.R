library(tidyverse)
library(readr)
library(broom)
library(cowplot)
library(caret)
library(tibble)
library(GGally)
library(stringr)
library(RColorBrewer)
library(forcats)
library(ggpubr)

setwd("/work/hs325/World_Corals")
Corals_clean <- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/richness_qc_clean.csv")
met_df <- read.csv("/work/hs325/World_Corals/Cleaned data CSVs/metabolite_clean.csv")
feature_importance_comparison_coralonly <- read.csv("/work/hs325/World_Corals/machine_learning/coral_mets_only/featureimportancecoralmets.csv")
feature_importance_comparison_all <- read.csv("/work/hs325/World_Corals/machine_learning/all_mets/featureimportanceallmets.csv")

feature_importance_comparison_coralonly <- feature_importance_comparison_coralonly %>%
  dplyr::rename(metabolite = Feature)
merged_df <- feature_importance_comparison_coralonly %>%
  inner_join(met_df, by = "metabolite")

feature_importance_comparison_all <- feature_importance_comparison_all %>%
  dplyr::rename(metabolite = Feature)
merged_df_all <- feature_importance_comparison_all %>%
  inner_join(met_df, by = "metabolite")

## TODO make colors consistent across all figures
## TODO make plot for all metabolites

################################################################################

########## coral only metabolites - xgb
merged_df <- merged_df %>%
  arrange(desc(XGBoost_Importance)) %>%
  mutate(metabolite = fct_reorder(metabolite, XGBoost_Importance, .desc = TRUE))
xgb_importances <- merged_df[1:30,]
xgb_importances$metabolite <- factor(xgb_importances$metabolite,
                                     levels = xgb_importances$metabolite)

p1 <- ggbarplot(xgb_importances, 
          x = "metabolite", 
          y = "XGBoost_Importance",
          fill = "compound_superclass",
          color = "white",             # Line color around bars
          palette = "jco",    
          xlab = "Metabolite",
          ylab = "Feature Importance",
          legend.title = "Compound Superclass") +
  theme_pubr() +
  theme(
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(),
    legend.position = c(0.7, 0.7),          
    legend.background = element_blank(),  
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8),  
    legend.key.size = unit(0.6, "cm")    
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))


########## coral only metabolites - rf
rf_importances <- merged_df %>%
  arrange(desc(RandomForest_Importance)) %>%
  mutate(metabolite = fct_reorder(metabolite, RandomForest_Importance, .desc = TRUE))
rf_importances <- rf_importances[1:100,]

top_10_classes <- rf_importances %>%
  count(compound_superclass, sort = TRUE) %>%
  slice_head(n = 10) %>%
  pull(compound_superclass)
plot_df <- rf_importances %>%
  mutate(display_class = if_else(compound_superclass %in% top_10_classes, 
                                 as.character(compound_superclass), 
                                 "Other")) %>%
  mutate(display_class = fct_relevel(factor(display_class), "Other", after = Inf))
my_pal <- get_palette("jco", 10)
names(my_pal) <- top_10_classes
my_pal["Other"] <- "#D3D3D3" # Light Gray

p2 <- ggbarplot(plot_df, 
          x = "metabolite", 
          y = "RandomForest_Importance",
          fill = "display_class",
          color = "transparent",       # Removing white borders makes the gray look cleaner
          palette = my_pal,    
          xlab = "Metabolite",
          ylab = "Feature Importance",
          legend.title = "Compound Superclass") +
  theme_pubr() +
  theme(
    axis.text.x = element_blank(),  
    axis.ticks.x = element_blank(),
    legend.position = c(0.7, 0.7),          
    legend.background = element_blank(),  
    legend.title = element_text(size = 9, face = "bold"),
    legend.text = element_text(size = 8),  
    legend.key.size = unit(0.4, "cm")    
  ) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1)))


################################################################################

##
p3 <- ggscatter(merged_df, 
                x = "XGBoost_Importance", 
                y = "RandomForest_Importance",
                add = "reg.line",              
                conf.int = TRUE,               
                add.params = list(color = "steelblue", fill = "#D3d3d3"),
                cor.coeff = TRUE,               # Add correlation coefficient (R)
                cor.method = "pearson",         # Or "spearman"
                cor.coeff.args = list(label.x = 0, label.sep = "\n"),
                xlab = "XGBoost",
                ylab = "Random Forest") +
  theme_pubr()




