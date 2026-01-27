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
feature_importance_comparison_all <- read.csv("/work/hs325/World_Corals/machine_learning/all_mets/featureimportanceallmets.csv")

feature_importance_comparison_all <- feature_importance_comparison_all %>%
  dplyr::rename(metabolite = Feature)
merged_df_all <- feature_importance_comparison_all %>%
  inner_join(met_df, by = "metabolite")


################################################################################

target_classes <- trimws(c(
  "Diacylglycerols", "Fatty amides", "Fatty esters", "Glycerolipids", 
  "Glycerophospholipids", "Monoalkyldiacylglycerols", "Phosphatidylglycerocholines", 
  "Sphingolipids", "Steroids", "Triacylglycerols", "Unknown"))

spec_colors <- c("#FFBB78FF", "#D62728FF", 
                 "#9467BDFF", "#8C564BFF", "#E377C2FF", "#BCBD22FF", 
                 "#17BECFFF", "#2CA02CFF", "#FF9896FF", "#98DF8AFF", "#1F77B4FF")

names(spec_colors) <- target_classes
final_palette <- c(spec_colors, "Other" = "#D3D3D3")

process_importance_data <- function(df, importance_col) {
  df %>%
    mutate(compound_superclass = trimws(as.character(compound_superclass))) %>%
    mutate(display_class = if_else(compound_superclass %in% names(final_palette), 
                                   compound_superclass, 
                                   "Other")) %>%
    mutate(display_class = fct_relevel(factor(display_class), "Other", after = Inf)) %>%
    mutate(metabolite = fct_reorder(metabolite, !!sym(importance_col), .desc = TRUE))
}

xgb_plot_df <- process_importance_data(merged_df_all[1:40,], "XGBoost_Importance")
rf_plot_df  <- process_importance_data(merged_df_all[1:120,], "RandomForest_Importance")
plot_df_all <- process_importance_data(merged_df_all, "XGBoost_Importance")

ordered_levels <- c(target_classes, "Other")

xgb_plot_df$display_class <- factor(xgb_plot_df$display_class, levels = ordered_levels)
rf_plot_df$display_class  <- factor(rf_plot_df$display_class,  levels = ordered_levels)
plot_df_all$display_class <- factor(plot_df_all$display_class, levels = ordered_levels)

#################### make plots ###########################

#xgb importance
p1 <- ggbarplot(xgb_plot_df, x = "metabolite", y = "XGBoost_Importance",
                fill = "display_class", palette = final_palette, color = "transparent",
                xlab = "Metabolite", ylab = "XGBoost Importance") +
  theme_pubr() +
  #  how to remove spaces between or add spaces between axes and plot!
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) + 
  scale_x_discrete(expand = expansion(add = c(1, 0.5))) + 
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(),
        legend.position = "none")

#rf importance
p2 <- ggbarplot(rf_plot_df, x = "metabolite", y = "RandomForest_Importance",
                fill = "display_class", palette = final_palette, color = "transparent",
                xlab = "Metabolite", ylab = "RF Importance") +
  theme_pubr() +
  scale_y_continuous(expand = expansion(mult = c(0, 0.1))) +
  scale_x_discrete(expand = expansion(add = c(1, 0.5))) +
  theme(axis.text.x = element_blank(), 
        axis.ticks.x = element_blank(),
        legend.position = "none")


# for p3 - only label top text
ordered_levels <- c(target_classes, "Other")
top <- plot_df_all %>%
  mutate(dist = sqrt(XGBoost_Importance^2 + RandomForest_Importance^2)) %>%
  arrange(desc(dist)) %>%
  slice_head(n = 5) %>%
  pull(metabolite)

active_classes <- ordered_levels[ordered_levels %in% unique(c(
  as.character(xgb_plot_df$display_class), 
  as.character(rf_plot_df$display_class)
))]

origin_shapes <- c("Host" = 16, "Symbiont" = 3, "Both" = 17, "Unknown" = 8)
p3 <- ggscatter(plot_df_all, 
                x = "XGBoost_Importance", 
                y = "RandomForest_Importance",
                color = "display_class", 
                shape = "refined_origin", 
                palette = final_palette,
                label = "metabolite", 
                label.select = top,
                repel = TRUE,                     
                font.label = c(10, "italic"),      
                cor.coeff = TRUE, 
                cor.method = "pearson",
                xlab = "XGBoost Feature Importance", 
                ylab = "RF Feature Importance") +
                scale_shape_manual(values = origin_shapes) +
  theme_pubr() +
  theme(legend.position = "none")
  
shared_legend <- get_legend(
    p3 + 
      scale_color_manual(
        values = final_palette, 
        breaks = active_classes, 
        name = "Compound Superclass"
      ) +
      scale_shape_manual(
        values = origin_shapes,
        name = "Metabolite Origin"
      ) +
      theme(legend.position = "right", 
            legend.text = element_text(size = 8),
            legend.title = element_text(size = 12)) +
      guides(
        color = guide_legend(
          ncol = 2, 
          order = 1,
          override.aes = list(shape = 15, size = 5) 
        ), 
        shape = guide_legend(
          ncol = 2, 
          order = 2,
          override.aes = list(size = 4)
        )
      )
  )

#make figure
top_row <- plot_grid(p1, p2, labels = c("A", "B"), ncol = 2)
bottom_row <- plot_grid(p3, shared_legend, labels = c("C", ""), 
                        ncol = 2, rel_widths = c(1.5, 1)) 

final_fig <- plot_grid(top_row, bottom_row, ncol = 1)
print(final_fig)
ggsave("/work/hs325/World_Corals/misc/figs/fip_all.jpg", final_fig, width = 12, height = 6, unit = "in")

