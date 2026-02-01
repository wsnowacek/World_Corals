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

cols_bleaching <- c(
  "Bleached" = "#019875FF", 
  "Non-Bleached" = "#FF847CFF", 
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
    host_family = factor(host_family)
  )
# color palettes
cols_location  <- c("#449DB3FF", "#A3BAC2FF", "#60BFAEFF", "#8C6E5DFF")
cols_symbiont  <- c("#D84D16FF", "#FFF800FF", "#8FDA04FF")
cols_sclero    <- c("1" = "#DE7862FF", "0" = "#D8AF39FF")

## for metabolites TBA later
# refined origin (4 levels) -> host, symbiont, both, unknown
# compound superclass (10? levels)
# NPC classifier pathway 7 levels
# TBD coral compound family? 9 levels 

permanova_numeric_data <- df %>% select(starts_with("x"))
keep_rows <- !is.na(df$host_family) & 
  complete.cases(permanova_numeric_data)

permanova_numeric_data2 <- permanova_numeric_data[keep_rows, , drop = FALSE]
meta2 <- df[keep_rows, , drop = FALSE]

################################################################################
#Scleractinia vs nonScleractinia PERMANOVA

# compute BC dissimilarity
bray_curtis_scleractinia <- vegdist(permanova_numeric_data2, method = "bray")
scleractinia_permanova_result <- adonis2(
  bray_curtis_scleractinia ~ scleractinia, 
  data = meta2, 
  permutations = 999
)
print(scleractinia_permanova_result)
# adonis2(formula = bray_curtis_scleractinia ~ scleractinia, data = meta2, permutations = 999)
#           Df SumOfSqs      R2      F Pr(>F)    
# Model      1   13.447 0.09373 55.846  0.001 ***
# Residual 540  130.028 0.90627                  
# Total    541  143.476 1.00000        

################ plot
pcoa_result <- cmdscale(bray_curtis_scleractinia, eig = TRUE, k = 2)
pcoa_points <- as.data.frame(pcoa_result$points)
colnames(pcoa_points) <- c("PCoA1", "PCoA2")
pcoa_points <- bind_cols(pcoa_points, meta2)

# compute percent variance explained (use only positive eigenvalues)
pos_eig <- pcoa_result$eig[pcoa_result$eig > 0]
var_explained <- round(100 * pcoa_result$eig[1:2] / sum(pos_eig), 1)
# 18.8 14.4

p <- ggplot(pcoa_points, aes(x = PCoA1, y = PCoA2, color = scleractinia, fill = scleractinia)) +
  geom_point(size = 3, alpha = 0.8) +
  stat_ellipse(
    geom = "polygon",
    alpha = 0.15,
    level = 0.95,
    type = "t",
    colour = NA
  ) +
  scale_color_manual(values = cols_sclero, labels = c("1" = "Scleractinia", "0" = "Non-Scleractinia")) +
  scale_fill_manual(values = cols_sclero, labels = c("1" = "Scleractinia", "0" = "Non-Scleractinia")) +
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Order",
    fill = "Order"
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "right")
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_scler.jpg", p)

################################################################################
#Location + bleaching status PERMANOVA

bray_curtis_locB <- vegdist(permanova_numeric_data2, method = "bray")
locB_permanova_result <- adonis2(
  bray_curtis_locB ~ location / bleaching, 
  data = meta2, 
  permutations = 999
)
locB_permanova_result
#           Df SumOfSqs      R2      F Pr(>F)    
# Model      8   51.266 0.35731 37.041  0.001 ***
#   Residual 533   92.210 0.64269                  
# Total    541  143.476 1.00000                  

shapes_bleaching <- c(
  "Bleached" = 4,       # Cross
  "Non-Bleached" = 17,   # Triangle
  "Not Applicable" = 1   # Open Circle
)

pcoa_result2 <- cmdscale(bray_curtis_locB, eig = TRUE, k = 2)
pcoa_points2 <- as.data.frame(pcoa_result2$points)
colnames(pcoa_points2) <- c("PCoA1", "PCoA2")
pcoa_points2 <- bind_cols(pcoa_points2, meta2)

# compute percent variance explained (use only positive eigenvalues)
pos_eig <- pcoa_result2$eig[pcoa_result2$eig > 0]
var_explained <- round(100 * pcoa_result2$eig[1:2] / sum(pos_eig), 1)

p2 <- ggplot(pcoa_points2, aes(x = PCoA1, y = PCoA2, color = location)) +
  # Ellipses only care about location
  stat_ellipse(
    aes(fill = location, group = location),
    geom = "polygon",
    alpha = 0.1,
    level = 0.95,
    colour = NA
  ) +
  # Move shape and bleaching mapping here so it doesn't interfere with the ellipse
  geom_point(aes(shape = bleaching), size = 3, alpha = 0.8) +
  scale_color_manual(values = cols_location) +
  scale_fill_manual(values = cols_location) +
  scale_shape_manual(values = shapes_bleaching) + 
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Location",
    fill = "Location",
    shape = "Status"
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "right")
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_locb.jpg", p2, width = 8, height = 6, dpi = 300)

################################################################################
## Location by symbiont potential PERMANOVA

################################################################################
## Bleaching status alone PERMANOVA

################################################################################
## Location alone PERMANOVA


################################################################################
################################################################################
# Box plots

################################################################################
################################################################################
# Dendrograms



