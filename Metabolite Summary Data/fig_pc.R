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

permanova_numeric_data <- df %>% select(starts_with("x"))
keep_rows <- !is.na(df$host_family) & 
  complete.cases(permanova_numeric_data)

permanova_numeric_data2 <- permanova_numeric_data[keep_rows, , drop = FALSE]
bray_curtis <- vegdist(permanova_numeric_data2, method = "bray")

meta2 <- df[keep_rows, , drop = FALSE]

# ################################################################################
# #Scleractinia vs nonScleractinia PERMANOVA

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

perm_p_val <- scleractinia_permanova_result$`Pr(>F)`[1]
perm_r2    <- round(scleractinia_permanova_result$R2[1], 3)
p_label <- if(perm_p_val == 0.001) "p < 0.001" else paste("p =", perm_p_val)
stats_annotation <- paste0("PERMANOVA: R² = ", perm_r2, ", ", p_label)

p <- ggplot(pcoa_points, aes(x = PCoA1, y = PCoA2, color = scleractinia, fill = scleractinia)) +
  geom_point(size = 3, alpha = 0.8) +
  stat_ellipse(
    geom = "polygon",
    alpha = 0.15,
    level = 0.95,
    type = "t",
    colour = NA
  ) +
  annotate(
    "text",
    x = -Inf,  
    y = -Inf, 
    label = stats_annotation,
    hjust = -0.05,
    vjust = -0.6,
    size = 4) +
  scale_color_manual(values = cols_sclero, labels = c("1" = "Scleractinia", "0" = "Other")) +
  scale_fill_manual(values = cols_sclero, labels = c("1" = "Scleractinia", "0" = "Other")) +
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Order",
    fill = "Order"
  ) +
  theme_cowplot(font_size = 18) +
  theme(legend.position = "right")
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_scler.jpg", p, width = 8, height = 8, dpi=300)

################################################################################
#Location + bleaching status PERMANOVA

# drop NAs for bleaching
keep_rows <- complete.cases(permanova_numeric_data2) & 
  !is.na(meta2$location) & 
  !is.na(meta2$bleaching)

# 2. Synchronize the metadata and numeric data
meta_clean <- meta2[keep_rows, ]
numeric_clean <- permanova_numeric_data2[keep_rows, ]

bray_curtis_clean <- vegdist(numeric_clean, method = "bray")
locB_permanova_result <- adonis2(
  bray_curtis_clean ~ location / bleaching, 
  data = meta_clean, 
  permutations = 999
)
print(locB_permanova_result)          
# Df SumOfSqs      R2      F Pr(>F)    
# Model      8   51.264 0.35811 37.101  0.001 ***
#   Residual 532   91.886 0.64189                  
# Total    540  143.150 1.00000                  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

perm_p_val <- locB_permanova_result$`Pr(>F)`[1]
perm_r2    <- round(locB_permanova_result$R2[1], 3)
p_label <- if(perm_p_val == 0.001) "p < 0.001" else paste("p =", perm_p_val)
stats_annotation <- paste0("PERMANOVA: R² = ", perm_r2, ", ", p_label)

shapes_bleaching <- c(
  "Bleached" = 4,       # Cross
  "Non-Bleached" = 17,   # Triangle
  "Not Applicable" = 1   # Open Circle
)

pcoa_result2 <- cmdscale(bray_curtis_clean, eig = TRUE, k = 2)
pcoa_points2 <- as.data.frame(pcoa_result2$points)
colnames(pcoa_points2) <- c("PCoA1", "PCoA2")
pcoa_points2 <- bind_cols(pcoa_points2, meta_clean)

# compute percent variance explained (use only positive eigenvalues)
pos_eig <- pcoa_result2$eig[pcoa_result2$eig > 0]
var_explained <- round(100 * pcoa_result2$eig[1:2] / sum(pos_eig), 1)

p2 <- ggplot(pcoa_points2, aes(x = PCoA1, y = PCoA2, color = location)) +
  geom_point(aes(shape = bleaching), size = 3, alpha = 0.8) +
  scale_color_manual(values = cols_location) +
  scale_fill_manual(values = cols_location) +
  scale_shape_manual(values = shapes_bleaching) + 
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Location",
    fill = "Location",
    shape = "Bleaching Status"
  ) +
  annotate(
    "text",
    x = -Inf,
    y = -Inf,
    label = stats_annotation,
    hjust = -0.05,
    vjust = -0.6,
    size = 4) +
    theme(legend.position = "right") +
  theme_cowplot(font_size = 14)
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_locb.jpg", p2, width = 8, height = 6, dpi = 300)



################################################################################
## Location by symbiont potential PERMANOVA


keep_rows <- !is.na(meta2$symbiont.potential)
meta_symb <- meta2[keep_rows, , drop = FALSE]
numeric_symb <- permanova_numeric_data2[keep_rows, , drop = FALSE]

bray_curtis_locS <- vegdist(numeric_symb, method = "bray")
locS_permanova_result <- adonis2(
  bray_curtis_locS ~ location / symbiont.potential, 
  data = meta_symb, 
  permutations = 999
)
print(locS_permanova_result)          
# adonis2(formula = bray_curtis_locS ~ location/symbiont.potential, data = meta_symb, permutations = 999)
# Df SumOfSqs      R2      F Pr(>F)    
# Model      5   46.735  53.259  0.001 ***
#   Residual 525   92.139 0.66347                  
# Total    530  138.874 1.00000                  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1


# global model stats
perm_p_val <- locS_permanova_result$`Pr(>F)`[1]
perm_r2    <- round(locS_permanova_result$R2[1], 3)
p_label    <- if(perm_p_val <= 0.001) "p < 0.001" else paste("p =", perm_p_val)
stats_annotation <- paste0("PERMANOVA: R² = ", perm_r2, ", ", p_label)

pcoa_result3 <- cmdscale(bray_curtis_locS, eig = TRUE, k = 2)
pcoa_points3 <- as.data.frame(pcoa_result3$points)
colnames(pcoa_points3) <- c("PCoA1", "PCoA2")
pcoa_points3 <- bind_cols(pcoa_points3, meta_symb)

pos_eig <- pcoa_result3$eig[pcoa_result3$eig > 0]
var_explained <- round(100 * pcoa_result3$eig[1:2] / sum(pos_eig), 1)

shapes_sym <- c(
  "Aposymbiotic" = 8,
  "Facultative" = 20,
  "Symbiotic" = 18
)

p3 <- ggplot(pcoa_points3, aes(x = PCoA1, y = PCoA2, color = location)) +
  geom_point(aes(shape = symbiont.potential), size = 3, alpha = 0.8) +
  scale_color_manual(values = cols_location) +
  scale_shape_manual(values = shapes_sym) + 
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Location",
    shape = "Symbiont Potential"
  ) +
  # PERMANOVA annotation to the bottom-left
  annotate(
    "text",
    x = -Inf, y = -Inf,
    label = stats_annotation,
    hjust = -0.05, vjust = -0.6,
    size = 4
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "right")

# Save
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_locsym_annotated.jpg", p3, width = 8, height = 6, dpi = 300)

# keep_rows <- !is.na(meta2$symbiont.potential)
# # subset the metadata and the numeric data simultaneously
# meta_symb <- meta2[keep_rows, , drop = FALSE]
# numeric_symb <- permanova_numeric_data2[keep_rows, , drop = FALSE]
# 
# # BC dissimilarity then nested PERMANOVA
# bray_curtis_locS <- vegdist(numeric_symb, method = "bray")
# locS_permanova_result <- adonis2(
#   bray_curtis_locS ~ location / symbiont.potential, 
#   data = meta_symb, 
#   permutations = 999
# )
# print(locS_permanova_result)               
# # adonis2(formula = bray_curtis_locS ~ location/symbiont.potential, data = meta_symb, permutations = 999)
# # Df SumOfSqs      R2      F Pr(>F)    
# # Model      5   46.735 0.33653 53.259  0.001 ***
# #   Residual 525   92.139 0.66347                  
# # Total    530  138.874 1.00000                  
# 
# shapes_sym <- c(
#   "Aposymbiotic" = 8,     
#   "Facultative" = 20, 
#   "Symbiotic" = 18   
# )
# 
# pcoa_result3 <- cmdscale(bray_curtis_locS, eig = TRUE, k = 2)
# pcoa_points3 <- as.data.frame(pcoa_result3$points)
# colnames(pcoa_points3) <- c("PCoA1", "PCoA2")
# pcoa_points3 <- bind_cols(pcoa_points3, meta_symb)
# 
# # compute percent variance explained (use only positive eigenvalues)
# pos_eig <- pcoa_result3$eig[pcoa_result3$eig > 0]
# var_explained <- round(100 * pcoa_result3$eig[1:2] / sum(pos_eig), 1)
# 
# p3 <- ggplot(pcoa_points3, aes(x = PCoA1, y = PCoA2, color = location)) +
#   # stat_ellipse(
#   #   aes(fill = location, group = location),
#   #   geom = "polygon",
#   #   alpha = 0.3,
#   #   level = 0.95,
#   #   colour = NA
#   # ) +
#   geom_point(aes(shape = symbiont.potential), size = 3, alpha = 0.8) +
#   scale_color_manual(values = cols_location) +
#   scale_fill_manual(values = cols_location) +
#   scale_shape_manual(values = shapes_sym) + 
#   labs(
#     x = paste0("PCoA1: (", var_explained[1], "%)"),
#     y = paste0("PCoA2: (", var_explained[2], "%)"),
#     color = "Location",
#     fill = "Location",
#     shape = "Symbiont Potential"
#   ) +
#   theme_cowplot(font_size = 14) +
#   theme(legend.position = "right")
# ggsave("/work/hs325/World_Corals/misc/figs/pcoa_locsym.jpg", p3, width = 8, height = 6, dpi = 300)

################################################################################

## Permanova bleaching by symbiont potential

keep_rows <- !is.na(meta2$bleaching) & !is.na(meta2$symbiont.potential)
meta_sub <- meta2[keep_rows, , drop = FALSE]
numeric_sub <- permanova_numeric_data2[keep_rows, , drop = FALSE]

bray_curtis_sub <- vegdist(numeric_sub, method = "bray")
bleach_sym_permanova <- adonis2(
  bray_curtis_sub ~ bleaching / symbiont.potential, 
  data = meta_sub, 
  permutations = 999
)
# adonis2(formula = bray_curtis_sub ~ bleaching/symbiont.potential, data = meta_sub, permutations = 999)
# Df SumOfSqs     R2      F Pr(>F)    
# Model      4   35.994 0.2598 46.067  0.001 ***
#   Residual 525  102.550 0.7402                  
# Total    529  138.543 1.0000                  
# ---

perm_p_val <- bleach_sym_permanova$`Pr(>F)`[1]
perm_r2    <- round(bleach_sym_permanova$R2[1], 3)
p_label    <- if(perm_p_val <= 0.001) "p < 0.001" else paste("p =", perm_p_val)
stats_annotation <- paste0("PERMANOVA: R² = ", perm_r2, ", ", p_label)

pcoa_res <- cmdscale(bray_curtis_sub, eig = TRUE, k = 2)
pcoa_pts <- as.data.frame(pcoa_res$points)
colnames(pcoa_pts) <- c("PCoA1", "PCoA2")
pcoa_pts <- bind_cols(pcoa_pts, meta_sub)

pos_eig <- pcoa_res$eig[pcoa_res$eig > 0]
var_explained <- round(100 * pcoa_res$eig[1:2] / sum(pos_eig), 1)

p_bleach_sym <- ggplot(pcoa_pts, aes(x = PCoA1, y = PCoA2, color = bleaching)) +
  geom_point(aes(shape = symbiont.potential), size = 3, alpha = 0.8) +
  scale_color_manual(values = cols_bleaching) +
  scale_shape_manual(values = shapes_sym) + 
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Bleaching Status",
    shape = "Symbiont Potential"
  ) +
  annotate(
    "text",
    x = -Inf, y = -Inf,
    label = stats_annotation,
    hjust = -0.05, vjust = -0.6,
    size = 4
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "right")

# Save the result
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_bsym.jpg", p_bleach_sym, width = 8, height = 6, dpi = 300)

################################################################################
pcoa_supp <- plot_grid(
  p3 + theme(legend.position = "right"), 
  p_bleach_sym + theme(legend.position = "right"), 
  labels = c("A", "B"), 
  label_size = 20,
  ncol = 1,              
  align = "v",
  axis = "lr"
)

ggsave("/work/hs325/World_Corals/misc/figs/pcoa_supp.jpg", 
       pcoa_supp, 
       width = 10, 
       height = 12, 
       dpi = 300)
################################################################################

## Bleaching status alone PERMANOVA

desired_levels <- c("Bleached", "Non-Bleached", "Not Applicable")
permanova_numeric_data <- df %>% select(starts_with("x"))
keep_rows <- df$bleaching %in% desired_levels & complete.cases(permanova_numeric_data)

# subset numeric matrix and metadata
permanova_numeric_data2 <- permanova_numeric_data[keep_rows, , drop = FALSE]
meta_bleach <- df[keep_rows, ] %>%
  mutate(bleaching = factor(as.character(bleaching), levels = desired_levels))

# compute Bray–Curtis dissimilarity
bray_curtis_bleach <- vegdist(permanova_numeric_data2, method = "bray")
# PERMANOVA: bleaching alone
bleaching_permanova_result <- adonis2(
  bray_curtis_bleach ~ bleaching,
  data = meta_bleach,
  permutations = 999
)
print(bleaching_permanova_result)
# Df SumOfSqs     R2      F Pr(>F)    
# Model      2   19.568 0.1284 41.541  0.001 ***
#   Residual 564  132.838 0.8716                  
# Total    566  152.407 1.0000                  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

pcoa_result <- cmdscale(bray_curtis_bleach, eig = TRUE, k = 2)
pcoa_points <- as.data.frame(pcoa_result$points)
colnames(pcoa_points) <- c("PCoA1", "PCoA2")
pcoa_points <- bind_cols(pcoa_points, meta_bleach)

r2_val <- round(bleaching_permanova_result$R2[1], 3)
p_val  <- bleaching_permanova_result$`Pr(>F)`[1]
p_lab <- if(p_val <= 0.001) "p < 0.001" else paste0("p = ", round(p_val, 3))
stats_label <- paste0("PERMANOVA: R² = ", r2_val, ", ", p_lab)

# compute percent variance explained using only positive eigenvalues
pos_eig <- pcoa_result$eig[pcoa_result$eig > 0]
var_explained <- round(100 * pcoa_result$eig[1:2] / sum(pos_eig), 1)

p4 <- ggplot(pcoa_points, aes(x = PCoA1, y = PCoA2, color = bleaching, fill = bleaching)) +
  geom_point(size = 3, alpha = 0.9) +
  stat_ellipse(
    geom = "polygon",
    alpha = 0.15,
    level = 0.95,
    type = "t",
    colour = NA
  ) + annotate(
    "text", 
    x = -Inf, y = -Inf, 
    label = stats_label, 
    hjust = -0.1, # Shift slightly right from the edge
    vjust = -1.2, # Shift slightly up from the edge
    size = 4, 
    fontface = "italic"
  ) +
  scale_color_manual(values = cols_bleaching) +
  scale_fill_manual(values = cols_bleaching) +
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Bleaching Status",
    fill = "Bleaching Status"
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "right")
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_bleaching.jpg", p4, width = 8, height = 6, dpi = 300)


################################################################################
## Symbiont potential alone PERMANOVA

desired_levels <- na.omit(unique(as.character(df$symbiont.potential)))
permanova_numeric_data <- df %>% select(starts_with("x"))
keep_rows <- !is.na(df$symbiont.potential) & complete.cases(permanova_numeric_data)

permanova_numeric_data2 <- permanova_numeric_data[keep_rows, , drop = FALSE]
desired_levels <- c("Aposymbiotic", "Facultative", "Symbiotic")
meta_sym <- df[keep_rows, ] %>%
  mutate(symbiont.potential = factor(as.character(symbiont.potential), 
                                     levels = desired_levels))

# Bray–Curtis dissimilarity
bray_curtis_sym <- vegdist(permanova_numeric_data2, method = "bray")

symbiont_permanova_result <- adonis2(
  bray_curtis_sym ~ symbiont.potential,
  data = meta_sym,
  permutations = 999
)
print(symbiont_permanova_result)
# adonis2(formula = bray_curtis_sym ~ symbiont.potential, data = meta_sym, permutations = 999)
# Df SumOfSqs     R2      F Pr(>F)    
# Model      2   32.374 0.2278 78.911  0.001 ***
#   Residual 535  109.746 0.7722                  
# Total    537  142.121 1.0000                  
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

pcoa_result <- cmdscale(bray_curtis_sym, eig = TRUE, k = 2)
pcoa_points <- as.data.frame(pcoa_result$points)
colnames(pcoa_points) <- c("PCoA1", "PCoA2")
pcoa_points <- bind_cols(pcoa_points, meta_sym)

r2_val <- round(symbiont_permanova_result$R2[1], 3)
p_val  <- symbiont_permanova_result$`Pr(>F)`[1]

# Format the p-value label
p_lab <- if(p_val <= 0.001) "p < 0.001" else paste0("p = ", round(p_val, 3))
stats_label <- paste0("PERMANOVA: R² = ", r2_val, ", ", p_lab)

p5 <- ggplot(pcoa_points, aes(x = PCoA1, y = PCoA2, color = symbiont.potential, fill = symbiont.potential)) +
  geom_point(size = 3, alpha = 0.9) +
  stat_ellipse(
    geom = "polygon",
    alpha = 0.15,
    level = 0.95,
    type = "t",
    colour = NA
  ) +
  scale_color_manual(values = cols_symbiont) +
  scale_fill_manual(values = cols_symbiont) +
  annotate(
    "text", 
    x = -Inf, y = -Inf, 
    label = stats_label, 
    hjust = -0.1, # Shift slightly right from the edge
    vjust = -1.2, # Shift slightly up from the edge
    size = 4, 
    fontface = "italic"
  ) +
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Symbiont Potential",
    fill = "Symbiont Potential"
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "right")
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_symbiont.jpg", 
       p5, width = 8, height = 6, dpi = 300)

################################################################################
## Location alone PERMANOVA

desired_levels <- na.omit(unique(as.character(df$location)))
permanova_numeric_data <- df %>% select(starts_with("x"))
keep_rows <- !is.na(df$location) & complete.cases(permanova_numeric_data)

permanova_numeric_data2 <- permanova_numeric_data[keep_rows, , drop = FALSE]
meta_loc <- df[keep_rows, ] %>%
  mutate(location = factor(as.character(location), levels = desired_levels))

# Bray–Curtis dissimilarity
bray_curtis_loc <- vegdist(permanova_numeric_data2, method = "bray")

location_permanova_result <- adonis2(
  bray_curtis_loc ~ location,
  data = meta_loc,
  permutations = 999
)
print(location_permanova_result)
# Df SumOfSqs      R2     F Pr(>F)    
# Model      3    39.27 0.26013 65.28  0.001 ***
#   Residual 557   111.69 0.73987                 
# Total    560   150.96 1.00000                 
# ---

pcoa_result <- cmdscale(bray_curtis_loc, eig = TRUE, k = 2)
pcoa_points <- as.data.frame(pcoa_result$points)
colnames(pcoa_points) <- c("PCoA1", "PCoA2")
pcoa_points <- bind_cols(pcoa_points, meta_loc)


r2_val <- round(location_permanova_result$R2[1], 3)
p_val  <- location_permanova_result$`Pr(>F)`[1]

# Format the p-value label
p_lab <- if(p_val <= 0.001) "p < 0.001" else paste0("p = ", round(p_val, 3))
stats_label <- paste0("PERMANOVA: R² = ", r2_val, ", ", p_lab)

# percent variance explained (positive eigenvalues only)
pos_eig <- pcoa_result$eig[pcoa_result$eig > 0]
var_explained <- round(100 * pcoa_result$eig[1:2] / sum(pos_eig), 1)

p6 <- ggplot(pcoa_points, aes(x = PCoA1, y = PCoA2, color = location, fill = location)) +
  geom_point(size = 3, alpha = 0.9) +
  stat_ellipse(
    geom = "polygon",
    alpha = 0.3,
    level = 0.95,
    type = "t",
    colour = NA
  ) +
  annotate(
    "text", 
    x = -Inf, y = -Inf, 
    label = stats_label, 
    hjust = -0.1, # Shift slightly right from the edge
    vjust = -1.2, # Shift slightly up from the edge
    size = 4, 
    fontface = "italic"
  ) +
  scale_color_manual(values = cols_location) +
  scale_fill_manual(values = cols_location) +
  labs(
    x = paste0("PCoA1: (", var_explained[1], "%)"),
    y = paste0("PCoA2: (", var_explained[2], "%)"),
    color = "Location",
    fill = "Location"
  ) +
  theme_cowplot(font_size = 14) +
  theme(legend.position = "right")

# save plot
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_location.jpg", p6, width = 8, height = 6, dpi = 300)

################################################################################

plot_grid(p4, p5, p6)
pcoa_supp <- plot_grid(
  p4, p5, p6, 
  labels = c("A", "B", "C"), 
  label_size = 16,
  ncol = 2,
  nrow = 2,
  align = "hv",
  axis = "tb" # aligns top and bottom axes
)
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_supp_2.jpg", 
       pcoa_supp, width = 12, height = 6, dpi = 300, bg = "white")

################################################################################
################################################################################
# Box plots

metabolite_data <- df %>%
  select(starts_with("x"))

########### abundance
avg_abundance <- metabolite_data %>%
  mutate(avg_abundance = rowMeans(., na.rm = TRUE)) %>%
  select(avg_abundance)

plot_data <- df %>%
  select(host_order) %>%
  bind_cols(avg_abundance) %>%
  mutate(
    group = ifelse(host_order == "Scleractinia",
                   "1",
                   "0")
  )
plot_data$group[is.na(plot_data$group)] <- 0
unique(plot_data$group)

plot_data_clean <- plot_data %>%
  mutate(group = factor(group)) %>% # Ensure it's a factor first
  mutate(group = fct_recode(group, 
                            "Scleractinia" = "1", 
                            "Other" = "0")) %>%
  mutate(group = fct_relevel(group, "Scleractinia", "Other"))
cols_sclero_named <- c("Scleractinia" = "#DE7862FF", "Other" = "#D8AF39FF")

p_abundance <- ggplot(plot_data_clean, aes(x = group, y = avg_abundance, fill = group, color = group)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5, show.legend = FALSE) +
  geom_boxplot(alpha = 0.7, width = 0.6, outlier.shape = NA, color = "black") + 
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.format",
    label.x = 1.4,
    symnum.args = list(
      cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1), 
      symbols = c("****", "***", "**", "*", "ns")
    )
  ) +
  scale_fill_manual(values = cols_sclero_named) +
  scale_color_manual(values = cols_sclero_named) +
  labs(
    y = "Metabolite Abundance"
  ) +
  theme_pubr(base_size = 13) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.title.x = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
p_abundance <- p_abundance + scale_y_continuous(
  labels = label_number(scale_cut = cut_short_scale()) # Formats as K, M, etc.
)
print(p_abundance)


########### ubiquity

plot_data_clean <- plot_data %>%
  mutate(group = factor(group)) %>% # Ensure it's a factor first
  mutate(group = fct_recode(group, 
                            "Scleractinia" = "1", 
                            "Other" = "0")) %>%
  mutate(group = fct_relevel(group, "Scleractinia", "Other"))
cols_sclero_named <- c("Scleractinia" = "#DE7862FF", "Other" = "#D8AF39FF")

ubiquity_values <- metabolite_data %>%
  mutate(avg_ubiquity = rowMeans(. > 0, na.rm = TRUE)) %>%
  pull(avg_ubiquity) # pull() extracts the column as a vector
plot_data_clean$avg_ubiquity <- ubiquity_values


p_ubiquity <- ggplot(plot_data_clean, aes(x = group, y = avg_ubiquity, fill = group, color = group)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5, show.legend = FALSE) +
  geom_boxplot(alpha = 0.7, width = 0.6, outlier.shape = NA, color = "black") + 
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.format",
    label.x = 1.4,
    symnum.args = list(
      cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1), 
      symbols = c("****", "***", "**", "*", "ns")
    )
  ) +
  scale_fill_manual(values = cols_sclero_named) +
  scale_color_manual(values = cols_sclero_named) +
  labs(
    y = "Metabolite Ubiquity"
  ) +
  theme_pubr(base_size = 13) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    axis.title.x = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
print(p_ubiquity)


########### richness
richness_values <- rowSums(metabolite_data > 0, na.rm = TRUE)
plot_data_clean$richness <- richness_values

p_richness <- ggplot(plot_data_clean, aes(x = group, y = richness, fill = group, color = group)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5, show.legend = FALSE) +
  geom_boxplot(alpha = 0.7, width = 0.6, outlier.shape = NA, color = "black") + 
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.format",
    label.x = 1.4,
    symnum.args = list(
      cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1), 
      symbols = c("****", "***", "**", "*", "ns")
    )
  ) +
  scale_fill_manual(values = cols_sclero_named) +
  scale_color_manual(values = cols_sclero_named) +
  labs(
    y = "Metabolite Richness"
  ) +
  theme_pubr(base_size = 13) +
  theme(
    legend.position = "none",
    axis.title.x = element_blank(),
    axis.text.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
print(p_richness)


############ shannon
shannon_values <- diversity(metabolite_data, index = "shannon")
plot_data_clean$shannon  <- shannon_values

p_entropy <- ggplot(plot_data_clean, aes(x = group, y = shannon, fill = group, color = group)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 1.5, show.legend = FALSE) +
  geom_boxplot(alpha = 0.7, width = 0.6, outlier.shape = NA, color = "black") + 
  stat_compare_means(
    method = "wilcox.test", 
    label = "p.format",
    label.x = 1.4,
    symnum.args = list(
      cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, 1), 
      symbols = c("****", "***", "**", "*", "ns")
    )
  ) +
  scale_fill_manual(values = cols_sclero_named) +
  scale_y_continuous(breaks = c(5, 6, 7), limits = c(4.5, 7.5)) +
  scale_color_manual(values = cols_sclero_named) +
  labs(
    y = "Shannon Entropy"
  ) +
  theme_pubr(base_size = 13) +
  theme(
    legend.position = "none",
    axis.title.x = element_blank(),
    axis.text.x = element_text(size = 16),
    axis.title.y = element_text(size = 16),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )
print(p_entropy)

################################################################################
################################################################################
# Dendrograms

avg_metabolite_values_family <- df |> 
  group_by(host_family) |> 
  filter(!is.na(host_family)) |> 
  summarise(across(starts_with("x"), mean, na.rm = TRUE)) |>
  ungroup()

mat_family <- avg_metabolite_values_family %>%
  as.data.frame()
rownames(mat_family) <- mat_family$host_family
mat_family <- mat_family[ , setdiff(names(mat_family), "host_family")]
mat_family[] <- lapply(mat_family, as.numeric)
mat_family <- mat_family %>% mutate_all(~ifelse(is.na(.), 0, .))

bray_curtis_family <- vegdist(sqrt(mat_family), method = 'bray')
cluster.average <- hclust(bray_curtis_family, method = 'ward.D') 
#method = average uses upgma

dend_data <- dendro_data(cluster.average, type = "rectangle")

branch_palette <- c(
  "Ochrophyta"         = "#32CD32", # Lime Green
  "Chlorophyta"        = "#00CED1", # Dark Turquoise
  "Cnidaria"           = "#304530", # Gold 
  "Chordata"           = "#9370DB", # Medium Purple
  "Porifera"           = "#ec93ed", # Hot Pink,
  # "Rhodophyta"         = "#b89d4d", # Gold
  "Scleractinia_Color" = "#DE7862FF", # Kept your specific Scleractinia color
  "grey40"             = "grey40"   # Internal nodes
)

#################################### with n=

family_counts <- df %>%
  filter(!is.na(host_family)) %>%
  group_by(host_family) %>%
  summarise(n = n()) %>%
  ungroup()

leaf_metadata <- df %>%
  filter(!is.na(host_family)) %>%
  group_by(host_family) %>%
  summarize(
    phylum = first(host_phylum),
    is_scler = first(host_order) == "Scleractinia"
  ) %>%
  left_join(family_counts, by = "host_family") %>%
  # Create the label string: "Family (n=X)"
  mutate(label_with_n = paste0(host_family, " (n=", n, ")")) %>%
  ungroup()

dend <- as.dendrogram(cluster.average)

target_metadata <- leaf_metadata %>%
  arrange(desc(is_scler), phylum, host_family)
target_order <- target_metadata$host_family

dend <- dendextend::rotate(dend, target_order)
dend_data <- dendro_data(dend, type = "rectangle")

dend_segments <- dend_data$segments %>%
  left_join(dend_data$labels %>% select(x, label), by = "x") %>% 
  left_join(leaf_metadata, by = c("label" = "host_family")) %>%
  mutate(
    branch_color = case_when(
      is_scler == TRUE ~ "Scleractinia_Color",
      !is.na(phylum)   ~ phylum,
      TRUE             ~ "grey40" 
    )
  )

dend_labels_phylum <- dend_data$labels %>%
  left_join(leaf_metadata, by = c("label" = "host_family")) %>%
  mutate(
    text_color_group = if_else(is_scler == TRUE, "Scleractinia_Color", phylum)
  )

p_dendro <- ggplot() +
  geom_segment(data = dend_segments, 
               aes(x = x, y = y, xend = xend, yend = yend, color = branch_color),
               linewidth = 0.8) +
  geom_text(data = dend_labels_phylum, 
            aes(x = x, y = y, label = label_with_n, color = text_color_group),
            hjust = -0.1, 
            size = 5, 
            show.legend = FALSE) +
  coord_flip() +
  scale_y_reverse(
    expand = c(0.5, 0), # Slightly increased expansion to fit longer text
    breaks = seq(0, 5, 1)
  ) +
  scale_color_manual(
    values = branch_palette,
    breaks = c("Scleractinia_Color","Chlorophyta", "Chordata", "Cnidaria", "Ochrophyta","Porifera"),
    labels = c("Scleractinia","Chlorophyta", "Chordata", "Cnidaria", "Ochrophyta","Porifera")
  ) +
  theme_pubr() +
  labs(y = "Bray-Curtis Distance", x = "", color = "Classification") +
  theme(
    axis.text.y = element_blank(), 
    axis.ticks.y = element_blank(),
    axis.line.y = element_blank(),
    legend.position = c(0.10, 0.5),   
    legend.background = element_blank(), 
    legend.box.background = element_blank(),
    legend.title = element_text(size = 20),
    legend.text = element_text(size = 16),
  ) +
  guides(color = guide_legend(
    override.aes = list(alpha = 1, size = 4, shape = 15)
  ))
print(p_dendro)
ggsave("/work/hs325/World_Corals/misc/figs/dendro.jpg", p_dendro, width=10, height=8, dpi = 600)
# current_order <- labels(dend)
# print(current_order)
# dend <- click_rotate(dend)

################################################################################
## combine plots with cowplot for talk
# row 1 legend, p, p_dendro


legend_sclero <- get_legend(
  p + 
    guides(color = guide_legend(title = "Order"), fill = guide_legend(title = "Order")) +
    theme(
      legend.position = "left", 
      legend.box.margin = margin(6, 0, 0, 6),
      legend.title = element_text(size = 24, face = "bold"), 
      legend.text = element_text(size = 20),
      legend.title.align = 0.5,
      legend.text.align = 0.5,
      legend.key.size = unit(1.2, "lines")))
clean_theme <- theme(
  legend.position = "none",
  plot.title = element_blank() # Remove titles to keep row 2 clean
)

p <- p + theme(legend.position = "none")

row1 <- plot_grid(
  legend_sclero, p, p_dendro,
  ncol = 3,
  rel_widths = c(0.4, 1, 1.2), # Legend is narrow, plots are equal
  labels = c("", "A", "B"),
  label_size = 24
)

p_abundance_clean <- p_abundance # + clean_theme()
p_ubiquity_clean  <- p_ubiquity # + clean_theme()
p_entropy_clean  <- p_entropy # + clean_theme()
p_richness_clean  <- p_richness # + clean_theme()

row2 <- plot_grid(
  p_richness_clean,p_entropy_clean,p_ubiquity_clean,p_abundance_clean,
  nrow = 1,
  rel_widths = c(1,1,1,1),
  align = 'h', axis = 'tb',
  # labels = c("C", "D", "E", "F"),
  label_size = 24
)

final_plot <- plot_grid(
  row1, 
  row2, 
  ncol = 1, 
  rel_heights = c(1, 1) 
)
ggsave("/work/hs325/World_Corals/misc/figs/fig2_ppt.jpg", row2, width = 12, height = 6, dpi = 300)


################################################################################
# combine plots with cowplot
# row 1: legend, p, p2 
# row 2: p_dendro, p_abundance, p_ubiquity, p_evenness, p_richness (boxplots should be tight)

legend_sclero <- get_legend(
  p + 
    guides(color = guide_legend(title = "Order"), fill = guide_legend(title = "Order")) +
    theme(
      legend.position = "left", 
      legend.box.margin = margin(6, 0, 0, 6),
      legend.title = element_text(size = 24, face = "bold"), 
      legend.text = element_text(size = 20),
      legend.title.align = 0.5,
      legend.text.align = 0.5,
      legend.key.size = unit(1.2, "lines")))
clean_theme <- theme(
  legend.position = "none",
  plot.title = element_blank() # Remove titles to keep row 2 clean
)

p <- p + theme(legend.position = "none")
# p_dendro_clean    <- p_dendro + clean_theme
p_abundance_clean <- p_abundance + clean_theme
p_ubiquity_clean  <- p_ubiquity + clean_theme
p_entropy_clean  <- p_entropy + clean_theme
p_richness_clean  <- p_richness + clean_theme

row1 <- plot_grid(
  legend_sclero, p, p2,
  ncol = 3,
  rel_widths = c(0.4, 1, 1.2), # Legend is narrow, plots are equal
  labels = c("", "A", "B"),
  label_size = 24
)

row2 <- plot_grid(
  p_richness_clean,p_entropy_clean,p_ubiquity_clean,p_abundance_clean,p_dendro,
  nrow = 1,
  rel_widths = c(0.6, 0.6, 0.6, 0.6, 1.5), # Give Dendro more space
  align = 'h', axis = 'tb',
  labels = c("C", "D", "E", "F", "G"),
  label_size = 24
)

final_plot <- plot_grid(
  row1, 
  row2, 
  ncol = 1, 
  rel_heights = c(1, 1) 
)

print(final_plot)
ggsave("/work/hs325/World_Corals/misc/figs/fig2.jpg", final_plot, width = 18, height = 12, dpi = 300)

################################################################################

# PERMANOVA by df$scleractinia for Curacao
# PERMANOVA by df$bleaching for all
# color points by cols_sclero for the first three, cols_bleaching for last one
# make combined plot with PERMANOVA values inset for each

##

keep_cur <- meta2$location == "Curaçao" & !is.na(meta2$scleractinia)
meta_cur <- meta2[keep_cur, ]
num_cur  <- permanova_numeric_data2[keep_cur, ]

bc_cur <- vegdist(num_cur, method = "bray")
perm_cur <- adonis2(bc_cur ~ scleractinia, data = meta_cur, permutations = 999)

p_val_cur <- perm_cur[["Pr(>F)"]][1]
r2_cur    <- round(perm_cur[["R2"]][1], 3)
p_lab_cur <- if(is.na(p_val_cur)) "p = NA" else if(p_val_cur <= 0.001) "p < 0.001" else paste("p =", p_val_cur)
stats_cur <- paste0("PERMANOVA: R² = ", r2_cur, ", ", p_lab_cur)

pcoa_cur <- cmdscale(bc_cur, eig = TRUE, k = 2)
df_cur   <- cbind(as.data.frame(pcoa_cur$points), meta_cur)
var_cur  <- round(100 * pcoa_cur$eig[1:2] / sum(pcoa_cur$eig[pcoa_cur$eig > 0]), 1)

p_cur <- ggplot(df_cur, aes(x = V1, y = V2)) +
  geom_point(
    aes(
      color = as.factor(scleractinia), 
      shape = "Curaçao"  # Mapping a constant string creates the legend entry
    ), 
    size = 3, 
    alpha = 0.7
  ) +
  scale_color_manual(
    values = cols_sclero, 
    labels = c("1" = "Scleractinia", "0" = "Other"),
    name = "Order"
  ) +
  scale_shape_manual(
    values = c("Curaçao" = 16), # Choose the shape (16 is solid circle)
    name = "Location"           # This becomes the header for the shape legend
  ) +
  labs(
    x = paste0("PCoA1: (", var_cur[1], "%)"), 
    y = paste0("PCoA2: (", var_cur[2], "%)")
  ) +
  annotate(
    "text", x = Inf, y = -Inf, 
    label = stats_cur, 
    hjust = 1.05, vjust = -1.2, 
    size = 3.5, fontface = "italic"
  ) +
  theme_cowplot()
p_cur
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_cur_scler.jpg", p_cur, width = 8, height = 6, dpi = 300)

################################################################################
### bleaching within  each location (Hawaii has no bleached samples)

locs_bleach <- c("Sri Lanka", "North Carolina", "Curaçao")
bleach_plots <- list()

for (loc in locs_bleach) {
  
  keep_idx <- meta2$location == loc & !is.na(meta2$bleaching)
  if(sum(keep_idx) < 3 || length(unique(meta2$bleaching[keep_idx])) < 2) next
  
  m_sub <- meta2[keep_idx, ]
  n_sub <- permanova_numeric_data2[keep_idx, ]
  
  # BC & PERMANOVA
  bc_sub <- vegdist(n_sub, method = "bray")
  perm_sub <- adonis2(bc_sub ~ bleaching, data = m_sub, permutations = 999)
  
  # Extract Stats
  p_val <- perm_sub[["Pr(>F)"]][1]
  r2_val <- round(perm_sub[["R2"]][1], 3)
  p_lab <- if(is.na(p_val)) "p = NA" else if(p_val <= 0.001) "p < 0.001" else paste("p =", p_val)
  stats_label <- paste0("PERMANOVA: R² = ", r2_val, ", ", p_lab)
  
  # PCoA
  pcoa_res <- cmdscale(bc_sub, eig = TRUE, k = 2)
  var_exp  <- round(100 * pcoa_res$eig[1:2] / sum(pcoa_res$eig[pcoa_res$eig > 0]), 1)
  pcoa_df  <- cbind(as.data.frame(pcoa_res$points), m_sub)
  
  # Generate Plot
  p <- ggplot(pcoa_df, aes(x = V1, y = V2, color = bleaching)) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_manual(values = cols_bleaching) +
    labs(
      title = loc,
      x = paste0("PCoA1: (", var_exp[1], "%)"), 
      y = paste0("PCoA2: (", var_exp[2], "%)")
    ) +
    annotate(
      "text", x = Inf, y = -Inf, 
      label = stats_label, 
      hjust = 1.05, vjust = -1.2, 
      size = 3, fontface = "italic"
    ) +
    theme_cowplot() +
    theme(legend.position = "none") 
  
  bleach_plots[[loc]] <- p
}

#combine
shared_legend <- get_legend(
  ggplot(meta2[!is.na(meta2$bleaching),], aes(x=1, y=1, color=bleaching)) +
    geom_point(size = 4) + 
    scale_color_manual(values = cols_bleaching, name = "Status") +
    theme_cowplot() + 
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.justification = "center",
      legend.box.just = "center"
    )
)

plot_grid <- plot_grid(
  plotlist = bleach_plots,
  labels = c("A", "B", "C"),
  label_size=24,
  ncol = 1,
  align = "v",
  axis = "lr"
)

final_bleaching_fig <- plot_grid(plot_grid, shared_legend, ncol = 1, nrow = 2, rel_heights = c(1, 0.05))
ggsave("/work/hs325/World_Corals/misc/figs/pcoa_bleachbyloc.jpg", 
       final_bleaching_fig, width = 12, height = 11, dpi = 300)