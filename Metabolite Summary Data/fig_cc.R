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

## Collectors Curves

## plot on top of each other
# a) Scler vs non Scler
# b) by location
# c) by bleaching status
# d) by symbiont potential
# e) total, by metabolic origin
# f) total, by metabolic superclass?


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

comm_matrix = df |>
  select(c(sample, grep("^x", names(df), value = TRUE))) |>
  column_to_rownames(var = "sample")

comm_matrix = comm_matrix |>
  mutate(across(where(is.numeric), ~ ifelse(.x > 0, 1, 0)))

################################################################################
# Scleractinia vs non-Scleractinia CC

# comm_matrix = df |>
#   select(c(sample, grep("^x", names(df), value = TRUE))) |>
#   column_to_rownames(var = "sample")
# 
# comm_matrix = comm_matrix |>
#   mutate(across(where(is.numeric), ~ ifelse(.x > 0, 1, 0)))

sclero_samples <- df %>% filter(scleractinia == "1") %>% pull(sample)
other_samples  <- df %>% filter(scleractinia == "0" | is.na(scleractinia)) %>% pull(sample)

comm_sclero <- comm_matrix[rownames(comm_matrix) %in% sclero_samples, ]
comm_other  <- comm_matrix[rownames(comm_matrix) %in% other_samples, ]

acc_sclero <- specaccum(comm_sclero, method = "random", permutations = 30)
acc_other  <- specaccum(comm_other, method = "random", permutations = 30)

plot_data_sclero <- data.frame(
  percent_samples = (acc_sclero$sites / max(acc_sclero$sites)) * 100,
  richness = acc_sclero$richness,
  sd = acc_sclero$sd,
  group = "1"
)

plot_data_other <- data.frame(
  percent_samples = (acc_other$sites / max(acc_other$sites)) * 100,
  richness = acc_other$richness,
  sd = acc_other$sd,
  group = "0"
)

combined_acc <- bind_rows(plot_data_sclero, plot_data_other) %>%
  mutate(group = factor(group, levels = c("1", "0"))) # Force Scleractinia to the top

p<-ggplot(combined_acc, aes(x = percent_samples, y = richness, color = group, fill = group)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.5, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = cols_sclero, labels = c("1" = "Scleractinia", "0" = "Other")) +
  scale_fill_manual(values = cols_sclero, labels = c("1" = "Scleractinia", "0" = "Other")) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "% of Total Samples",
    y = "Metabolite Richness",
    color = "Order",
    fill = "Order"
  ) +
  theme_pubr() +
  theme(
    axis.title = element_text(),
    legend.position = c(0.98, 0.05),     
    legend.justification = c(1, 0),      
    legend.background = element_blank(), 
    legend.box.background = element_blank()
  )
p

################################################################################
# By location CC

loc_list <- split(df$sample, df$location)

# 2. Run specaccum for each location and format the data
plot_data_locs <- lapply(names(loc_list), function(loc) {
  # Subset community matrix
  comm_sub <- comm_matrix[rownames(comm_matrix) %in% loc_list[[loc]], ]
  
  # Run accumulation
  acc <- specaccum(comm_sub, method = "random", permutations = 30)
  
  # Return data frame with normalized x-axis
  data.frame(
    percent_samples = (acc$sites / max(acc$sites)) * 100,
    richness = acc$richness,
    sd = acc$sd,
    location = loc
  )
})

combined_acc_loc <- bind_rows(plot_data_locs)

p2<-ggplot(combined_acc_loc, aes(x = percent_samples, y = richness, color = location, fill = location)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.4, color = NA) +
  geom_line(aes(linetype = location), linewidth = 1.2) +
  
  scale_color_manual(values = cols_location) +
  scale_fill_manual(values = cols_location) +
  scale_linetype_manual(values = c("Curaçao" = "solid", "Hawaii" = "dashed", 
                                   "North Carolina" = "twodash", "Sri Lanka" = "dotted")) +
  
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "% of Total Samples",
    y = "Metabolite Richness",
    color = "Location",
    fill = "Location",
    linetype = "Location"
  ) +
  theme_pubr() +
  theme(
    axis.title = element_text(),
    legend.position = c(0.98, 0.05),      
    legend.justification = c(1, 0),      
    legend.background = element_blank(), 
    legend.box.background = element_blank(),
    # Increase legend spacing to make the linetypes visible in the legend
    legend.key.width = unit(1.5, "cm") 
  )
p2

################################################################################
# By bleaching status CC

bleach_list <- split(df$sample, df$bleaching)
plot_data_bleach <- lapply(names(bleach_list), function(status) {
  comm_sub <- comm_matrix[rownames(comm_matrix) %in% bleach_list[[status]], ]
  acc <- specaccum(comm_sub, method = "random", permutations = 30)
  
  # Return data frame with normalized x-axis
  data.frame(
    percent_samples = (acc$sites / max(acc$sites)) * 100,
    richness = acc$richness,
    sd = acc$sd,
    bleaching = status
  )
})

combined_acc_bleach <- bind_rows(plot_data_bleach) %>%
  mutate(bleaching = factor(bleaching, levels = c("Bleached", "Non-Bleached", "Not Applicable")))

p3 <- ggplot(combined_acc_bleach, aes(x = percent_samples, y = richness, color = bleaching, fill = bleaching)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.5, color = NA) +
  geom_line(aes(linetype = bleaching), linewidth = 1.2) +
  scale_color_manual(values = cols_bleaching) +
  scale_fill_manual(values = cols_bleaching) +
  scale_linetype_manual(values = c("Bleached" = "solid", 
                                   "Non-Bleached" = "solid", 
                                   "Not Applicable" = "dotted")) +
  
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "% of Total Samples",
    y = "Metabolite Richness",
    color = "Bleaching Status",
    fill = "Bleaching Status",
    linetype = "Bleaching Status"
  ) +
  theme_pubr() +
  theme(
    axis.title = element_text(),
    legend.position = c(0.98, 0.05),      
    legend.justification = c(1, 0),      
    legend.background = element_blank(), 
    legend.box.background = element_blank(),
    legend.key.width = unit(1.5, "cm") 
  )
p3

################################################################################
# Symbiont potential

sym_list <- split(df$sample, df$symbiont.potential)
plot_data_sym <- lapply(names(sym_list), function(status) {
  comm_sub <- comm_matrix[rownames(comm_matrix) %in% sym_list[[status]], ]
  acc <- specaccum(comm_sub, method = "random", permutations = 30)
  
  data.frame(
    percent_samples = (acc$sites / max(acc$sites)) * 100,
    richness = acc$richness,
    sd = acc$sd,
    symbiont.potential = status
  )
})

# combine and set factor levels for consistent plotting order
combined_acc_sym <- bind_rows(plot_data_sym) %>%
  mutate(symbiont.potential = factor(symbiont.potential, 
                                     levels = c("Aposymbiotic", "Facultative", "Symbiotic")))

p4<-ggplot(combined_acc_sym, aes(x = percent_samples, y = richness, 
                             color = symbiont.potential, fill = symbiont.potential)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.5, color = NA) +
  geom_line(aes(linetype = symbiont.potential), linewidth = 1.2) +
    scale_color_manual(values = cols_symbiont) +
  scale_fill_manual(values = cols_symbiont) +
  scale_linetype_manual(values = c("Aposymbiotic" = "solid", 
                                   "Facultative" = "solid", 
                                   "Symbiotic" = "solid")) +
  
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "% of Total Samples",
    y = "Metabolite Richness",
    color = "Symbiont Potential",
    fill = "Symbiont Potential",
    linetype = "Symbiont Potential"
  ) +
  theme_pubr() +
  theme(
    axis.title = element_text(),
    legend.position = c(0.98, 0.05),      
    legend.justification = c(1, 0),      
    legend.background = element_blank(), 
    legend.box.background = element_blank(),
    legend.key.width = unit(1.2, "cm") 
  )
p4

################################################################################
# Refined_origin path plot

## the next step is to use met_df which includes all the columns in comm_matrix
## met_df has a column called refined_origin which contains information about metabolic origin
## refined_origin has values "Host", "Symbiont" "Both" and "Unknown"as a factor in that order
## # of metabolites of each refined_origin

cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
                 "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")

met_df$refined_origin <- factor(met_df$refined_origin, 
                                levels = c("Host", "Symbiont", "Both", "Unknown"))

# compute the accumulation order using 'random' method
acc_total <- specaccum(comm_matrix, method = "random", permutations = 100)

# extract the order of metabolite appearance
# calculate richness per origin at each step of the accumulated samples.
calc_stacked_acc <- function(comm, origins) {
  n_samples <- nrow(comm)
  # Randomize sample order
  sample_order <- sample(1:n_samples)
  
  accumulated_data <- lapply(1:n_samples, function(i) {
    # Subset to the first 'i' samples
    sub_comm <- comm[sample_order[1:i], , drop = FALSE]
    # Find which metabolites are present (colSum > 0)
    present_mets <- colnames(sub_comm)[colSums(sub_comm) > 0]
    # Count origins of present metabolites
    data.frame(samples = i) %>%
      bind_cols(
        met_df %>%
          filter(metabolite %in% present_mets) %>%
          group_by(refined_origin) %>%
          tally() %>%
          pivot_wider(names_from = refined_origin, values_from = n, values_fill = 0)
      )
  })
  bind_rows(accumulated_data)
}

plot_data_stacked <- calc_stacked_acc(comm_matrix, met_df$refined_origin)
plot_data_long <- plot_data_stacked %>%
  pivot_longer(cols = -samples, names_to = "Origin", values_to = "Richness") %>%
  mutate(
    Origin = factor(Origin, levels = c("Host", "Symbiont", "Both", "Unknown")),
    percent_samples = (samples / max(samples)) * 100
  )

p5 <- ggplot(plot_data_long, aes(x = percent_samples, y = Richness, fill = Origin)) +
  geom_area(alpha = 0.85, color = "white", linewidth = 0.2) +
  scale_fill_manual(values = cols_origin) +
  scale_x_continuous(expand = c(0, 0), breaks = seq(0, 100, 25)) +
  scale_y_continuous(expand = c(0, 0)) +
  labs(
    x = "% of Total Samples",
    y = "Metabolite Richness",
    fill = "Metabolic Origin"
  ) +
  theme_pubr() +
  theme(
    axis.title = element_text(),
    legend.position = "right"
  )
p5

################################################################################
# combine

top_row <- plot_grid(
  p5, p, 
  labels = c("A", "B"), 
  label_size = 20,
  label_x=0,
  label_y=1,
  hjust = 0,   
  vjust = 1.5,
  ncol = 2
)
bottom_row <- plot_grid(
  p2, p3, p4, 
  labels = c("C", "D", "E"), 
  label_size = 20,
  label_x=0,
  label_y=1,
  hjust = 0,   
  vjust = 1.5,
  ncol = 3
)
final_plot <- plot_grid(
  top_row, 
  bottom_row, 
  ncol = 1, 
  rel_heights = c(1, 1) 
)
ggsave("/work/hs325/World_Corals/misc/figs/fig3.jpg", final_plot, width=12,height=10,dpi=300)

################################################################################

## for metabolites TBA later - talk with Ty 
# compound superclass (10? levels)
# NPC classifier pathway 7 levels
# coral compound family? 9 levels 

# NPC superclass path plot



