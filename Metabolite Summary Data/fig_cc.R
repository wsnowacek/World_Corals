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
library(here)

# read in data
df <- read.csv(here("Cleaned data CSVs", "qc_data.csv"))
met_df <- read.csv(here("Cleaned data CSVs", "merged_met_plot_df.csv"))

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

## modify if needed
n_sclero <- nrow(df %>% filter(scleractinia == "1"))
n_other  <- nrow(df %>% filter(scleractinia == "0" | is.na(scleractinia)))
label_sclero <- paste0("Scleractinia (n = ", n_sclero, ")")
label_other  <- paste0("Other (n = ", n_other, ")")

p <- ggplot(combined_acc, aes(x = percent_samples, y = richness, color = group, fill = group)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.5, color = NA) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(
    values = cols_sclero, 
    labels = c("1" = label_sclero, "0" = label_other)
  ) +
  scale_fill_manual(
    values = cols_sclero, 
    labels = c("1" = label_sclero, "0" = label_other)
  ) +
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "Percentage of Total Samples",
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

df <- df %>%
  mutate(location = fct_na_value_to_level(location, level = "Curaçao"))

loc_list <- split(df$sample, df$location)

plot_data_locs <- lapply(names(loc_list), function(loc) {
  comm_sub <- comm_matrix[rownames(comm_matrix) %in% loc_list[[loc]], ]
  acc <- specaccum(comm_sub, method = "random", permutations = 30)
  data.frame(
    percent_samples = (acc$sites / max(acc$sites)) * 100,
    richness = acc$richness,
    sd = acc$sd,
    location = loc
  )
})

combined_acc_loc <- bind_rows(plot_data_locs)

loc_counts <- df %>%
  group_by(location) %>%
  summarise(n = n()) %>%
  mutate(label_full = paste0(location, " (n = ", n, ")"))

loc_labels <- setNames(loc_counts$label_full, loc_counts$location)

p2 <- ggplot(combined_acc_loc, aes(x = percent_samples, y = richness, color = location, fill = location)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.4, color = NA) +
  geom_line(aes(linetype = location), linewidth = 1.2) +
  
  # Map the new labels across all scales to synchronize the legend
  scale_color_manual(values = cols_location, labels = loc_labels) +
  scale_fill_manual(values = cols_location, labels = loc_labels) +
  scale_linetype_manual(
    values = c("Curaçao" = "solid", "Hawaii" = "dashed", 
               "North Carolina" = "twodash", "Sri Lanka" = "dotted"),
    labels = loc_labels
  ) +
  
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "Percentage of Total Samples",
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
    legend.key.width = unit(1.5, "cm") 
  )

p2

################################################################################
# By bleaching status CC 
# (one "pale" sample was removed, so 567 instead of 568 total)

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

bleach_counts <- df %>%
  group_by(bleaching) %>%
  summarise(n = n()) %>%
  mutate(label_full = paste0(bleaching, " (n = ", n, ")")) %>%
  arrange(factor(bleaching, levels = c("Bleached", "Non-Bleached", "Not Applicable")))

bleach_labels <- setNames(bleach_counts$label_full, bleach_counts$bleaching)

p3 <- ggplot(combined_acc_bleach, aes(x = percent_samples, y = richness, color = bleaching, fill = bleaching)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.5, color = NA) +
  geom_line(aes(linetype = bleaching), linewidth = 1.2) +
  
  scale_color_manual(values = cols_bleaching, labels = bleach_labels) +
  scale_fill_manual(values = cols_bleaching, labels = bleach_labels) +
  scale_linetype_manual(
    values = c("Bleached" = "solid", 
               "Non-Bleached" = "solid", 
               "Not Applicable" = "solid"),
    labels = bleach_labels
  ) +
  
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "Percentage of Total Samples",
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
    legend.key.width = unit(1.2, "cm") 
  )

p3

################################################################################
# Symbiont potential
## 30 NAs for symbiont potential are removed in this plot

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

sym_counts <- df %>%
  group_by(symbiont.potential) %>%
  summarise(n = n()) %>%
  mutate(label_full = paste0(symbiont.potential, " (n = ", n, ")"))

sym_labels <- setNames(sym_counts$label_full, sym_counts$symbiont.potential)

# 3. Plot p4
p4 <- ggplot(combined_acc_sym, aes(x = percent_samples, y = richness, 
                                   color = symbiont.potential, fill = symbiont.potential)) +
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.5, color = NA) +
  geom_line(aes(linetype = symbiont.potential), linewidth = 1.2) +
  
  scale_color_manual(values = cols_symbiont, labels = sym_labels) +
  scale_fill_manual(values = cols_symbiont, labels = sym_labels) +
  
  scale_linetype_manual(
    values = c("Aposymbiotic" = "solid", 
               "Facultative" = "solid", 
               "Symbiotic" = "solid"),
    labels = sym_labels
  ) +
  
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "Percentage of Total Samples",
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

print(p4)

################################################################################
# Refined_origin path plot

## the next step is to use met_df which includes all the columns in comm_matrix
## met_df has a column called refined_origin which contains information about metabolic origin
## refined_origin has values "Host", "Symbiont" "Both" and "Unknown"as a factor in that order
## # of metabolites of each refined_origin

# cols_origin <- c("Host" = "#97B9CBFF", "Symbiont" = "#9057C6FF", 
#                  "Both" = "#FFE1BDFF", "Unknown" = "#8DC657FF")
# # 
# 
# met_df$refined_origin <- factor(met_df$refined_origin, 
#                                 levels = c("Host", "Symbiont", "Both", "Unknown"))
# 
# origin_counts <- met_df %>%
#   count(refined_origin, .drop = FALSE)
# 
# # compute the accumulation order using 'random' method
# acc_total <- specaccum(comm_matrix, method = "random", permutations = 30)
# 
# # extract the order of metabolite appearance
# # calculate richness per origin at each step of the accumulated samples.
# calc_stacked_acc <- function(comm, origins) {
#   n_samples <- nrow(comm)
#   # Randomize sample order
#   sample_order <- sample(1:n_samples)
#   
#   accumulated_data <- lapply(1:n_samples, function(i) {
#     # Subset to the first 'i' samples
#     sub_comm <- comm[sample_order[1:i], , drop = FALSE]
#     # Find which metabolites are present (colSum > 0)
#     present_mets <- colnames(sub_comm)[colSums(sub_comm) > 0]
#     # Count origins of present metabolites
#     data.frame(samples = i) %>%
#       bind_cols(
#         met_df %>%
#           filter(metabolite %in% present_mets) %>%
#           group_by(refined_origin) %>%
#           tally() %>%
#           pivot_wider(names_from = refined_origin, values_from = n, values_fill = 0)
#       )
#   })
#   bind_rows(accumulated_data)
# }
# 
# plot_data_stacked <- calc_stacked_acc(comm_matrix, met_df$refined_origin)
# plot_data_long <- plot_data_stacked %>%
#   pivot_longer(cols = -samples, names_to = "Origin", values_to = "Richness") %>%
#   mutate(
#     Origin = factor(Origin, levels = c("Host", "Symbiont", "Both", "Unknown")),
#     percent_samples = (samples / max(samples)) * 100
#   )
# 
# present_metabolites <- df %>% 
#   select(starts_with("x")) %>% 
#   colnames()
# 
# met_df_filtered <- met_df %>%
#   filter(met_df$metabolite %in% present_metabolites)
# 
# origin_counts <- met_df_filtered %>%
#   count(refined_origin, .drop = FALSE)
# 
# # print(origin_counts)
# 
# origin_labels <- origin_counts %>%
#   mutate(label_full = paste0(refined_origin, " (", n, " mets)" )) %>%
#   { setNames(.$label_full, .$refined_origin) }
# 
# p5 <- ggplot(plot_data_long, aes(x = percent_samples, y = Richness, fill = Origin)) +
#   geom_area(alpha = 0.7, color = "black", linewidth = 0.3) +
#   scale_fill_manual(
#     values = cols_origin, 
#     labels = origin_labels
#   ) +
#   
#   scale_x_continuous(expand = c(0, 0), breaks = seq(0, 100, 25)) +
#   scale_y_continuous(expand = c(0, 0)) +
#   labs(
#     x = "Percentage of Total Samples",
#     y = "Metabolite Richness",
#     fill = "Metabolic Origin"
#   ) +
#   theme_pubr() +
#   theme(
#     axis.title = element_text(),
#     legend.position = "right",
#     legend.title = element_text()
#   )
# p5

################################################################################


##  plot saturation of different symbiont genera
its2_df <- read.csv(here("Cleaned data CSVs", "ITS2Full.csv")) 
its2_df <- its2_df %>%
  filter(ITS2.Letter != "Mix" & ITS2.Letter != "No Seq" ) %>%
  select(-X.1, -X, -X.Location...as.character.Location..)

nrow(its2_df)

its2_df <- its2_df %>%
  mutate(
    ITS2.Letter = case_when(
      is.na(ITS2.Letter)           ~ NA_character_,
      str_detect(ITS2.Letter, ":") ~ "Mix",
      ITS2.Letter == "A" ~ "Symbiodinium",
      ITS2.Letter == "B" ~ "Breviolum",
      ITS2.Letter == "C" ~ "Cladocopium",
      ITS2.Letter == "D" ~ "Durusdinium",
      TRUE ~ ITS2.Letter   # keeps "Mix", "NoSeq", etc.
    )
  )

its2_palette <- c(
  "Breviolum"        = "#FF0000FF",
  "Cladocopium"      = "#00A08AFF",
  "Durusdinium"      = "#F2AD00FF",
  "Symbiodinium"  = "#5BBCD6FF",
  "Mix"              = "#F98400FF"
)

its2_levels <- c(
  "Symbiodinium",
  "Breviolum",
  "Cladocopium",
  "Durusdinium",
  "Mix"
)

its2_df <- its2_df %>%
  mutate(sample_id = paste0(sample, "_", row_number()))

comm_matrix <- its2_df %>%
  select(sample_id, starts_with("x")) %>%
  column_to_rownames(var = "sample_id")
comm_matrix[comm_matrix > 0] <- 1
genus_list <- split(its2_df$sample_id, its2_df$ITS2.Letter)

plot_data_its2 <- lapply(names(genus_list), function(genus_name) {
  # Subset matrix to only rows belonging to this genus
  comm_sub <- comm_matrix[rownames(comm_matrix) %in% genus_list[[genus_name]], ]
  
  if(nrow(comm_sub) < 2) return(NULL)
  acc <- specaccum(comm_sub, method = "random", permutations = 30)
  data.frame(
    percent_samples = (acc$sites / max(acc$sites)) * 100,
    richness = acc$richness,
    sd = acc$sd,
    ITS2.Letter = genus_name
  )
})

combined_acc_its2 <- bind_rows(plot_data_its2)

its2_counts <- its2_df %>%
  group_by(ITS2.Letter) %>%
  summarise(n = n()) %>%
  mutate(label_full = paste0(ITS2.Letter, " (n = ", n, ")")) %>%
  mutate(ITS2.Letter = factor(ITS2.Letter, levels = its2_levels)) %>%
  arrange(ITS2.Letter)

its2_labels <- setNames(its2_counts$label_full, its2_counts$ITS2.Letter)

p6 <- ggplot(combined_acc_its2, aes(x = percent_samples, y = richness, 
                                    color = ITS2.Letter, fill = ITS2.Letter)) +
  # Standard Deviation ribbon
  geom_ribbon(aes(ymin = richness - sd, ymax = richness + sd), alpha = 0.2, color = NA) +
  # Accumulation Line
  geom_line(aes(linetype = ITS2.Letter), linewidth = 1.2) +
  
  scale_color_manual(values = its2_palette, labels = its2_labels) +
  scale_fill_manual(values = its2_palette, labels = its2_labels) +
  scale_linetype_manual(values = rep("solid", length(its2_labels)), labels = its2_labels) +
  
  scale_x_continuous(breaks = seq(0, 100, 25)) +
  labs(
    x = "Percentage of Total Samples",
    y = "Metabolite Richness",
    color = "Symbiont Genus",
    fill = "Symbiont Genus",
    linetype = "Symbiont Genus"
  ) +
  theme_pubr() +
  theme(
    legend.position = c(0.98, 0.05),      
    legend.justification = c(1, 0),      
    legend.background = element_blank(), 
    legend.box.background = element_blank(),
    legend.key.width = unit(1.2, "cm") 
  )

print(p6)

################################################################################
# combine

top_row <- plot_grid(
  p, p2, 
  labels = c("A", "B"), 
  label_size = 20,
  label_x=0,
  label_y=1,
  hjust = 0,   
  vjust = 1.5,
  ncol = 2,
  rel_widths = c(1,0.8)
)
bottom_row <- plot_grid(
  p3, p4, p6, 
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
ggsave("/Users/henrysun_1/Desktop/Duke/PhD/coral/World_Corals/misc/figs/fig3_ITS2.jpg", final_plot, width=12,height=10,dpi=300)

################################################################################
