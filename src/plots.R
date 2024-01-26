# Used to plot model cancer incidence and SEER incidence for a specific birth cohort in the DES model

# Define working directory
setwd("/Users/francescalim/Documents/Documents - Francesca’s MacBook Pro/hire/multicancer_model/repo/HIRE-Multi-Cancer")

# Load packages
library(readxl)
library(tidyverse)

# Define parameters
birthyear <- 1960
cohort_type <- "am"
# Define SEER and age range
ages <- 25:70

# Get SEER target data
seer_incidence <- read.csv("data/cancer_incidence/1950_BC_All_Incidence.csv")
seer_incidence <- seer_incidence[,c(2,4)]
seer_incidence$Type <- "Target"
colnames(seer_incidence)[2] <- "Incidence"

# Get model data
model_incidence <- read_excel("outputs/phase1/all_cancer_incidence.xlsx")
colnames(model_incidence) <- c("Age", "Incidence")
model_incidence$Type <- "Model"

# Function to plot incidence
plot_cancer_incid <- function(model_incidence, target_incidence) {
  df <- rbind(target_incidence, model_incidence)
  ggplot(data=df, aes(x=Age, y=Incidence)) +
    geom_line(aes(color = Type))
}

# Save plots
incid_plot <- plot_cancer_incid(model_incidence, seer_incidence)
ggsave(paste('outputs/plots/', birthyear, "_all_cancer_incid_", cohort_type, ".png", sep=""),
       width = 7, height = 5)

