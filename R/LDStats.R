library(nlme)
library(emmeans)

run <- "publication"
# Linear Mixed Effects Model
print(run)

ad_table <- read.csv(
    paste0("/Users/anantajit/Documents/UCSF/results/LD/output_dataset.csv"), header = TRUE, stringsAsFactors = FALSE)

# Centering
adlme <- lme(Centering ~ Group * Tercile,
             random = ~1 | Subject / Trial,
             data = ad_table,
             cor = corSymm(form = ~ 1 | Subject / Trial))

anova(adlme)

lsm <- lsmeans(adlme, ~ Group * Tercile)
summary(pairs(lsm, adjust = "none"))

# Normalized Pitch Movement statistics 
print("Magnitude of Pitch Movement")

ad_table$MagnitudePitchMovement <- abs(ad_table$PitchMovement)

adlme <- lme(MagnitudePitchMovement ~ Group * Tercile,
             random = ~1 | Subject / Trial,
             data = ad_table,
             cor = corSymm(form = ~ 1 | Subject / Trial))

anova(adlme)

lsm <- lsmeans(adlme, ~ Group * Tercile)
summary(pairs(lsm, adjust = "none"))
