library(nlme)
library(emmeans)

run <- "publication"
# Linear Mixed Effects Model
print(run)

ad_table <- read.csv(
    paste0("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/",
    run, "/centering-analysis.csv"), header = TRUE, stringsAsFactors = FALSE)

# LME
adlme <- lme(Centering ~ Group * Tercile,
             random = ~1 | Subject / Trial,
             data = ad_table,
             cor = corSymm(form = ~ 1 | Subject / Trial))

anova(adlme)

lsm <- lsmeans(adlme, ~ Group * Tercile)
summary(pairs(lsm, adjust = "none"))

# Analysis of Variance: Kruskal-Wallis Test
# Test of Normality: Kolmogorov-Smirnov Test
groups <- unique(ad_table$Group)
for (group in groups) {
    group_data <- ad_table[ad_table$Group == group, ]
    print(group)
    print(kruskal.test(Centering ~ Tercile, data = group_data))
    print(ks.test(group_data$InitialPitch, "pnorm"))
    print(ks.test(group_data$EndingPitch, "pnorm"))
}

print("Variance comparison tests:")

# Variance comparison test: F-test
var.test(ad_table[ad_table$Group == "AD Patients", ]$InitialPitch,
         ad_table[ad_table$Group == "AD Patients", ]$EndingPitch,
         alternative = "two.sided")

var.test(ad_table[ad_table$Group == "Controls", ]$InitialPitch,
         ad_table[ad_table$Group == "Controls", ]$EndingPitch,
         alternative = "two.sided")

# Variance comparison test: Anova
var_table <- read.csv(paste0(
"/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/",
run,
"/pitch_variance_table.csv", sep = "")
, header = TRUE, stringsAsFactors = FALSE)

var_aov <- aov(PitchVariance ~ Group * SamplingTime, data = var_table)
summary(var_aov)
var_lsm <- lsmeans(var_aov, ~ Group * SamplingTime)
summary(pairs(var_lsm))

# Show that the initial pitch is comparable
# (no significant differences across groups)
ad_table$InitialDeviation <- abs(ad_table$InitialPitch)
print("Initial Pitch Comparisons")
initial_pitch_lme <- lme(InitialDeviation ~ Group * Tercile,
             random = ~1 | Subject / Trial,
             data = ad_table,
             cor = corSymm(form = ~ 1 | Subject / Trial))
anova(initial_pitch_lme)

initial_pitch_lsm <- lsmeans(initial_pitch_lme, ~ Group * Tercile)
summary(pairs(initial_pitch_lsm, adjust = "none"))

# Centering in peripheral vs central trials
all_ad_table <- read.csv(
    paste0("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/",
    run, "/all-terciles-centering-analysis.csv"
    ), header = TRUE, stringsAsFactors = FALSE)
all_ad_table$Tercile <- ifelse(all_ad_table$Tercile %in% c("LOWER", "UPPER"),
                               "PERI", all_ad_table$Tercile)

print("Central vs Peripheral Centering")
central_peripheral_lme <- lme(Centering ~ Group * Tercile,
                              random = ~1 | Subject / Trial,
                              data = all_ad_table,
                              cor = corSymm(form = ~ 1 | Subject / Trial))
anova(central_peripheral_lme)

central_peripheral_lsm <- lsmeans(central_peripheral_lme, ~ Group * Tercile)
summary(pairs(central_peripheral_lsm, adjust = "none"))

# Pitch Movement statistics 
print("Pitch Movement (unnormalized)")
ad_table <- read.csv(
    paste0("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/",
    run, "/pitch-movement-table.csv"), header = TRUE, stringsAsFactors = FALSE)

# LME
adlme <- lme(PitchMovement ~ Group * Tercile,
             random = ~1 | Subject / Trial,
             data = ad_table,
             cor = corSymm(form = ~ 1 | Subject / Trial))

anova(adlme)

lsm <- lsmeans(adlme, ~ Group * Tercile)
summary(pairs(lsm, adjust = "none"))

# Normalized Pitch Movement statistics 
print("Normalized Pitch Movement")
ad_table <- read.csv(
    paste0("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/",
    run, "/pitch-movement-table.csv"), header = TRUE, stringsAsFactors = FALSE)

# LME
adlme <- lme(NormPitchMovement ~ Group * Tercile,
             random = ~1 | Subject / Trial,
             data = ad_table,
             cor = corSymm(form = ~ 1 | Subject / Trial))

anova(adlme)

lsm <- lsmeans(adlme, ~ Group * Tercile)
summary(pairs(lsm, adjust = "none"))
