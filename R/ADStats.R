library(nlme)
library(emmeans)

run <- "publication"
# Linear Mixed Effects Model
print(run)

ad_table <- read.csv(
    paste0("/Users/anantajit/Documents/UCSF/tables-and-figures/AD/",
    run, "/centering-analysis.csv"), header = TRUE, stringsAsFactors = FALSE)

# LME
adlme <- lme(Centering ~ Group * Tercile,
             random = ~1 | Subject / Trial,
             data = ad_table,
             cor =
               corSymm(form = ~ 1 | Subject / Trial)
             )

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
