library(nlme)
library(lsmeans)
run = "manual_exclude_2_by_group"
ADtable <- read.csv(paste("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/post-analysis/",  run, "/centering-analysis.csv", sep=""), header=TRUE, stringsAsFactors=FALSE)
ADLME <- lme(Centering ~ Group * Tercile, random = ~1|Subject, data=ADtable, cor=corCompSymm())
anova(ADLME)
LSM <- lsmeans(ADLME, ~ Group * Tercile)
summary(pairs(LSM))

# Analysis of Variance: Kruskal-Wallis Test
# Test of Normality: Kolmogorov-Smirnov Test
groups = unique(ADtable$Group)
for (group in groups) {
	group_data <- ADtable[ADtable$Group == group, ]
	print(group)
	print(kruskal.test(Centering ~ Tercile, data = group_data))
	print(ks.test(group_data$InitialPitch, "pnorm"))
	print(ks.test(group_data$EndingPitch, "pnorm"))
}
