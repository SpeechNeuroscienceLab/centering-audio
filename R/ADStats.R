library(nlme)
library(emmeans)

run <- "manual_exclude_2_by_group"
# Linear Mixed Effects Model
print(run)

ADtable <- read.csv(paste0("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/post-analysis/",
						   run, "/centering-analysis.csv"), header=TRUE, stringsAsFactors=FALSE)
ADLME <- lme(Centering ~ Group * Tercile, random = ~1 | Trial / Subject, data=ADtable,
			 cor=corSymm(form = ~1 | Trial / Subject))
anova(ADLME)
LSM <- lsmeans(ADLME, ~ Group * Tercile)
summary(pairs(LSM))

# Analysis of Variance: Kruskal-Wallis Test
# Test of Normality: Kolmogorov-Smirnov Test
groups <- unique(ADtable$Group)
for (group in groups) {
	group_data <- ADtable[ADtable$Group == group, ]
	print(group)
	print(kruskal.test(Centering ~ Tercile, data = group_data))
	print(ks.test(group_data$InitialPitch, "pnorm"))
	print(ks.test(group_data$EndingPitch, "pnorm"))
}

# Variance comparison test: F-test
var.test(ADtable[ADtable$Group == "AD Patients",]$InitialPitch,
				 ADtable[ADtable$Group == "AD Patients",]$EndingPitch,
				 alternative = "two.sided")

var.test(ADtable[ADtable$Group == "Controls",]$InitialPitch,
				 ADtable[ADtable$Group == "Controls",]$EndingPitch,
				 alternative = "two.sided")

# Variance comparison test: Anova
VarTable <- read.csv(paste0("/Users/anantajit/Documents/Research/UCSF/tables-and-figures/AD/post-analysis/",  run,
							"/peripheral_pitch_variance_table.csv", sep=""), header=TRUE, stringsAsFactors=FALSE)
VarAOV <- aov(PitchVariance ~ Group * SamplingTime, data=VarTable)
summary(VarAOV)
VarLSM <- lsmeans(VarAOV, ~ Group * SamplingTime)
summary(pairs(VarLSM))

