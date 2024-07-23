dataprep.out <- dataprep(
	foo=basque,
	predictors = c("school.illit", "school.prim", "school.med", "school.high", "school.post.high", "invest"),
			predictors.op = "mean",
			time.predictors.prior = 1964:1969,
			special.predictors = list(
				list("gdpcap", 1960:1969, "mean"),
				list("sec.agriculture", seq(1961, 1969, 2), "mean"),
				list("sec.energy",seq(1961, 1969, 2), "mean"),
				list("sec.industry", seq(1961, 1969, 2), "mean"),
				list("sec.construction", seq(1961, 1969, 2), "mean"),
				list("sec.services.venta", seq(1961, 1969, 2), "mean"),
				list("sec.services.nonventa", seq(1961, 1969, 2), "mean"),
				list("popdens", 1969, "mean")),
			dependent = 'gdpcap',
			unit.variable="regionno",
			unit.names.variable="regionname",
			time.variable= "year",
			treatment.identifier=17,
			controls.identifier= c(2:16, 18),
			time.optimize.ssr = 1960:1969,
			time.plot= 1955:1997)

synth.out <- synth(data.prep.obj = dataprep.out, method= "BFGS")

path.plot(synth.res = synth.out,
		dataprep.res = dataprep.out,
		Ylab = "real per-capita GDP(1986 USD, thousand)",
		Xlab = "year",
		Ylim = c(0,12),
		Legend = c("Basque country", "synthetic Basque country"),
		Legend.position = "bottomright")

gaps.plot(synth.res = synth.out,
		dataprep.res = dataprep.out,
		Ylab = "gap in real per-capita GDP (1986 USD, thousand)",
		Xlab = "year",
		Ylim = c(-1.5, 1.5),
		Main = NA)

tdf <- generate.placebos(dataprep.out, synth.out)
plot.placebos(tdf=tdf, discard.etreme=True, mspe.limit=5)


###
data_causal <- read.csv("~/book_use.csv")
target_list <- c(1.5)
y <- "PM2.5"
time <- "ymd"
data_causal$station_id <- as.numeric(data_causal$station)
data_causal$ymd <- as.Date(data_causal$ymd)
data_split <- data_split(data_causal, target_list, y, time)

data_target <- data_split$data_target
data_pool <- data_split$data_pool

result_target <-pivot(data_in=data_target, null_replace=NA)
pivot_data_target<-result_target$pivot_data

result_pool <- pivot(data_in=data_pool, null_replace=NA)
pivot_data_pool <- result_pool$pivot_data

for(i in 2:dim(pivot_data_target)[2]){
	for (j in 1:dim(pivot_data_target)[1]){
		pivot_data_target_use[j, i] <- ifelse(is.na(pivot_data_target[j,i])==TRUE,
			mean(pivot_data_target[,i], na.rm=TRUE), pivot_data_target[j,i])
	}
}

for(i in 2:dim(pivot_data_pool)[2]){
	for (j in 1:dim(pivot_data_pool)[1]){
		pivot_data_pool_use[j, i] <- ifelse(is.na(pivot_data_pool[j,i])==TRUE,
			mean(pivot_data_pool[,i], na.rm=TRUE), pivot_data_pool[j,i])
	}
}

pivot_data_target_use <- pivot_data_target
pivot_data_pool_use <- pivot_data_pool

pivot_data_target_use[which(months(as.Date(pivot_data_target_use$date_vector))=="July"),
		c("station_1", "station_5")] <- pivot_data_target_use[which(months(as.Date(pivot_data_target_use$date_vector))=="July"),
		c("station_1", "station_5")] + rnorm(n=31, mean=5, sd=3.75)

pivot_data_target[which(months(as.Date(pivot_data_target$date_vector))=="July"),
		c("station_1", "station_5")] <- pivot_data_target[which(months(as.Date(pivot_data_target$date_vector))=="July"),
		c("station_1", "station_5")] + rnorm(n=31, mean=5, sd=3.75)

data_use <- cbind(y=pivot_data_target_use[,c("station_1")], pivot_data_pool[,c(2:dim(pivot_data_pool_use)[2])])



