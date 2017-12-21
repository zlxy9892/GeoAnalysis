library(ggplot2)
library(sp)
library(gstat)
library(raster)
library(rgdal)
library(spcosa)
library(spsann)

df<-read.csv(file="./data/2dcov.csv",header=TRUE)
head(df)

samples_kms = read.csv(file="./data/result/test6/kms_10.csv",header=TRUE)
samples_cLHS = read.csv(file="./data/result/test6/cLHS_10.csv",header=TRUE)

plot(density(df$cov1))
plot(density(df$cov2))

ggplot()+
  geom_point(mapping = aes(x=df$cov1, y=df$cov2))+
  #geom_density2d(mapping = aes(x=df$cov1, y=df$cov2))+
  #geom_point(mapping = aes(x=samples_kms$cov1, y=samples_kms$cov2), colour = "red")+
  geom_point(mapping = aes(x=samples_cLHS$cov1, y=samples_cLHS$cov2), colour = "green")
