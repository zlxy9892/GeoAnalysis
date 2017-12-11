library(sp)
library(fields)
library(ggplot2)

#Read data with coordinates and other attributes of fine grid (discretization of study area)

dat <- read.csv(file="./data/envdata_raf.csv")
summary(dat)
feat = c(4,5,6,7,8,9)
cor(dat[,feat])

#Set number of sampling locations to be selected

n<-50

#Compute clusters

set.seed(314)
myClusters <- kmeans(scale(dat[,c(3,4,5,6,7)]), centers=n, iter.max=100,nstart=10)
dat$clusters <- myClusters$cluster

#Select locations closest to the centers of the clusters

rdist.out <- rdist(x1=myClusters$centers,x2=scale(dat[,feat]))
ids.mindist <- apply(rdist.out,MARGIN=1,which.min)
mySample <- dat[ids.mindist,]
print(mySample[,c('x','y')])

#Plot clusters and sampling points

ggplot(dat) +
  geom_tile(mapping = aes(x = 25*round(x/25,0), y = 25*round(y/25,0), fill = factor(clusters))) +
  scale_fill_discrete(name = "cluster") +
  geom_point(data=mySample,mapping=aes(x=x,y=y),size=2) +
  scale_x_continuous(name = "") +
  scale_y_continuous(name = "") +
  coord_fixed() +
  theme(legend.position="none")

ggplot(dat) +
  geom_point(mapping=aes(y=env2,x=env6,colour=factor(clusters))) +
  geom_point(data=mySample,mapping=aes(y=env2,x=env6),size=2) +
  scale_x_continuous(name = "Elevation") +
  scale_y_continuous(name = "Twi") +
  theme(legend.position="none")
