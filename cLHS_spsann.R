library(sp)
library(spsann)
library(reshape)
library(ggplot2)

#Read data

grid <- read.csv(file="HunterValley4Practicals.csv")
head(grid)

#Grid has too many points (memory problems hereafter), so a subgrid is selected.
#A problem is that the spacing of the grid is not exactly constant, which causes problems with typecasting the SpatialPointsDataFrame to a SpatialPixelsDataFrame.
#This problem is solved by rounding the coordinates.

grid$Easting <- 25*round(grid$Easting/25,0)
grid$Northing <- 25*round(grid$Northing/25,0)
coordinates(grid) <- c("Easting","Northing")
gridded(grid) <- T
subgrid <- spsample(grid,type="regular",cellsize=50,offset=c(0.5,0.5))
subgriddata <- (subgrid %over% grid)
grd <- data.frame(coordinates(subgrid),subgriddata)

#Specify the candidate sampling points and the covariates to be used in conditioned Latin Hypercube Sampling

candi <- grd[,1:2]
names(candi) <- c("x","y")
covars <- grd[, 3:7]

#Define the schedule for simulated annealing.

#Note that both the initial acceptance rate  and the initial temperature are set,
#which may seem weird as the acceptance rate is a function of the initial temperature: $P =e^{\frac{-\Delta f}{T}}$. 
#The initial acceptance rate is used as a threshold value. If an initial temperature is chosen that leads to an acceptance rate smaller than the chosen value for the initial acceptance rate, then the optimization stops. In this case a larger value for the initial temperature must be chosen.

schedule <- scheduleSPSANN(initial.acceptance = 0.8,initial.temperature = 0.02,
                           temperature.decrease=0.95,
                           chains=1000,
                           chain.length=4,
                           stopping=10,
                           x.min=10,y.min=10,
                           cellsize=50)

#Now start the simulated annealing algorithm.

weights <- list(O1 = 0.5, O3 = 0.5)
samplesize<-50
set.seed(314)
res <- optimCLHS(
  points = samplesize, candi = candi, covars = covars, use.coords = FALSE, 
  schedule = schedule, track=TRUE, weights = weights, progress=NULL)

#Compute number of points in marginal strata

by=1/samplesize
probs<-seq(from=0,to=1,by=by)
last<-length(probs)
p <- 5
lb<-matrix(nrow=length(probs)-1,ncol=p)
for (i in 1:p) {
  q<-quantile(grd[,i+2],probs=probs,type=7)
  lb[,i]<-q[-last]
}

mySample <- res$points
mySample <- data.frame(mySample,grd[mySample$id,3:7])

stratum<-matrix(nrow=nrow(mySample),ncol=p)
for ( i in 1:p) {
  stratum[,i]<-findInterval(mySample[,i+3],lb[,i])
}

counts<-matrix(nrow=nrow(lb),ncol=p)
for (i in 1:nrow(lb)) {
  counts[i,]<-apply(stratum, MARGIN=2, function(x,i) sum(x==i), i=i)
}

#Compute O1 criterion (not standardized)

(sum(abs(counts-1)))

#Plot the optimized sample in geographical space.

ggplot(data=grd) +
    geom_tile(mapping = aes(x = x1, y = x2, fill = cti))+  
    geom_point(data = mySample, mapping = aes(x = x, y = y), colour = "black") +
    scale_x_continuous(name = "Easting (km)") +
    scale_y_continuous(name = "Northing (km)") +    
    scale_fill_gradient(name="cti",low = "darkblue", high = "red")+
    coord_fixed()

#Plot the optimized sample in covariate space by making scatter plots

ggplot(data=grd) +
        geom_point(mapping = aes(y = elevation_m, x = cti), colour = "black",size=1,alpha=0.5) +
        geom_point(data=mySample, mapping = aes(y = elevation_m, x = cti), colour = "red",size=2) +
        scale_y_continuous(name = "Elevation") +
        scale_x_continuous(name = "CTI")

#Plot the sample sizes in the marginal strata

index<-seq(1:samplesize)
countsdf<-as.data.frame(counts)
names(countsdf)<-names(grd[c(3,4,5,6,7)])

countslf<-melt(countsdf)
countslf$index<-rep(index,times=5)
ggplot(countslf) +
        geom_point(mapping = aes(x=index,y = value), colour = "black",size=1) +
        facet_wrap(~variable) +
        scale_x_continuous(name = "Stratum") +
        scale_y_continuous(name = "Sample size",breaks=c(0,1,2,3,4))