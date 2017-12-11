#Author: Lei Zhang

#Functions for constraint kmeans algorithm

calcDist_Euclid <- function(v1, v2) {
  dist = sum((v1 - v2)^2)
  return(dist)
}

cKMS <- function(x, centers, priorPoint=NULL, nTry=1) {
  categories = numeric()
  for (tryTime in 1:nTry) {
    
  }
}
