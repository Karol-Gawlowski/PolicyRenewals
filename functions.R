# Policy_channel_helper=function(channels=group_1,x){
#   
#   return(sum((x==channels)*1 %*% channels))
#   
# }

Normalize=function(x){return((x-min(x))/(max(x)-min(x)))}

# in to out
'%ni%' <- Negate('%in%')

FactorToInteger=function(x){return(as.integer(as.character(x)))}