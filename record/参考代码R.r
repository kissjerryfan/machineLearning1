##lasso regression
#    lxx<-as.matrix(trainData2)
#    lyy<-as.matrix(trainLabel2)
#    lassomodel<-lars(lxx,lyy,trace=TRUE) # type缺省是lasso
#    optlambda1<-lassomodel$lambda[which.min(lassomodel$Cp)-1]
#    ytrainlasso1<-predict(lassomodel,s=optlambda1, trainData2,type="fit",mode="lambda")$fit##predicted value
#    ytrainlasso1<-ifelse(ytrainlasso1>0.5,1,-1);
#    ytestlasso1<-predict(lassomodel,s=optlambda1, testData,type="fit",mode="lambda")$fit##predicted value
#    ytestlasso1<-ifelse(ytestlasso1>0.5,1,-1);
#    lartestAcc<-lartestAcc+Acc(ytestlasso1,testLabel);
#    lartrainAcc<-lartrainAcc+Acc(ytrainlasso1,trainLabel);
、
ridge.sol<- lm.ridge(trainLabel2 ~ ., lambda=seq(0,1000,0.1), data=trainData2,model=TRUE) 
#    optlambda<-ridge.sol$lambda[which.min(ridge.sol$GCV)]
#    ridgemodel<-linearRidg(trainLabel2 ~ .,lambda=optlambda, data=trainData2)