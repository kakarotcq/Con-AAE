library("Matrix")
library("Seurat")
library("hash")

ATAC<-read.csv("SNARE-ATAC.csv",header=T,row.names=1)
RNA<-read.csv("SNARE-RNA.csv",header=T,row.names=1)
label<-read.csv("CellLine_labels.csv",header=T,row.names=1)

ATAC<-as(as.matrix(ATAC),"dgCMatrix")
RNA<-as(as.matrix(RNA),"dgCMatrix")

ATAC.seu<-CreateSeuratObject(counts=ATAC,project="ATAC",min.cells=3,min.features=200)
RNA.seu<-CreateSeuratObject(counts=RNA,project="RNA",min.cells=3,min.features=200)

inte.list<-list(ATAC.seu,RNA.seu)

for(i in 1:length(inte.list)){
  #inte.list[[i]]<-NormalizeData(inte.list[[i]],verbose=FALSE)
  inte.list[[i]]<-FindVariableFeatures(inte.list[[i]],selection.method="vst",nfeatures=200,verbose=FALSE)
  inte.list[[i]]<-ScaleData(inte.list[[i]])
}

inte.anchors<-FindIntegrationAnchors(object.list=inte.list,dim=1:30)

tmp=inte.anchors@anchors


h=hash()
for(i in seq(1,dim(label)[1],by=1)){.set(h,keys=as.character(i),values=c(0,0))}
for(i in seq_along(tmp[,1])){
if(tmp[i,3]>h[[as.character(tmp[i,2])]][2])
{h[[as.character(tmp[i,2])]]=c(as.integer(tmp[i,1]),tmp[i,3])
}
}
count=0
for(i in seq(dim(label)[1])){
   if(h[[as.character(i)]][1]!=0){ 
     if(label[i,1]==label[as.integer(h[[as.character(i)]][1]),1])
         {count=count+1}
}
}

print(count/dim(label)[1])