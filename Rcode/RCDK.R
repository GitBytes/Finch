# load the library
library(rcdk)
library(readr)
library(mlbench)
library(caret)
library(dplyr)
library(pROC)

#ensure the results are repeatable
set.seed(42)

##READ IN SINGLE INGREDIENTS SMILES, CONCENTRATION, AND CYTOTOXICITY
#READ In compound name and SMILES then calculate the Molecular Descriptors
Ingredient_Name_SMILES <- read_csv("kim_single_ingredient.csv")

##CREATE MOLECULAR DESCRIPTORS
#Calculate the Molecular Descriptors for the Mixture components using RCDK
smiles <- as.character(Ingredient_Name_SMILES$smiles)  #this makes a character string 
mols <- parse.smiles(smiles) #use rcdk to check and rewrite smiles string
#make a list of all descriptor names in each descriptor category from the rcdk package
descriptor_names <- unique(unlist(sapply(get.desc.categories(), get.desc.names)))
#take smiles and evaluate the molecular descriptors using rcdk for all the descriptors
molecular_descriptors <- eval.desc(mols,descriptor_names)
#clean up the molecular descriptor list by removing all NA
red_mol_desc <- molecular_descriptors[, !apply(molecular_descriptors, 2, function(x) any(is.na(x)))]
red_mol_desc <- red_mol_desc[, !apply(red_mol_desc, 2, function(x) length(unique(x))==1)]
molecular_descriptors <- red_mol_desc #save cleaned molecular descriptors
mdnumrows <<- nrow(molecular_descriptors) #calculate the number of rows in molecular descriptors
mdnumcol <<- ncol(molecular_descriptors) #calculate the number of columns in molecular descriptors

#Normalize molecular descriptors before selecting features
maxs <<- apply(red_mol_desc, 2, max) #find the max in each column
mins <<- apply(red_mol_desc, 2, min) #find the min in each column
scaled_molecular_descriptors <<- (scale(molecular_descriptors, center = mins, scale = maxs - mins)) #normalize the descriptor values

#Combine concentration and cytotoxicity to descriptors.
Data <<- cbind(Ingredient_Name_SMILES$compound,scaled_molecular_descriptors) #combine names and scaled molecular descriptors
colnames(Data)[1] <- "compound" #rename the column
Data_df <<- data.frame(Data) #make a data frame

#Write out the RCDK
write.csv(Data_df,"KIM_rcdk_descriptors.csv",row.names=FALSE)
