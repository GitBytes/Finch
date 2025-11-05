import pandas as pd
import numpy as np
from functools import reduce

def get_component_embedding(compound: str, mole_frac: float, embeddings: pd.DataFrame) -> pd.Series:
    """
    Retrieve the embedding for a given compound and scale it by the mole fraction.
    
    args:
    - compound (str): The name of the compound.
    - mole_frac (float): The mole fraction of the compound.
    - embeddings (pd.DataFrame): DataFrame containing the embeddings for compounds.
    
    return:
    - pd.Series: Scaled embedding for the compound.
    
    """
    return embeddings.loc[compound] * mole_frac

def calculate_mixture_embedding(compound_names: list, mole_fractions: list, embeddings: pd.DataFrame) -> pd.Series:
    """
    Calculate the mixture embedding for a set of compounds and mole fractions.
    
    args:
    - compound_names (list): List of compound names.
    - mole_fractions (list): List of mole fractions corresponding to the compound names.
    - embeddings (pd.DataFrame): DataFrame containing the embeddings for compounds.
    
    return:
    - pd.Series: Embedding for the mixture.
    
    """
    component_embeddings = [
        get_component_embedding(compound, mole_frac / 100, embeddings)
        for compound, mole_frac in zip(compound_names, mole_fractions)
    ]
    return reduce(lambda x, y: x + y, component_embeddings)

def formulate_descriptors(mixture_info: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    """
    Formulate molecular descriptors for a given set of mixtures.
    
    args:
    - mixture_info (pd.DataFrame): DataFrame containing compound names and mole fractions for mixtures.
    - embeddings (pd.DataFrame): DataFrame containing the embeddings for compounds.
    
    return:
    - pd.DataFrame: DataFrame containing the embeddings for the mixtures.
    
    """
    mixture_embeddings = {
        index: calculate_mixture_embedding(row["Compound Names"], row["Mole Fractions"], embeddings)
        for index, row in mixture_info.iterrows()
    }
    
    return pd.DataFrame(mixture_embeddings).transpose()

if __name__ == "__main__":
    
    # Load the HepG2 mixture data for mapping mole fractions to embeddings
    mixture_info = pd.read_json('../finalized/mixture_data_for_mapping.json', orient='index')

    # Load the embeddings (or descriptors) of interest
    chemberta_embs = pd.read_csv("../data/KIM_chemberta_embeddings.csv", index_col=0)
    # OPTIONAL - only meant for this csv file - we only want the descriptors as columns and the compound name as index
    chemberta_embs = chemberta_embs[["compound"]+[ "emb_{}".format(i) for i in range(0,384) ]]
    chemberta_embs.set_index('compound', inplace=True)
    
    # Finally, calculate formula molecular descriptors
    mix_emb_df = formulate_descriptors(mixture_info, chemberta_embs)
    mix_emb_df.to_csv("mixture_embeddings.csv")
