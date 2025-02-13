import pandas as pd
from sklearn.preprocessing import LabelEncoder   


def pre_processamento(df):

    # Padronizar os nomes das colunas (remover espaços e acentos)
    df.columns = (
        df.columns
        .str.strip()  # Remover espaços extras
        .str.replace(" ", "_")  # Substituir espaços por underscore
        .str.replace("á", "a")
        .str.replace("ã", "a")
        .str.replace("ç", "c")
        .str.replace("é", "e")
        .str.replace("í", "i")
        .str.replace("ó", "o")
        .str.replace("ú", "u")
    )

    # Inicializar o LabelEncoder
    le_dict = {}
    cols_to_convert = [
        'Tipo_de_Diagnostico', 'Tratamento', 'Acesso_a_Servicos',
        'Regiao', 'Apoio_Familiar', 'Renda_Familiar',
        'Tipo_de_Servico', 'Zona', 'Estado'
    ]

    # Criar um LabelEncoder para cada coluna e aplicar a transformação
    df_ajustado = df.copy()
    label_mappings = []

    for col in cols_to_convert:
        if col in df_ajustado.columns:
            le = LabelEncoder()
            df_ajustado[col] = le.fit_transform(df_ajustado[col])
            le_dict[col] = le

            # Criar um dataframe com os mapeamentos
            mapping_df = pd.DataFrame({
                'Coluna': col,
                'Valor_Original': le.classes_,
                'Valor_Transformado': le.transform(le.classes_)
            })
            label_mappings.append(mapping_df)

    # Concatenar os mapeamentos e salvar em CSV com separador ';'
    if label_mappings:
        label_mappings_df = pd.concat(label_mappings, ignore_index=True)
        mapping_file_path = "data/label_mappings_semicolon.csv"
        label_mappings_df.to_csv(mapping_file_path, index=False, sep=';')

    return df_ajustado




if __name__ == "__main__":
    df = pd.read_csv("base_ajustada_realista.csv", sep=';')
    df = pre_processamento(df)
    df.to_csv("data/base_ajustada_realista_pre_processada.csv", sep=';', index=False)
