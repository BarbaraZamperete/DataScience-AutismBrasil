import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from src.ia.modelos import ModeloAutismo


def plot_chart(fig, title, xlabel, ylabel, xangle=0, show_legend=True, rotate=0):
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
        showlegend=show_legend
    )
    
    if xangle:
        fig.update_xaxes(tickangle=xangle)
    
    container = st.container(border=True)
    fig.update_xaxes(tickangle=rotate)
    container.plotly_chart(fig, use_container_width=True)

def validate_data(df):
    required_columns = ["Idade", "Ano_Diagnostico", "Regiao", "Tipo_de_Diagnostico", 
                        "Numero_de_Consultas", "Renda_Familiar", "Acesso_a_Servicos", 
                        "Tratamento"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"**Erro de validação:** As seguintes colunas estão ausentes: {', '.join(missing_columns)}")
        return False
    
    if df.isnull().values.any():
        st.warning("**Aviso:** O conjunto de dados contém valores nulos. Alguns gráficos podem ser afetados.")
    
    return True

def visualizar_graficos():
    st.header("📊 Visualizações e Análises", divider="blue")
    
    """
    Base de dados pre-processada, onde os valores categorigos foram mapeados para valores numéricos
    """
    @st.cache_data
    def load_data_processado():
        try:
            # Primeiro tenta carregar do diretório data/
            try:
                df = pd.read_csv("data/base_ajustada_realista_pre_processada.csv", sep=';')
                return df
            except FileNotFoundError:
                # Se não encontrar, tenta carregar do diretório raiz
                df = pd.read_csv("base_ajustada_realista_pre_processada.csv", sep=';')
                return df
        except FileNotFoundError:
            st.error("""
            **Arquivo não encontrado:** O arquivo 'base_ajustada_realista_pre_processada.csv' não foi encontrado.
            
            Por favor, verifique se o arquivo existe em uma das seguintes localizações:
            - data/base_ajustada_realista_pre_processada.csv
            - base_ajustada_realista_pre_processada.csv
            """)
            st.stop()

    df_processado = load_data_processado()

    st.sidebar.header("Filtros", divider='blue')

    idade_min, idade_max = st.sidebar.slider("**Selecione a faixa de idade:**", 5, 45, (5, 10))
    
    min_ano = df_processado["Ano_Diagnostico"].min()
    max_ano = df_processado["Ano_Diagnostico"].max()
    ano_min, ano_max = st.sidebar.slider("**Selecione o intervalo de anos:**", min_ano, max_ano, (min_ano, max_ano))
    
    regiao_filtro = st.sidebar.multiselect("**Selecione a região:**", df_processado["Regiao"].unique(), default=df_processado["Regiao"].unique())

    df_filtrado = df_processado[
        (df_processado["Idade"].between(idade_min, idade_max)) &
        (df_processado["Regiao"].isin(regiao_filtro)) &
        (df_processado["Ano_Diagnostico"]).between(ano_min, ano_max)
    ]

    if regiao_filtro:
        # Gráfico 1: Distribuição de Tipos de Diagnóstico
        diagnostico_counts = df_filtrado["Tipo_de_Diagnostico"].value_counts().reset_index()
        diagnostico_counts.columns = ["Tipo de Diagnóstico", "Contagem"]
        fig1 = px.bar(
            diagnostico_counts,
            x="Tipo de Diagnóstico",
            y="Contagem",
            color="Tipo de Diagnóstico"
        )
        plot_chart(fig1, "Distribuição de Tipos de Diagnóstico", "Tipo de Diagnóstico", "Contagem")
        st.write("""
        Este gráfico mostra a distribuição dos tipos de diagnóstico entre os pacientes. 
        A maioria dos diagnósticos é leve, seguida por moderados e, por último, os graves. 
        Essa distribuição pode indicar a prevalência de diagnósticos leves na população analisada.
        """)

        # Gráfico 2: Número de Consultas por Idade
        fig2 = px.scatter(
            df_filtrado,
            x="Idade",
            y="Numero_de_Consultas",
            color="Tipo_de_Diagnostico"
        )
        plot_chart(fig2, "Número de Consultas por Idade", "Idade", "Número de Consultas", show_legend=False)
        st.write("""
        Este gráfico de dispersão mostra a relação entre a idade dos pacientes e o número de consultas realizadas.
        Os pontos são coloridos de acordo com o tipo de diagnóstico, permitindo visualizar se há padrões 
        específicos de consultas para diferentes tipos de diagnóstico em diferentes faixas etárias.
        """)

        # Gráfico 3: Acesso a Serviços por Renda Familiar
        acesso_renda = df_filtrado.groupby(["Renda_Familiar", "Acesso_a_Servicos"]).size().reset_index(name="Contagem")
        fig3 = px.bar(
            acesso_renda,
            x="Renda_Familiar",
            y="Contagem",
            color="Acesso_a_Servicos",
            barmode="stack"
        )
        plot_chart(fig3, "Acesso a Serviços por Renda Familiar", "Renda Familiar", "Contagem")
        st.write("""
        Este gráfico de barras empilhadas mostra como o acesso a serviços de saúde varia de acordo com a renda familiar. 
        A maioria dos pacientes de renda alta tem acesso a serviços, enquanto os de renda baixa apresentam uma 
        proporção significativa sem acesso, o que pode indicar desigualdades no sistema de saúde.
        """)

        # Gráfico 4: Evolução do Número de Diagnósticos ao Longo dos Anos
        diagnostico_ano = df_filtrado.groupby("Ano_Diagnostico")["ID"].count().reset_index(name="Número de Diagnósticos")
        diagnostico_ano["Ano_Diagnostico"] = diagnostico_ano["Ano_Diagnostico"].astype(str)
        fig4 = px.line(
            diagnostico_ano,
            x="Ano_Diagnostico",
            y="Número de Diagnósticos",
            markers=True
        )
        plot_chart(fig4, "Evolução do Número de Diagnósticos ao Longo dos Anos", "Ano Diagnóstico", "Número de Diagnósticos", xangle=45)
        st.write("""
        Este gráfico de linha mostra a evolução do número de diagnósticos ao longo dos anos. 
        É possível observar tendências de aumento ou diminuição nos diagnósticos, o que pode ser 
        influenciado por fatores como maior conscientização, mudanças nas políticas de saúde ou 
        melhorias na detecção precoce.
        """)

        # Gráfico 5: Distribuição de Tratamentos por Região
        tratamento_regiao = df_filtrado.groupby(["Regiao", "Tratamento"]).size().reset_index(name="Contagem")
        fig5 = px.bar(
            tratamento_regiao,
            x="Regiao",
            y="Contagem",
            color="Tratamento",
            barmode="stack"
        )
        plot_chart(fig5, "Distribuição de Tratamentos por Região", "Região", "Contagem")
        st.write("""
        Este gráfico de barras empilhadas ilustra a distribuição dos tipos de tratamento recebidos pelos pacientes 
        em diferentes regiões do Brasil. A comparação entre regiões pode revelar desigualdades no acesso a 
        diferentes tipos de tratamento, o que é crucial para entender a eficácia das intervenções de saúde.
        """)

    else:
        st.info('*Selecione ao menos uma região para visualização dos gráficos*')


if __name__ == "__main__":
    visualizar_graficos()