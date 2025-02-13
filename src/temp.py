import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_chart(fig, title, xlabel, ylabel, xangle=0):
    """Configura e exibe gráficos no Streamlit."""
    fig.update_layout(
        title={'text': title, 'x': 0.5, 'xanchor': 'center'},
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        template="plotly_white",
    )
    if xangle:
        fig.update_xaxes(tickangle=xangle)
    st.plotly_chart(fig, use_container_width=True)

def load_data(filepath, error_message):
    """Carrega dados de um arquivo CSV e exibe mensagem de erro caso não seja encontrado."""
    try:
        return pd.read_csv(filepath, sep=';')
    except FileNotFoundError:
        st.error(error_message)
        st.stop()

def aplicar_filtros(df):
    """Aplica filtros na base de dados conforme seleção do usuário."""
    st.sidebar.header("Filtros", divider='blue')
    idade_min, idade_max = st.sidebar.slider("Selecione a faixa de idade:", 5, 45, (5, 10))
    ano_min, ano_max = st.sidebar.slider(
        "Selecione o intervalo de anos:", df["Ano_Diagnostico"].min(), df["Ano_Diagnostico"].max(), 
        (df["Ano_Diagnostico"].min(), df["Ano_Diagnostico"].max())
    )
    regiao_filtro = st.sidebar.multiselect("Selecione a região:", df["Regiao"].unique(), default=df["Regiao"].unique())
    
    return df[(df["Idade"].between(idade_min, idade_max)) & 
              (df["Regiao"].isin(regiao_filtro)) & 
              (df["Ano_Diagnostico"].between(ano_min, ano_max))]

def main():
    # st.set_page_config(page_title="Dashboard de Análises", layout="wide")
    st.header("Análises e Visualizações", divider="blue")
    
    df = load_data("data/base_ajustada_realista_pre_processada.csv", "Arquivo não encontrado: Gere uma base de dados para análise")
    label_mappings = load_data("data/label_mappings_semicolon.csv", "Arquivo não encontrado: Gere uma base de dados para análise")
    dados_brutos = load_data("base_ajustada_realista.csv", "Arquivo não encontrado: Gere uma base de dados para análise")
    df_filtrado = aplicar_filtros(dados_brutos)
    
    if df_filtrado.empty:
        st.warning("Nenhum dado disponível para os filtros selecionados.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        diagnostico_counts = df_filtrado["Tipo_de_Diagnostico"].value_counts()
        fig1 = px.bar(
            x=diagnostico_counts.index, y=diagnostico_counts.values, color=diagnostico_counts.index,
            labels={"x": "Tipo de Diagnóstico", "y": "Contagem"}, template="plotly_white"
        )
        plot_chart(fig1, "Distribuição de Tipos de Diagnóstico", "Tipo de Diagnóstico", "Contagem")
    
    with col2:
        acesso_renda = df_filtrado.groupby(["Renda_Familiar", "Acesso_a_Servicos"]).size().unstack().reset_index()
        fig2 = px.bar(
            acesso_renda.melt(id_vars="Renda_Familiar", var_name="Acesso_a_Servicos", value_name="Contagem"),
            x="Renda_Familiar", y="Contagem", color="Acesso_a_Servicos", barmode="stack", template="plotly_white"
        )
        plot_chart(fig2, "Acesso a Serviços por Renda Familiar", "Renda Familiar", "Contagem")
    
    diagnostico_ano = df_filtrado.groupby("Ano_Diagnostico")["Idade"].count()
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=diagnostico_ano.index.astype(str), y=diagnostico_ano.values, mode="lines+markers", marker=dict(color="purple")
    ))
    plot_chart(fig3, "Evolução do Número de Diagnósticos", "Ano Diagnóstico", "Número de Diagnósticos")
    
if __name__ == "__main__":
    main()
