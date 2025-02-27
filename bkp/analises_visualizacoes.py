import streamlit as st
import pandas as pd
import plotly.express as px


def plot_chart(fig, title, xlabel, ylabel, xangle=0, show_legend=True):
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
    container.plotly_chart(fig, use_container_width=True)


def validate_data(df):
    required_columns = ["Idade", "Ano Diagnóstico", "Região", "Tipo de Diagnóstico", 
                        "Número de Consultas", "Renda Familiar", "Acesso a Serviços", 
                        "Tratamento"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"**Erro de validação:** As seguintes colunas estão ausentes: {', '.join(missing_columns)}")
        return False
    
    if df.isnull().values.any():
        st.warning("**Aviso:** O conjunto de dados contém valores nulos. Alguns gráficos podem ser afetados.")
    
    return True


def main():
    st.header("Análises e Visualizações", divider="blue")
    
    # @st.cache_data
    def load_data():
        try:
            df = pd.read_csv("base_ajustada_realista.csv", sep=';')
            if not validate_data(df):
                st.stop()
            return df
        except FileNotFoundError:
            st.error("**Arquivo não encontrado:** Por favor, gere uma base de dados para análise")
            st.stop()
    
    df = load_data()
    
    st.sidebar.header("Filtros", divider='blue')
    idade_min, idade_max = st.sidebar.slider("**Selecione a faixa de idade:**", 5, 45, (5, 10))
    
    min_ano = int(df["Ano Diagnóstico"].min())
    max_ano = int(df["Ano Diagnóstico"].max())
    ano_min, ano_max = st.sidebar.slider("**Selecione o intervalo de anos:**", min_ano, max_ano, (min_ano, max_ano))
    
    regiao_filtro = st.sidebar.multiselect("**Selecione a região:**", df["Região"].unique(), default=df["Região"].unique())
    
    df_filtrado = df[
        (df["Idade"].between(idade_min, idade_max)) &
        (df["Região"].isin(regiao_filtro)) &
        (df["Ano Diagnóstico"].between(ano_min, ano_max))
    ]
    
    if regiao_filtro:

        # 1. Distribuição de Tipos de Diagnóstico (Gráfico de Barras)
        diagnostico_counts = df_filtrado["Tipo de Diagnóstico"].value_counts().reset_index()
        diagnostico_counts.columns = ["Tipo de Diagnóstico", "Contagem"]
        fig1 = px.bar(
            diagnostico_counts,
            x="Tipo de Diagnóstico",
            y="Contagem",
            color="Tipo de Diagnóstico",
            # labels={"Tipo de Diagnóstico": "Tipo de Diagnóstico", "Contagem": "Contagem"},
        )
        plot_chart(fig1, "Distribuição de Tipos de Diagnóstico", "Tipo de Diagnóstico", "Contagem")


        # 2. Número de Consultas por Idade (Scatter Plot)
        fig2 = px.scatter(
            df_filtrado,
            x="Idade",
            y="Número de Consultas",
            color="Tipo de Diagnóstico",
            # labels={"Idade": "Idade", "Número de Consultas": "Número de Consultas"},
        )
        plot_chart(fig2, "Número de Consultas por Idade", "Idade", "Número de Consultas", show_legend=False)


        # 3. Acesso a Serviços por Renda Familiar (Gráfico de Barras Empilhadas)
        acesso_renda = (
            df_filtrado
            .groupby(["Renda Familiar", "Acesso a Serviços"])
            .size()
            .reset_index(name="Contagem")
        )
        fig3 = px.bar(
            acesso_renda,
            x="Renda Familiar",
            y="Contagem",
            color="Acesso a Serviços",
            # labels={"Renda Familiar": "Renda Familiar", "Contagem": "Contagem"},
            barmode="stack"
        )
        plot_chart(fig3, "Acesso a Serviços por Renda Familiar", "Renda Familiar", "Contagem")


        # 4. Evolução do Número de Diagnósticos ao Longo dos Anos (Line Plot)
        diagnostico_ano = (
            df_filtrado
            .groupby("Ano Diagnóstico")["ID"]
            .count()
            .reset_index(name="Número de Diagnósticos")
        )
        diagnostico_ano["Ano Diagnóstico"] = diagnostico_ano["Ano Diagnóstico"].astype(str)
        fig4 = px.line(
            diagnostico_ano,
            x="Ano Diagnóstico",
            y="Número de Diagnósticos",
            markers=True,
            # labels={"Ano Diagnóstico": "Ano Diagnóstico", "Número de Diagnósticos": "Número de Diagnósticos"}
        )
        plot_chart(fig4, "Evolução do Número de Diagnósticos ao Longo dos Anos", "Ano Diagnóstico", "Número de Diagnósticos", xangle=45)


        # 5. Distribuição de Tratamentos por Região (Gráfico de Barras Empilhadas)
        tratamento_regiao = (
            df_filtrado
            .groupby(["Região", "Tratamento"])
            .size()
            .reset_index(name="Contagem")
        )
        fig5 = px.bar(
            tratamento_regiao,
            x="Região",
            y="Contagem",
            color="Tratamento",
            # labels={"Região": "Região", "Contagem": "Contagem"},
            barmode="stack"
        )
        plot_chart(fig5, "Distribuição de Tratamentos por Região", "Região", "Contagem")
    else:
        st.info('*Selecione ao menos uma região para visualização dos gráficos*')
    
    # Exibição dos dados brutos
    st.header("Dados Brutos", divider='blue')
    st.write(df)