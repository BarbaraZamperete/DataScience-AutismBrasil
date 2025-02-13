import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff


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
    

def main():
    st.header("Análises e Visualizações", divider="blue")

    """
    Base de dados pre-processada, onde os valores categorigos foram mapeados para valores numéricos
    """
    @st.cache_data
    def load_data_processado():
        try:
            df = pd.read_csv("data/base_ajustada_realista_pre_processada.csv", sep=';')
            # if not validate_data(df):
            #     st.stop()
            return df 
        except FileNotFoundError:
            st.error("**Arquivo não encontrado:** Por favor, gere uma base de dados para análise")
            st.stop()

    """
    Base de dados bruto, onde os valores categorigos ainda não foram mapeados para valores numéricos
    """
    @st.cache_data
    def load_data_bruto():
        try:
            df = pd.read_csv("base_ajustada_realista.csv", sep=';')
            # if not validate_data(df):
            #     st.stop()
            return df 
        except FileNotFoundError:
            st.error("**Arquivo não encontrado:** Por favor, gere uma base de dados para análise")
            st.stop()

    """
    Arquivo de mapeamento de valores numéricos para valores categorigos
    """
    @st.cache_data
    def load_label_mappings():
        try:
            df = pd.read_csv("data/label_mappings_semicolon.csv", sep=';')
            # if not validate_data(df):
            #     st.stop()
            return df 
        except FileNotFoundError:
            st.error("**Arquivo não encontrado:** Por favor, gere uma base de dados para análise")
            st.stop()

    df_processado = load_data_processado()
    label_mappings = load_label_mappings()
    df_bruto = load_data_bruto()

    # Mapeando os valores numéricos de volta para as labels originais
    for column in label_mappings.columns:
        if column in df_processado.columns:
            mapping = dict(zip(label_mappings[column + '_num'], label_mappings[column + '_label']))
            df_processado[column] = df_processado[column].map(mapping)
    
    
    # print(df_processado)

    st.sidebar.header("Filtros", divider='blue')

    idade_min, idade_max = st.sidebar.slider("**Selecione a faixa de idade:**", 5, 45, (5, 10))
    
    min_ano = df_bruto["Ano_Diagnostico"].min()
    max_ano = df_bruto["Ano_Diagnostico"].max()
    ano_min, ano_max = st.sidebar.slider("**Selecione o intervalo de anos:**", min_ano, max_ano, (min_ano, max_ano))
    
    regiao_filtro = st.sidebar.multiselect("**Selecione a região:**", df_bruto["Regiao"].unique(), default=df_bruto["Regiao"].unique())

    df_filtrado = df_bruto[
        (df_bruto["Idade"].between(idade_min, idade_max)) &
        (df_bruto["Regiao"].isin(regiao_filtro)) &
        (df_bruto["Ano_Diagnostico"]).between(ano_min, ano_max)
    ]


    if regiao_filtro:

        # Gráfico 1: Distribuição de Tipos de Diagnóstico
        diagnostico_counts = df_filtrado["Tipo_de_Diagnostico"].value_counts()
        fig1 = px.bar(
        x=diagnostico_counts.index,
        y=diagnostico_counts.values,
        color=diagnostico_counts.index,  # Opcional, para colorir as barras
        labels={"x": "Tipo de Diagnóstico", "y": "Contagem"},
        title="Distribuição de Tipos de Diagnóstico",
        template="plotly_white")
        
        plot_chart(fig1, "Distribuição de Tipos de Diagnóstico", "Tipo de Diagnóstico", "Contagem")
        st.write("""
        Este gráfico mostra a distribuição dos tipos de diagnóstico entre os pacientes. 
        A maioria dos diagnósticos é leve, seguida por moderados e, por último, os graves. 
        Essa distribuição pode indicar a prevalência de diagnósticos leves na população analisada.
        """)

        # Gráfico 3: Acesso a Serviços por Renda Familiar
        acesso_renda = df_filtrado.groupby(["Renda_Familiar", "Acesso_a_Servicos"]).size().unstack()
        acesso_renda = acesso_renda.reset_index()
        fig2 = px.bar(
            acesso_renda.melt(id_vars="Renda_Familiar", var_name="Acesso_a_Servicos", value_name="Contagem"),
            x="Renda_Familiar",
            y="Contagem",
            color="Acesso_a_Servicos",
            title="Acesso a Serviços por Renda Familiar",
            labels={"Renda_Familiar": "Renda Familiar", "Contagem": "Contagem"},
            template="plotly_white",
            barmode="stack"  # Empilhado
        )
        plot_chart(fig2, "Acesso a Serviços por Renda Familiar", "Renda Familiar", "Contagem")
        st.write("""
        Este gráfico de barras empilhadas mostra como o acesso a serviços de saúde varia de acordo com a renda familiar. 
        A maioria dos pacientes de renda alta tem acesso a serviços, enquanto os de renda baixa apresentam uma 
        proporção significativa sem acesso, o que pode indicar desigualdades no sistema de saúde.
        """)


        # Gráfico 4: Evolução do Número de Diagnósticos ao Longo dos Anos
        diagnostico_ano = df_filtrado.groupby("Ano_Diagnostico")["ID"].count()
        # fig, ax = plt.subplots(figsize=(8, 4))
        # sns.lineplot(
        #     x=diagnostico_ano.index.astype(str),
        #     y=diagnostico_ano.values,
        #     marker="o",
        #     color="purple",
        #     ax=ax
        # )
        

        # Criando o gráfico com Plotly
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=diagnostico_ano.index.astype(str),
            y=diagnostico_ano.values,
            mode="lines+markers",
            marker=dict(color="purple"),
            line=dict(color="purple")
        ))

        fig.update_layout(
            title="Evolução do Número de Diagnósticos ao Longo dos Anos",
            xaxis_title="Ano Diagnóstico",
            yaxis_title="Número de Diagnósticos",
            template="plotly_white"
        )

        plot_chart(fig, "Evolução do Número de Diagnósticos ao Longo dos Anos", "Ano Diagnóstico", "Número de Diagnósticos", rotate=0)
        st.write("""
        Este gráfico de linha mostra a evolução do número de diagnósticos ao longo dos anos. 
        É possível observar tendências de aumento ou diminuição nos diagnósticos, o que pode ser 
        influenciado por fatores como maior conscientização, mudanças nas políticas de saúde ou 
        melhorias na detecção precoce.
        """)


        # Gráfico 5: Distribuição de Tratamentos por Região
        tratamento_regiao = df_filtrado.groupby(["Regiao", "Tratamento"]).size().unstack().reset_index()
        fig = px.bar(
        tratamento_regiao.melt(id_vars="Regiao", var_name="Tratamento", value_name="Contagem"),
        x="Regiao",
        y="Contagem",
        color="Tratamento",
        title="Distribuição de Tratamentos por Região",
        labels={"Regiao": "Região", "Contagem": "Contagem"},
        template="plotly_white",
        barmode="stack"  # Empilhado
)
        plot_chart(fig, "Distribuição de Tratamentos por Região", "Região", "Contagem")
        st.write("""
        Este gráfico de barras empilhadas ilustra a distribuição dos tipos de tratamento recebidos pelos pacientes 
        em diferentes regiões do Brasil. A comparação entre regiões pode revelar desigualdades no acesso a 
        diferentes tipos de tratamento, o que é crucial para entender a eficácia das intervenções de saúde.
        """)

        consultas_ano = df_filtrado.groupby("Ano_Diagnostico")["Numero_de_Consultas"].sum()
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=consultas_ano.index,
            y=consultas_ano.values,
            mode="lines+markers",
            marker=dict(color="purple"),
            line=dict(color="purple")
        ))

        fig.update_layout(
            title="Evolução do Número de Consultas ao Longo dos Anos",
            xaxis_title="Ano",
            yaxis_title="Número de Consultas",
            template="plotly_white")
        
        plot_chart(fig, "Evolução do Número de Consultas ao Longo dos Anos", "Ano_Diagnostico", "Numero_de_Consultas")
        st.write("""
        Este gráfico de Evolução do Número de Consultas ao Longo dos Anos
        """)

        tratamento_diagnostico = df_filtrado.groupby(["Tipo_de_Diagnostico", "Tratamento"]).size().unstack().reset_index()
        fig = px.bar(
            tratamento_diagnostico.melt(id_vars="Tipo_de_Diagnostico", var_name="Tratamento", value_name="Contagem"),
            x="Tipo_de_Diagnostico",
            y="Contagem",
            color="Tratamento",
            title="Distribuição de Tratamentos por Diagnóstico",
            labels={"value": "Contagem", "variable": "Tratamento", "index": "Diagnóstico"},
            template="plotly_white",
            barmode="stack"
        )
        
        plot_chart(fig, "Distribuição de Tratamentos por Tipo de Diagnóstico", "Tipo_de_Diagnostico", "Contagem")
        st.write("""
        Distribuição de Tratamentos por Tipo de Diagnóstico
        """)
        
        """
        GRAFICO DE CORRELAÇÃO ENTRE AS VARIÁVEIS
        """

        st.subheader("🔍 Gráfico de Correlação entre Variáveis")
        correlation_matrix = df_processado.corr()
        # fig, ax = plt.subplots(figsize=(10, 8))
        # sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, square=True, cbar_kws={"shrink": .8})
        # ax.set_title("Matriz de Correlação", fontsize=16)
        # st.pyplot(fig)
        fig = ff.create_annotated_heatmap(
            z=correlation_matrix.values.round(2),  # Arredonda valores para melhor legibilidade
            x=correlation_matrix.columns.tolist(),
            y=correlation_matrix.index.tolist(),
            colorscale="RdBu",
            showscale=True,
            zmid=0  # Centraliza a escala de cores no zero para destacar correlações positivas e negativas
        )

        # Ajuste da aparência do gráfico
        fig.update_layout(
            title="Matriz de Correlação",
            width=800,
            # height=800,
            # margin=dict(l=100, r=100, t=50, b=100),  # Aumenta a margem inferior para acomodar as labels
        )
        fig.update_xaxes(
            tickangle=90,
            side="bottom",
            automargin=True
        )

        # Ajusta tamanho das anotações
        for annotation in fig.layout.annotations:
            annotation.font = dict(size=14)

        # Exibe o gráfico usando a função personalizada
        plot_chart(fig, "Matriz de Correlação", "Variáveis", "Variáveis")
        
        """
        
        FIM GRAFICO DE CORRELAÇÃO ENTRE AS VARIÁVEIS
        
        """


        # Gráfico 1: Apoio Familiar vs. Renda e Acesso a Serviços
        st.subheader("🔍 Apoio Familiar em Relação à Renda e Acesso a Serviços")
        apoio_renda = df_filtrado.groupby(['Apoio_Familiar', 'Renda_Familiar']).size().unstack().reset_index()
        fig1 = px.bar(apoio_renda.melt(id_vars=['Apoio_Familiar'], var_name='Renda_Familiar', value_name='Contagem'),
                    x='Renda_Familiar', y='Contagem', color='Apoio_Familiar', barmode='stack', 
                    color_discrete_sequence=px.colors.qualitative.Set2)
        plot_chart(fig1, "Apoio Familiar por Renda Familiar", "Renda Familiar", "Contagem", xangle=45)

        # Gráfico 2: Zona vs. Tipo de Serviço
        st.subheader("🔍 Zona em Relação ao Tipo de Serviço")
        zona_servico = df_filtrado.groupby(['Zona', 'Tipo_de_Servico']).size().unstack().reset_index()
        fig2 = px.bar(zona_servico.melt(id_vars=['Zona'], var_name='Tipo_de_Servico', value_name='Contagem'),
                    x='Zona', y='Contagem', color='Tipo_de_Servico', barmode='stack', 
                    color_discrete_sequence=px.colors.qualitative.Set3)
        plot_chart(fig2, "Tipo de Serviço por Zona", "Zona", "Contagem", xangle=45)

        # Gráfico 3: Ano de Diagnóstico vs. Número de Consultas
        st.subheader("🔍 Ano de Diagnóstico em Relação ao Número de Consultas")
        fig3 = px.scatter(df_filtrado, x="Ano_Diagnostico", y="Numero_de_Consultas", 
                        color="Tipo_de_Diagnostico", 
                        color_discrete_sequence=px.colors.qualitative.Set2, 
                        title="Ano de Diagnóstico vs. Número de Consultas")
        plot_chart(fig3, "Ano de Diagnóstico vs. Número de Consultas", "Ano de Diagnóstico", "Número de Consultas")

        # Gráfico 4: Tipo de Diagnóstico por Região
        st.subheader("🔍 Tipo de Diagnóstico por Região")
        diagnostico_regiao = df_filtrado.groupby(['Regiao', 'Tipo_de_Diagnostico']).size().unstack().reset_index()
        fig4 = px.bar(diagnostico_regiao.melt(id_vars=['Regiao'], var_name='Tipo_de_Diagnostico', value_name='Contagem'),
                    x='Regiao', y='Contagem', color='Tipo_de_Diagnostico', barmode='stack', 
                    color_discrete_sequence=px.colors.qualitative.Set1)
        plot_chart(fig4, "Tipo de Diagnóstico por Região", "Região", "Contagem", xangle=45)

    st.header("Dados Brutos")
    st.write("Os dados brutos são os dados originais que foram utilizados para a criação da base de dados.")
    st.write(df_bruto)

if __name__ == "__main__":
    main()