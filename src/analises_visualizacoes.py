import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from src.ia.modelos import ModeloAutismo, treinar_e_salvar_modelos


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
    st.header("Análises, Visualizações e Predições", divider="blue")
    
    tab1, tab2, tab3 = st.tabs(["📊 Visualizações", "🤖 Predições", "📑 Dados Brutos"])

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
        
        # Identificar colunas numéricas
        colunas_numericas = df_processado.select_dtypes(include=['int64', 'float64']).columns
        
        if len(colunas_numericas) > 0:
            # Calcular correlação apenas para colunas numéricas
            correlation_matrix = df_processado[colunas_numericas].corr()
            
            # Criar heatmap com plotly
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
                title="Matriz de Correlação (Variáveis Numéricas)",
                width=800,
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
            plot_chart(fig, "Matriz de Correlação (Variáveis Numéricas)", "Variáveis", "Variáveis")
            
            # Adicionar explicação
            st.info("""
            💡 **Sobre a Matriz de Correlação:**
            - Mostra a correlação entre as variáveis numéricas do conjunto de dados
            - Valores próximos a 1 indicam correlação positiva forte (azul escuro)
            - Valores próximos a -1 indicam correlação negativa forte (vermelho escuro)
            - Valores próximos a 0 indicam pouca ou nenhuma correlação (branco)
            """)
        else:
            st.warning("Não foram encontradas variáveis numéricas para calcular a correlação.")
        
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

    with tab1:
        st.subheader("📊 Visualizações e Análises dos Dados")
        st.write("""
        Esta seção apresenta visualizações interativas dos dados sobre autismo no Brasil.
        Use os filtros no menu lateral para personalizar sua análise.
        """)
        
        if regiao_filtro:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total de Pacientes", len(df_filtrado))
            with col2:
                media_consultas = round(df_filtrado["Numero_de_Consultas"].mean(), 2)
                st.metric("Média de Consultas", media_consultas)
            
            # Todos os gráficos anteriores aqui...
            
    with tab2:
        st.header("Predições com Modelos de Machine Learning")
        
        with st.expander("ℹ️ Informações sobre os Modelos", expanded=False):
            st.markdown("""
            ### Modelos de Predição
            
            1. **Número de Consultas Médicas**
               - Prevê quantas consultas um paciente realizará no ano
               - Usa Random Forest e Regressão Linear
            
            2. **Demanda Futura por Serviços**
               - Prevê a demanda total por região e ano
               - Considera tendências históricas
               - Usa Random Forest e Regressão Linear
            
            3. **Necessidade de Medicamentos**
               - Prevê a probabilidade de necessitar medicamentos
               - Usa Random Forest e Regressão Logística
            
            > 💡 **Nota:** Os modelos são atualizados periodicamente com novos dados
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            idade = st.number_input("Idade do Paciente", min_value=0, max_value=100, value=30)
            ano_diagnostico = st.number_input("Ano do Diagnóstico", min_value=2000, max_value=2030, value=2024)
            regiao = st.selectbox("Região", ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"])
        
        with col2:
            renda = st.selectbox("Renda Familiar", ["Baixa", "Média", "Alta"])
            acesso = st.selectbox("Acesso a Serviços", ["Sim", "Não"])
            zona = st.selectbox("Zona", ["Urbana", "Rural"])
            tratamento = st.selectbox("Tratamento", ["Terapias", "Medicamentos", "Ambos"])

        if st.button("Realizar Predições", type="primary"):
            with st.spinner("Realizando predições..."):
                try:
                    modelo = ModeloAutismo()
                    modelo.carregar_modelos()
                    
                    dados_entrada = {
                        "Idade": idade,
                        "Ano_Diagnostico": ano_diagnostico,
                        "Regiao": regiao,
                        "Renda_Familiar": renda,
                        "Acesso_a_Servicos": acesso,
                        "Zona": zona,
                        "Tratamento": tratamento
                    }
                    
                    resultados = modelo.fazer_predicoes(dados_entrada)
                    
                    # Criar três colunas para os resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.info("##### Consultas Previstas")
                        consultas_rf = resultados['consultas']['random_forest']
                        consultas_lr = resultados['consultas']['regressao_linear']
                        media_consultas = (consultas_rf + consultas_lr) / 2
                        st.metric("Média de Consultas/Ano", f"{media_consultas:.1f}")
                        st.caption(f"Random Forest: {consultas_rf:.1f}")
                        st.caption(f"Regressão Linear: {consultas_lr:.1f}")
                    
                    with col2:
                        st.info("##### Demanda Futura")
                        demanda_rf = resultados['demanda_futura']['random_forest']
                        demanda_lr = resultados['demanda_futura']['regressao_linear']
                        media_demanda = (demanda_rf + demanda_lr) / 2
                        st.metric("Demanda Total na Região", f"{media_demanda:.0f}")
                        st.caption(f"Random Forest: {demanda_rf:.0f}")
                        st.caption(f"Regressão Linear: {demanda_lr:.0f}")
                    
                    with col3:
                        st.info("##### Probabilidade de Medicamentos")
                        prob_rf = resultados['medicamentos']['random_forest']
                        prob_lr = resultados['medicamentos']['regressao_logistica']
                        media_prob = (prob_rf + prob_lr) / 2
                        st.metric("Probabilidade", f"{media_prob:.1%}")
                        st.caption(f"Random Forest: {prob_rf:.1%}")
                        st.caption(f"Regressão Logística: {prob_lr:.1%}")
                    
                    # Adicionar gráfico de tendência de demanda
                    if st.checkbox("Ver Tendência de Demanda"):
                        anos = list(range(ano_diagnostico, ano_diagnostico + 5))
                        demanda_futura = []
                        
                        for ano in anos:
                            dados_temp = dados_entrada.copy()
                            dados_temp['Ano_Diagnostico'] = ano
                            res = modelo.fazer_predicoes(dados_temp)
                            demanda_futura.append(
                                (res['demanda_futura']['random_forest'] + 
                                 res['demanda_futura']['regressao_linear']) / 2
                            )
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=anos,
                            y=demanda_futura,
                            mode='lines+markers',
                            name='Demanda Prevista'
                        ))
                        
                        plot_chart(
                            fig=fig,
                            title=f"Tendência de Demanda Futura - {regiao}",
                            xlabel="Ano",
                            ylabel="Demanda Total",
                            xangle=0
                        )
                    
                except Exception as e:
                    st.error(f"Erro ao realizar predições: {str(e)}")
    
    with tab3:
        st.subheader("📑 Dados Brutos")
        st.write("Os dados brutos são os dados originais que foram utilizados para a criação da base de dados.")
        
        # Adiciona opções de filtro para os dados brutos
        col1, col2 = st.columns(2)
        with col1:
            search = st.text_input("🔍 Buscar nos dados")
        with col2:
            n_rows = st.slider("Número de linhas", 5, 100, 10)
            
        if search:
            filtered_df = df_processado[df_processado.astype(str).apply(lambda x: x.str.contains(search, case=False)).any(axis=1)]
            st.dataframe(filtered_df.head(n_rows))
        else:
            st.dataframe(df_processado.head(n_rows))

if __name__ == "__main__":
    main()